"""
Analysis Agent — LangGraph agent for dashboard narration.

Graph: START → gather_data → generate_narrative → END

Run with:
    python -m src.agents.analysis.analyzer --page executive_summary
    python -m src.agents.analysis.analyzer --page analytics_deep_dive --output json
"""

import argparse
import json
import logging
import operator
import time
from datetime import datetime
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from ..config import AgentConfig
from ..pipelines.ddp_nodes import _extract_json_from_response, _run_agent_loop
from ..prompts import ANALYSIS_AGENT_PROMPT
from ..tools import (
    get_account_risk_scores,
    get_churn_cohort_analysis,
    get_engagement_trends,
    get_executive_kpis,
    get_feature_importance_global,
    get_model_health_status,
    get_payment_health,
    get_subscription_distribution,
    get_support_health,
    segment_high_risk_accounts,
)
from ..utils import safe_json_serialize
from .narratives import (
    AnalysisNarrative,
    NarrativeSection,
    PageContext,
    assess_kpi_health,
    compute_overall_sentiment,
    format_narrative_markdown,
)

logger = logging.getLogger("retain.agents.analysis")


# =============================================================================
# State
# =============================================================================


class AnalysisState:
    """LangGraph state schema for the Analysis Agent."""

    __annotations__ = {
        "messages": Annotated[list[BaseMessage], add_messages],
        "config": AgentConfig,
        "page_context": str,
        "raw_data": dict | None,
        "narrative": dict | None,
        "errors": Annotated[list[str], operator.add],
        "metadata": dict,
    }


# =============================================================================
# Tool sets per page context
# =============================================================================

# Maps page_context to (tool_function, kwargs) tuples
TOOL_SETS: dict[str, list[tuple]] = {
    "executive_summary": [
        (get_executive_kpis, {}),
        (get_account_risk_scores, {"risk_tier": "high"}),
        (get_model_health_status, {}),
    ],
    "analytics_deep_dive": [
        (get_executive_kpis, {}),
        (get_subscription_distribution, {}),
        (get_engagement_trends, {}),
        (get_support_health, {}),
        (get_payment_health, {}),
        (get_churn_cohort_analysis, {}),
        (get_model_health_status, {}),
    ],
    "at_risk_detail": [
        (get_account_risk_scores, {"risk_tier": "high"}),
        (get_subscription_distribution, {}),
        (get_executive_kpis, {}),
        (segment_high_risk_accounts, {}),
    ],
    "prescription_summary": [
        (get_account_risk_scores, {"risk_tier": "high"}),
        (get_executive_kpis, {}),
        (get_feature_importance_global, {}),
    ],
}


# =============================================================================
# Node: gather_data (deterministic — no LLM)
# =============================================================================


def gather_data(state: dict) -> dict:
    """Gather data by calling appropriate tools per page context.

    Deterministic node — calls tools directly, no LLM.
    """
    config = state.get("config") or AgentConfig()
    page_context = state.get("page_context", "executive_summary")
    ref_date = config.reference_date
    errors: list[str] = []

    logger.info(f"Gathering data for page_context={page_context}")

    tool_set = TOOL_SETS.get(page_context, TOOL_SETS["executive_summary"])

    # Collect raw data from each tool
    supplementary_data: dict[str, Any] = {}
    kpis: dict[str, dict] = {}

    for tool_fn, kwargs in tool_set:
        tool_name = tool_fn.name
        try:
            # Add reference_date where applicable
            call_kwargs = {**kwargs}
            if "reference_date" in tool_fn.args_schema.model_fields:
                call_kwargs.setdefault("reference_date", ref_date)

            result_str = tool_fn.invoke(call_kwargs)

            # Try to parse as JSON
            try:
                parsed = json.loads(result_str)
                supplementary_data[tool_name] = parsed
            except (json.JSONDecodeError, TypeError):
                supplementary_data[tool_name] = result_str

        except Exception as e:
            logger.warning(f"Tool {tool_name} failed: {e}")
            errors.append(f"{tool_name}: {e}")
            supplementary_data[tool_name] = f"Error: {e}"

    # Extract KPIs from executive_kpis tool result
    exec_kpis = supplementary_data.get("get_executive_kpis", {})
    if isinstance(exec_kpis, dict):
        for key in [
            "churn_rate_30d", "retention_rate_30d", "high_risk_count",
            "monthly_recurring_revenue", "avg_customer_lifetime_days",
            "total_active_accounts", "total_churned_accounts",
            "avg_churn_probability",
        ]:
            if key in exec_kpis:
                value = exec_kpis[key]
                health = assess_kpi_health(key, float(value))
                kpis[key] = {"value": value, "health": health}

    # Extract payment failure rate if available
    payment_data = supplementary_data.get("get_payment_health", {})
    if isinstance(payment_data, dict) and "failure_rate" in payment_data:
        val = payment_data["failure_rate"]
        kpis["payment_failure_rate"] = {
            "value": val,
            "health": assess_kpi_health("payment_failure_rate", float(val)),
        }

    # Extract support resolution time if available
    support_data = supplementary_data.get("get_support_health", {})
    if isinstance(support_data, dict) and "avg_resolution_hours" in support_data:
        val = support_data["avg_resolution_hours"]
        kpis["avg_resolution_hours"] = {
            "value": val,
            "health": assess_kpi_health("avg_resolution_hours", float(val)),
        }

    raw_data = {
        "page_context": page_context,
        "kpis": kpis,
        "supplementary_data": supplementary_data,
        "data_collection_errors": errors,
    }

    return {
        "raw_data": raw_data,
        "errors": errors,
        "metadata": {
            "data_gathered_at": datetime.now().isoformat(),
        },
    }


# =============================================================================
# Node: generate_narrative (LLM node)
# =============================================================================


def generate_narrative(state: dict) -> dict:
    """Generate natural language narrative using LLM.

    No tools bound — the LLM only generates text from pre-gathered data.
    """
    config = state.get("config") or AgentConfig()
    raw_data = state.get("raw_data") or {}
    errors: list[str] = []

    logger.info("Generating narrative via LLM")

    llm = ChatAnthropic(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # Build the data context message
    data_json = json.dumps(safe_json_serialize(raw_data), indent=2)

    messages: list = [
        SystemMessage(content=ANALYSIS_AGENT_PROMPT),
        HumanMessage(
            content=(
                f"Generate a narrative for the '{raw_data.get('page_context', 'executive_summary')}' "
                f"page. Here is the raw data:\n\n```json\n{data_json}\n```"
            )
        ),
    ]

    try:
        # No tools — just generate text
        response = llm.invoke(messages)
        messages.append(response)

        final_content = response.content if response else ""
        narrative_dict = _extract_json_from_response(final_content)

        if narrative_dict:
            # Ensure raw_kpis is populated
            if "raw_kpis" not in narrative_dict or not narrative_dict["raw_kpis"]:
                narrative_dict["raw_kpis"] = {
                    k: v.get("value", v) if isinstance(v, dict) else v
                    for k, v in raw_data.get("kpis", {}).items()
                }

            # Enforce max 4 key callouts
            callouts = narrative_dict.get("key_callouts", [])
            narrative_dict["key_callouts"] = callouts[:4]

            return {
                "messages": messages,
                "narrative": narrative_dict,
                "errors": errors,
            }
        else:
            errors.append("LLM did not return valid narrative JSON")
            # Build a fallback narrative
            kpis = raw_data.get("kpis", {})
            fallback = {
                "headline": "Dashboard analysis generated with partial results.",
                "sections": [{
                    "heading": "Summary",
                    "body": final_content[:500] if final_content else "Analysis unavailable.",
                    "kpis_referenced": list(kpis.keys()),
                    "sentiment": "neutral",
                }],
                "overall_sentiment": "healthy",
                "key_callouts": [],
                "raw_kpis": {
                    k: v.get("value", v) if isinstance(v, dict) else v
                    for k, v in kpis.items()
                },
            }
            return {
                "messages": messages,
                "narrative": fallback,
                "errors": errors,
            }

    except Exception as e:
        logger.error(f"generate_narrative failed: {e}")
        errors.append(f"generate_narrative error: {e}")
        kpis = raw_data.get("kpis", {})
        return {
            "narrative": {
                "headline": "Analysis unavailable due to error.",
                "sections": [],
                "overall_sentiment": "healthy",
                "key_callouts": [],
                "raw_kpis": {
                    k: v.get("value", v) if isinstance(v, dict) else v
                    for k, v in kpis.items()
                },
            },
            "errors": errors,
        }


# =============================================================================
# Graph Factory
# =============================================================================


def create_analysis_agent(
    config: AgentConfig | None = None,
) -> Any:
    """Create the Analysis Agent LangGraph.

    Args:
        config: Agent configuration. Uses defaults if None.

    Returns:
        Compiled StateGraph.
    """
    graph = StateGraph(AnalysisState)

    graph.add_node("gather_data", gather_data)
    graph.add_node("generate_narrative", generate_narrative)

    graph.set_entry_point("gather_data")
    graph.add_edge("gather_data", "generate_narrative")
    graph.add_edge("generate_narrative", END)

    return graph.compile()


# =============================================================================
# Convenience Function
# =============================================================================


def run_analysis(
    page_context: str = "executive_summary",
    reference_date: str = "2025-12-01",
    config: AgentConfig | None = None,
    verbose: bool = False,
) -> AnalysisNarrative:
    """Run the Analysis Agent end-to-end.

    Args:
        page_context: Which page is requesting analysis.
        reference_date: Reference date for analysis.
        config: Agent configuration.
        verbose: If True, log at DEBUG level.

    Returns:
        AnalysisNarrative with results.

    Raises:
        ValueError: If page_context is invalid.
    """
    # Validate page context
    valid_contexts = {pc.value for pc in PageContext}
    if page_context not in valid_contexts:
        raise ValueError(
            f"Invalid page_context '{page_context}'. "
            f"Must be one of: {valid_contexts}"
        )

    if verbose:
        logging.getLogger("retain.agents.analysis").setLevel(logging.DEBUG)

    config = config or AgentConfig()
    config_with_date = AgentConfig(
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        reference_date=reference_date,
        high_risk_threshold=config.high_risk_threshold,
        medium_risk_threshold=config.medium_risk_threshold,
        min_cohort_size=config.min_cohort_size,
        max_cohorts=config.max_cohorts,
        top_features_count=config.top_features_count,
        shap_sample_size=config.shap_sample_size,
        max_sql_rows=config.max_sql_rows,
        read_only_sql=config.read_only_sql,
    )

    start_time = time.time()
    logger.info(f"Starting analysis: page={page_context}, date={reference_date}")

    graph = create_analysis_agent(config_with_date)

    initial_state = {
        "messages": [],
        "config": config_with_date,
        "page_context": page_context,
        "raw_data": None,
        "narrative": None,
        "errors": [],
        "metadata": {"started_at": datetime.now().isoformat()},
    }

    final_state = graph.invoke(initial_state)

    elapsed = round(time.time() - start_time, 2)
    narrative_dict = final_state.get("narrative", {})
    raw_data = final_state.get("raw_data", {})

    # Build NarrativeSection objects
    sections = []
    for s in narrative_dict.get("sections", []):
        sections.append(NarrativeSection(
            heading=s.get("heading", ""),
            body=s.get("body", ""),
            kpis_referenced=s.get("kpis_referenced", []),
            sentiment=s.get("sentiment", "neutral"),
            data_points=s.get("data_points", {}),
        ))

    # Compute overall sentiment from sections
    section_sentiments = [s.sentiment for s in sections]
    overall = narrative_dict.get(
        "overall_sentiment",
        compute_overall_sentiment(section_sentiments),
    )

    narrative = AnalysisNarrative(
        page_context=page_context,
        generated_at=datetime.now().isoformat(),
        reference_date=reference_date,
        headline=narrative_dict.get("headline", ""),
        sections=sections,
        overall_sentiment=overall,
        key_callouts=narrative_dict.get("key_callouts", [])[:4],
        raw_kpis=narrative_dict.get("raw_kpis", {}),
        metadata={
            **final_state.get("metadata", {}),
            "elapsed_seconds": elapsed,
        },
        success=True,
        errors=list(final_state.get("errors", [])),
    )

    logger.info(f"Analysis complete in {elapsed}s: {narrative.headline}")

    return narrative


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dashboard analysis narrative"
    )
    parser.add_argument(
        "--page",
        choices=[
            "executive_summary", "analytics_deep_dive",
            "at_risk_detail", "prescription_summary",
        ],
        default="executive_summary",
        help="Page context for the narrative",
    )
    parser.add_argument(
        "--reference-date", default="2025-12-01",
        help="Reference date for analysis",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output", choices=["markdown", "json"], default="markdown",
        help="Output format",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    narrative = run_analysis(
        page_context=args.page,
        reference_date=args.reference_date,
        verbose=args.verbose,
    )

    if args.output == "json":
        print(json.dumps(safe_json_serialize(narrative), indent=2))
    else:
        print(format_narrative_markdown(narrative))
