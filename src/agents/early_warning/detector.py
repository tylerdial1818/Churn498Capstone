"""
Early Warning Agent — LangGraph agent for proactive churn risk detection.

Graph: START → score_comparison → investigate_escalations → group_and_report → END

Run with:
    python -m src.agents.early_warning.detector
    python -m src.agents.early_warning.detector --output json
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
from ..prompts import EARLY_WARNING_AGENT_PROMPT
from ..tools import (
    explain_account_prediction,
    get_account_payment_history,
    get_account_profile,
    get_account_support_history,
    get_account_viewing_summary,
    get_new_high_risk_accounts,
    get_risk_tier_transitions,
    query_database_readonly,
)
from ..utils import safe_json_serialize
from .alerts import (
    AlertGroup,
    EarlyWarningReport,
    RiskTransition,
    classify_transitions,
    compose_headline,
    compose_narrative,
    compute_alert_priority,
    format_alert_report_markdown,
    _recommended_action,
    VALID_ROOT_CAUSES,
)

logger = logging.getLogger("retain.agents.early_warning")


# =============================================================================
# State
# =============================================================================


class EarlyWarningState:
    """LangGraph state schema for the Early Warning Agent."""

    __annotations__ = {
        "messages": Annotated[list[BaseMessage], add_messages],
        "config": AgentConfig,
        "reference_date": str,
        "previous_date": str,
        "transitions": dict | None,
        "investigation_results": dict | None,
        "report": dict | None,
        "errors": Annotated[list[str], operator.add],
        "metadata": dict,
    }


# =============================================================================
# Node: score_comparison (deterministic — no LLM)
# =============================================================================


def score_comparison(state: dict) -> dict:
    """Compare risk tiers between two scoring runs.

    Calls tools directly (not through LLM) to get risk transitions
    and newly high-risk accounts. Parses into structured output.
    """
    config = state.get("config") or AgentConfig()
    ref_date = state.get("reference_date", config.reference_date)
    prev_date = state.get("previous_date", "2025-11-01")
    errors: list[str] = []

    logger.info(f"Score comparison: {prev_date} → {ref_date}")

    try:
        # Call tools directly
        transitions_raw = get_risk_tier_transitions.invoke({
            "current_date": ref_date,
            "previous_date": prev_date,
        })
        new_high_raw = get_new_high_risk_accounts.invoke({
            "current_date": ref_date,
            "previous_date": prev_date,
        })

        # Parse results
        transitions_data = json.loads(transitions_raw)
        new_high_data = json.loads(new_high_raw)

        # Build RiskTransition objects from escalated accounts
        escalated = transitions_data.get("escalated_accounts", [])
        improved = transitions_data.get("improved_accounts", [])

        transitions_output = {
            "total_scored": transitions_data.get("total_scored", 0),
            "escalated": escalated,
            "improved": improved,
            "stable_count": transitions_data.get("stable_count", 0),
            "new_high_risk_count": new_high_data.get(
                "total_new_high_risk", 0
            ),
        }

        return {
            "transitions": transitions_output,
            "errors": errors,
            "metadata": {
                "score_comparison_completed_at": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"score_comparison failed: {e}")
        errors.append(f"score_comparison error: {e}")
        return {
            "transitions": {
                "total_scored": 0,
                "escalated": [],
                "improved": [],
                "stable_count": 0,
                "new_high_risk_count": 0,
            },
            "errors": errors,
        }


# =============================================================================
# Node: investigate_escalations (LLM node)
# =============================================================================


def investigate_escalations(state: dict) -> dict:
    """Investigate escalated accounts using LLM with tools.

    For a sample of escalated accounts, uses SHAP and context tools
    to determine root causes and group by category.
    """
    config = state.get("config") or AgentConfig()
    transitions = state.get("transitions") or {}
    escalated = transitions.get("escalated", [])
    errors: list[str] = []

    # Short-circuit if no escalations
    if not escalated:
        logger.info("No escalations — skipping investigation")
        return {
            "investigation_results": {
                "groups": [],
                "ungrouped_count": 0,
            },
            "errors": errors,
        }

    # Select sample to investigate (up to shap_sample_size or 5)
    sample_size = min(len(escalated), config.shap_sample_size, 5)
    sample = escalated[:sample_size]
    sample_ids = [a["account_id"] for a in sample]

    logger.info(
        f"Investigating {sample_size} of {len(escalated)} escalated accounts"
    )

    # Build LLM
    llm = ChatAnthropic(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # Investigation tools
    tools = [
        explain_account_prediction,
        get_account_profile,
        get_account_support_history,
        get_account_payment_history,
        get_account_viewing_summary,
        query_database_readonly,
    ]

    # Prepare context message
    context = (
        f"The following {len(escalated)} accounts escalated to higher risk tiers.\n\n"
        f"Sample accounts to investigate (first {sample_size}):\n"
    )
    for a in sample:
        context += (
            f"- {a['account_id']}: {a['previous_tier']}→{a['current_tier']} "
            f"(delta: {a.get('probability_delta', a.get('delta', 'N/A'))})\n"
        )
    context += (
        f"\nTotal escalated: {len(escalated)}. "
        f"Investigate each sample account and group ALL escalated accounts "
        f"by root cause category."
    )

    messages: list = [
        SystemMessage(content=EARLY_WARNING_AGENT_PROMPT),
        HumanMessage(content=context),
    ]

    try:
        messages = _run_agent_loop(
            llm=llm,
            tools=tools,
            messages=messages,
        )

        # Extract JSON from final response
        final_content = messages[-1].content if messages else ""
        investigation = _extract_json_from_response(final_content)

        if investigation and "groups" in investigation:
            # Validate root causes
            for group in investigation["groups"]:
                if group.get("root_cause") not in VALID_ROOT_CAUSES:
                    group["root_cause"] = "disengagement"  # safe default

            return {
                "messages": messages,
                "investigation_results": investigation,
                "errors": errors,
            }
        else:
            errors.append("LLM did not return valid investigation JSON")
            # Build a default single group
            return {
                "messages": messages,
                "investigation_results": {
                    "groups": [{
                        "root_cause": "disengagement",
                        "account_ids": [a["account_id"] for a in escalated],
                        "representative_ids": sample_ids[:5],
                        "evidence_summary": (
                            "Investigation returned partial results. "
                            "Accounts grouped under disengagement as default."
                        ),
                        "common_features": {},
                    }],
                    "ungrouped_count": 0,
                },
                "errors": errors,
            }

    except Exception as e:
        logger.error(f"investigate_escalations failed: {e}")
        errors.append(f"investigate_escalations error: {e}")
        return {
            "investigation_results": {
                "groups": [{
                    "root_cause": "disengagement",
                    "account_ids": [a["account_id"] for a in escalated],
                    "representative_ids": sample_ids[:5],
                    "evidence_summary": f"Investigation failed: {e}",
                    "common_features": {},
                }],
                "ungrouped_count": 0,
            },
            "errors": errors,
        }


# =============================================================================
# Node: group_and_report (deterministic — no LLM)
# =============================================================================


def group_and_report(state: dict) -> dict:
    """Build the final EarlyWarningReport from transitions and investigation.

    Deterministic node — no LLM calls.
    """
    config = state.get("config") or AgentConfig()
    transitions = state.get("transitions") or {}
    investigation = state.get("investigation_results") or {"groups": []}
    errors: list[str] = []

    ref_date = state.get("reference_date", config.reference_date)
    prev_date = state.get("previous_date", "2025-11-01")

    escalated_raw = transitions.get("escalated", [])
    improved_raw = transitions.get("improved", [])

    # Build RiskTransition objects
    def _to_transition(raw: dict) -> RiskTransition:
        return RiskTransition(
            account_id=raw["account_id"],
            previous_tier=raw.get("previous_tier", "unknown"),
            current_tier=raw.get("current_tier", "unknown"),
            previous_probability=raw.get("previous_probability", 0.0),
            current_probability=raw.get("current_probability", 0.0),
            probability_delta=raw.get(
                "probability_delta", raw.get("delta", 0.0)
            ),
            direction=raw.get("direction", "escalated"),
        )

    escalated_transitions = [_to_transition(r) for r in escalated_raw]

    # Build AlertGroups from investigation results
    alert_groups: list[AlertGroup] = []
    investigation_groups = investigation.get("groups", [])
    all_escalated_ids = {t.account_id for t in escalated_transitions}

    for inv_group in investigation_groups:
        root_cause = inv_group.get("root_cause", "disengagement")
        group_account_ids = set(inv_group.get("account_ids", []))

        # Match transitions to this group
        group_transitions = [
            t for t in escalated_transitions
            if t.account_id in group_account_ids
        ]

        # If no transitions matched, assign from the full list
        if not group_transitions and escalated_transitions:
            group_transitions = escalated_transitions

        avg_delta = (
            sum(abs(t.probability_delta) for t in group_transitions)
            / max(len(group_transitions), 1)
        )

        rep_ids = inv_group.get(
            "representative_ids",
            [t.account_id for t in group_transitions[:5]],
        )

        priority = compute_alert_priority(
            len(group_transitions), avg_delta
        )

        alert_groups.append(AlertGroup(
            root_cause=root_cause,
            accounts=group_transitions,
            representative_account_ids=rep_ids[:5],
            evidence_summary=inv_group.get("evidence_summary", ""),
            priority=priority,
            recommended_action=_recommended_action(root_cause),
        ))

    # Count new high-risk
    total_new_high_risk = transitions.get("new_high_risk_count", 0)

    # Build the report
    report = EarlyWarningReport(
        reference_date=ref_date,
        previous_date=prev_date,
        total_accounts_scored=transitions.get("total_scored", 0),
        total_escalated=len(escalated_raw),
        total_new_high_risk=total_new_high_risk,
        total_improved=len(improved_raw),
        alert_groups=alert_groups,
        headline="",  # filled below
        narrative="",  # filled below
        model_health="Not checked in this run",
        metadata=state.get("metadata", {}),
        success=True,
        errors=list(state.get("errors", [])) + errors,
    )

    # Compose headline and narrative
    report.headline = compose_headline(
        report.total_escalated,
        report.total_new_high_risk,
        report.alert_groups,
    )
    report.narrative = compose_narrative(report)

    # Serialize for state
    report_dict = safe_json_serialize(report)

    return {
        "report": report_dict,
        "errors": errors,
        "metadata": {
            "report_completed_at": datetime.now().isoformat(),
        },
    }


# =============================================================================
# Graph Factory
# =============================================================================


def create_early_warning_agent(
    config: AgentConfig | None = None,
) -> Any:
    """Create the Early Warning Agent LangGraph.

    Args:
        config: Agent configuration. Uses defaults if None.

    Returns:
        Compiled StateGraph.
    """
    graph = StateGraph(EarlyWarningState)

    graph.add_node("score_comparison", score_comparison)
    graph.add_node("investigate_escalations", investigate_escalations)
    graph.add_node("group_and_report", group_and_report)

    graph.set_entry_point("score_comparison")
    graph.add_edge("score_comparison", "investigate_escalations")
    graph.add_edge("investigate_escalations", "group_and_report")
    graph.add_edge("group_and_report", END)

    return graph.compile()


# =============================================================================
# Convenience Function
# =============================================================================


def run_early_warning(
    reference_date: str = "2025-12-01",
    previous_date: str = "2025-11-01",
    config: AgentConfig | None = None,
    verbose: bool = False,
) -> EarlyWarningReport:
    """Run the Early Warning Agent end-to-end.

    Args:
        reference_date: Current scoring run date.
        previous_date: Previous scoring run date for comparison.
        config: Agent configuration.
        verbose: If True, log at DEBUG level.

    Returns:
        EarlyWarningReport with detection results.
    """
    if verbose:
        logging.getLogger("retain.agents.early_warning").setLevel(
            logging.DEBUG
        )

    config = config or AgentConfig()
    start_time = time.time()

    logger.info(
        f"Starting early warning: {previous_date} → {reference_date}"
    )

    graph = create_early_warning_agent(config)

    initial_state = {
        "messages": [],
        "config": config,
        "reference_date": reference_date,
        "previous_date": previous_date,
        "transitions": None,
        "investigation_results": None,
        "report": None,
        "errors": [],
        "metadata": {"started_at": datetime.now().isoformat()},
    }

    final_state = graph.invoke(initial_state)

    elapsed = round(time.time() - start_time, 2)
    report_dict = final_state.get("report", {})

    # Reconstruct EarlyWarningReport from dict
    alert_groups = []
    for g in report_dict.get("alert_groups", []):
        transitions = []
        for t in g.get("accounts", []):
            if isinstance(t, dict):
                transitions.append(RiskTransition(**t))
            else:
                transitions.append(t)

        alert_groups.append(AlertGroup(
            root_cause=g.get("root_cause", "disengagement"),
            accounts=transitions,
            representative_account_ids=g.get(
                "representative_account_ids", []
            ),
            evidence_summary=g.get("evidence_summary", ""),
            priority=g.get("priority", 1),
            recommended_action=g.get("recommended_action", ""),
        ))

    report = EarlyWarningReport(
        reference_date=report_dict.get("reference_date", reference_date),
        previous_date=report_dict.get("previous_date", previous_date),
        total_accounts_scored=report_dict.get("total_accounts_scored", 0),
        total_escalated=report_dict.get("total_escalated", 0),
        total_new_high_risk=report_dict.get("total_new_high_risk", 0),
        total_improved=report_dict.get("total_improved", 0),
        alert_groups=alert_groups,
        headline=report_dict.get("headline", ""),
        narrative=report_dict.get("narrative", ""),
        model_health=report_dict.get("model_health", ""),
        metadata={
            **report_dict.get("metadata", {}),
            "elapsed_seconds": elapsed,
        },
        success=report_dict.get("success", True),
        errors=report_dict.get("errors", []),
    )

    logger.info(f"Early warning complete in {elapsed}s: {report.headline}")

    return report


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run early warning churn detection"
    )
    parser.add_argument(
        "--reference-date", default="2025-12-01",
        help="Current scoring run date",
    )
    parser.add_argument(
        "--previous-date", default="2025-11-01",
        help="Previous scoring run date for comparison",
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

    report = run_early_warning(
        reference_date=args.reference_date,
        previous_date=args.previous_date,
        verbose=args.verbose,
    )

    if args.output == "json":
        print(json.dumps(safe_json_serialize(report), indent=2))
    else:
        print(format_alert_report_markdown(report))
