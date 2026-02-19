"""
Intervention Drafter — standalone email drafting agent.

A LangGraph agent that diagnoses churn drivers, selects strategies,
and drafts production-ready retention emails with A/B variants.

Usage:
    python -m src.agents.intervention.drafter --account-id ACC_00000001
    python -m src.agents.intervention.drafter --account-ids ACC_00000001 ACC_00000002
    python -m src.agents.intervention.drafter --account-id ACC_00000001 --churn-driver disengagement
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from ..config import AgentConfig
from ..prompts import INTERVENTION_DRAFTER_PROMPT
from ..tools import (
    explain_account_prediction,
    get_account_payment_history,
    get_account_profile,
    get_account_support_history,
    get_account_viewing_summary,
)
from ..utils import safe_json_serialize
from .strategies import InterventionStrategy, select_strategy

logger = logging.getLogger("retain.agents.intervention")


# =============================================================================
# Output Dataclasses
# =============================================================================


@dataclass
class DraftedEmail:
    """A drafted retention email."""

    variant: str  # "A" or "B"
    subject: str
    preview_text: str  # < 100 chars
    greeting: str
    body: str
    cta_text: str
    cta_url: str
    closing: str
    tone: str
    personalization_fields_used: list[str] = field(default_factory=list)


@dataclass
class InterventionResult:
    """Result of the intervention drafting process."""

    account_id: str | None
    account_ids: list[str] | None
    churn_driver: str
    strategy: InterventionStrategy
    emails: list[DraftedEmail]  # 2 variants
    account_context_summary: str
    confidence: float  # 0-1
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Drafter State
# =============================================================================


class DrafterState(TypedDict, total=False):
    """State schema for the intervention drafter graph."""

    account_id: str | None
    account_ids: list[str] | None
    churn_driver: str | None
    strategy: dict | None
    account_context: dict
    emails: list[dict]
    messages: list
    human_feedback: str | None
    revision_count: int
    config: AgentConfig


# =============================================================================
# Node Implementations
# =============================================================================


DIAGNOSIS_TOOLS = [
    get_account_profile,
    get_account_support_history,
    get_account_payment_history,
    get_account_viewing_summary,
    explain_account_prediction,
]


def diagnose_driver_node(state: dict) -> dict:
    """Diagnose the churn driver for the target account(s).

    If churn_driver is already provided, this node is a no-op.

    Args:
        state: Current drafter state.

    Returns:
        Partial state update with churn_driver and account_context.
    """
    if state.get("churn_driver"):
        logger.info("Churn driver pre-specified, skipping diagnosis")
        return {}

    config: AgentConfig = state["config"]
    account_id = state.get("account_id")
    account_ids = state.get("account_ids")

    # Determine which accounts to investigate
    target_ids = []
    if account_id:
        target_ids = [account_id]
    elif account_ids:
        target_ids = account_ids[:5]  # Sample up to 5

    logger.info(f"Diagnosing churn driver for {len(target_ids)} accounts")

    try:
        llm = ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        tool_map = {t.name: t for t in DIAGNOSIS_TOOLS}
        llm_with_tools = llm.bind_tools(DIAGNOSIS_TOOLS)

        messages = [
            SystemMessage(
                content=(
                    "You are diagnosing the churn driver for specific accounts. "
                    "Use the available tools to understand why these accounts are at risk. "
                    "Classify the primary driver as one of: payment_issues, disengagement, "
                    "support_frustration, price_sensitivity, content_gap, technical_issues.\n\n"
                    "After investigating, respond with a JSON object:\n"
                    '{"churn_driver": "<category>", "confidence": <0-1>, "context_summary": "<brief explanation>"}'
                )
            ),
            HumanMessage(
                content=f"Investigate these accounts: {', '.join(target_ids)}"
            ),
        ]

        # Tool-calling loop
        for _ in range(10):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                if tc["name"] in tool_map:
                    try:
                        result = tool_map[tc["name"]].invoke(tc["args"])
                    except Exception as e:
                        result = f"Tool error: {e}"
                else:
                    result = f"Unknown tool: {tc['name']}"

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        # Parse result
        from ..pipelines.ddp_nodes import _extract_json_from_response

        parsed = _extract_json_from_response(response.content)

        if parsed and "churn_driver" in parsed:
            return {
                "churn_driver": parsed["churn_driver"],
                "account_context": {
                    "summary": parsed.get("context_summary", ""),
                    "confidence": parsed.get("confidence", 0.5),
                },
                "messages": messages,
            }

        # Fallback
        return {
            "churn_driver": "disengagement",
            "account_context": {
                "summary": response.content,
                "confidence": 0.3,
            },
            "messages": messages,
        }

    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        return {
            "churn_driver": "disengagement",
            "account_context": {"error": str(e), "confidence": 0.1},
        }


def select_strategy_node(state: dict) -> dict:
    """Select intervention strategy — pure Python, no LLM.

    Args:
        state: Current drafter state.

    Returns:
        Partial state update with selected strategy.
    """
    churn_driver = state.get("churn_driver", "disengagement")
    account_context = state.get("account_context", {})

    strategy = select_strategy(churn_driver, account_context)

    logger.info(f"Selected strategy: {strategy.name} (driver: {churn_driver})")

    return {
        "strategy": {
            "id": strategy.id,
            "name": strategy.name,
            "churn_driver": strategy.churn_driver,
            "email_tone": strategy.email_tone,
            "typical_offer": strategy.typical_offer,
            "subject_line_templates": strategy.subject_line_templates,
            "cta_options": strategy.cta_options,
            "priority": strategy.priority,
        },
    }


def draft_email_node(state: dict) -> dict:
    """Draft retention emails using the LLM.

    Produces 2 A/B variants tailored to the account context and strategy.

    Args:
        state: Current drafter state.

    Returns:
        Partial state update with drafted emails.
    """
    config: AgentConfig = state["config"]
    strategy_dict = state.get("strategy", {})
    account_context = state.get("account_context", {})
    account_id = state.get("account_id")
    account_ids = state.get("account_ids")
    human_feedback = state.get("human_feedback")

    logger.info("Drafting email variants")

    try:
        llm = ChatAnthropic(
            model=config.model_name,
            temperature=0.4,  # Slightly higher for creative writing
            max_tokens=config.max_tokens,
        )

        context_text = json.dumps(safe_json_serialize(account_context), indent=2)
        strategy_text = json.dumps(safe_json_serialize(strategy_dict), indent=2)

        target_info = ""
        if account_id:
            target_info = f"Target: single account {account_id}"
        elif account_ids:
            target_info = f"Target: cohort of {len(account_ids)} accounts ({', '.join(account_ids[:3])}...)"

        revision_instruction = ""
        if human_feedback and human_feedback not in ("approve", "reject"):
            revision_instruction = f"\n\n**REVISION REQUESTED:** {human_feedback}"

        prompt = (
            f"{target_info}\n\n"
            f"**Strategy:**\n```json\n{strategy_text}\n```\n\n"
            f"**Account Context:**\n```json\n{context_text}\n```\n\n"
            "Draft 2 email variants (A and B) for A/B testing. "
            "Variants should differ in tone or angle, not just word choice.\n\n"
            "Return a JSON array of 2 objects, each with these fields:\n"
            "variant, subject, preview_text, greeting, body, cta_text, cta_url, "
            "closing, tone, personalization_fields_used"
            f"{revision_instruction}"
        )

        messages = [
            SystemMessage(content=INTERVENTION_DRAFTER_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)

        # Parse email variants
        from ..pipelines.ddp_nodes import _extract_json_from_response

        parsed = _extract_json_from_response(response.content)

        emails = []
        if isinstance(parsed, list):
            emails = parsed
        elif isinstance(parsed, dict) and "emails" in parsed:
            emails = parsed["emails"]
        elif isinstance(parsed, dict):
            # Single email, wrap in list
            emails = [parsed]

        return {"emails": emails, "messages": [*state.get("messages", []), response]}

    except Exception as e:
        logger.error(f"Email drafting failed: {e}")
        return {"emails": [], "messages": state.get("messages", [])}


def human_review_node(state: dict) -> dict:
    """Human review checkpoint.

    Presents drafted emails for approval, rejection, or revision.

    Args:
        state: Current drafter state.

    Returns:
        Partial state update.
    """
    logger.info("Awaiting human review of drafted emails")
    return {}


# =============================================================================
# Routing
# =============================================================================


def _should_skip_diagnosis(state: dict) -> str:
    """Route: skip diagnosis if churn_driver already provided."""
    if state.get("churn_driver"):
        return "select_strategy"
    return "diagnose_driver"


def _review_decision(state: dict) -> str:
    """Route: after review, end or revise."""
    feedback = state.get("human_feedback")
    revision_count = state.get("revision_count", 0)

    if feedback is None or feedback.lower() == "approve":
        return "__end__"
    elif feedback.lower() == "reject":
        return "__end__"
    elif revision_count >= 3:
        logger.info("Max revision cycles reached, auto-completing")
        return "__end__"
    else:
        return "draft_email"


# =============================================================================
# Graph Factory
# =============================================================================


def create_intervention_drafter(
    config: AgentConfig | None = None,
) -> StateGraph:
    """Create and compile the intervention drafter graph.

    Args:
        config: Agent configuration.

    Returns:
        Compiled LangGraph StateGraph.
    """
    config = config or AgentConfig()

    graph = StateGraph(DrafterState)

    graph.add_node("diagnose_driver", diagnose_driver_node)
    graph.add_node("select_strategy", select_strategy_node)
    graph.add_node("draft_email", draft_email_node)
    graph.add_node("human_review", human_review_node)

    # Entry: skip diagnosis if driver provided
    graph.set_conditional_entry_point(
        _should_skip_diagnosis,
        {
            "diagnose_driver": "diagnose_driver",
            "select_strategy": "select_strategy",
        },
    )

    graph.add_edge("diagnose_driver", "select_strategy")
    graph.add_edge("select_strategy", "draft_email")
    graph.add_edge("draft_email", "human_review")

    graph.add_conditional_edges(
        "human_review",
        _review_decision,
        {
            "draft_email": "draft_email",
            "__end__": END,
        },
    )

    compiled = graph.compile(interrupt_before=["human_review"])

    logger.info("Intervention drafter compiled successfully")
    return compiled


# =============================================================================
# Convenience Function
# =============================================================================


def draft_intervention(
    account_id: str | None = None,
    account_ids: list[str] | None = None,
    churn_driver: str | None = None,
    config: AgentConfig | None = None,
) -> InterventionResult:
    """Draft a retention intervention for an account or cohort.

    Args:
        account_id: Single account to target (mutually exclusive with account_ids).
        account_ids: List of accounts to target (mutually exclusive with account_id).
        churn_driver: Pre-specified churn driver (skips diagnosis if provided).
        config: Agent configuration.

    Returns:
        InterventionResult with strategy and drafted emails.
    """
    if account_id and account_ids:
        raise ValueError("Specify account_id or account_ids, not both")
    if not account_id and not account_ids:
        raise ValueError("Must specify either account_id or account_ids")

    config = config or AgentConfig()

    graph = create_intervention_drafter(config)

    initial_state = {
        "account_id": account_id,
        "account_ids": account_ids,
        "churn_driver": churn_driver,
        "strategy": None,
        "account_context": {},
        "emails": [],
        "messages": [],
        "human_feedback": None,
        "revision_count": 0,
        "config": config,
    }

    # Run until interrupt
    state = graph.invoke(initial_state)

    # Auto-approve for non-interactive mode
    if state.get("emails"):
        state["human_feedback"] = "approve"
        state = graph.invoke(state)

    # Build result
    strategy_dict = state.get("strategy", {})
    from .strategies import STRATEGY_REGISTRY

    strategy = STRATEGY_REGISTRY.get(
        strategy_dict.get("id", "engagement_reignite"),
        STRATEGY_REGISTRY["engagement_reignite"],
    )

    emails = []
    for email_dict in state.get("emails", []):
        if isinstance(email_dict, dict):
            emails.append(
                DraftedEmail(
                    variant=email_dict.get("variant", "A"),
                    subject=email_dict.get("subject", ""),
                    preview_text=email_dict.get("preview_text", ""),
                    greeting=email_dict.get("greeting", ""),
                    body=email_dict.get("body", ""),
                    cta_text=email_dict.get("cta_text", ""),
                    cta_url=email_dict.get("cta_url", ""),
                    closing=email_dict.get("closing", ""),
                    tone=email_dict.get("tone", strategy.email_tone),
                    personalization_fields_used=email_dict.get(
                        "personalization_fields_used", []
                    ),
                )
            )

    context = state.get("account_context", {})

    return InterventionResult(
        account_id=account_id,
        account_ids=account_ids,
        churn_driver=state.get("churn_driver", "unknown"),
        strategy=strategy,
        emails=emails,
        account_context_summary=context.get("summary", ""),
        confidence=context.get("confidence", 0.5),
        metadata={
            "revision_count": state.get("revision_count", 0),
            "reference_date": config.reference_date,
        },
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """CLI entry point for the intervention drafter."""
    parser = argparse.ArgumentParser(
        description="Draft retention intervention emails",
    )
    parser.add_argument("--account-id", type=str, help="Single account ID")
    parser.add_argument(
        "--account-ids", nargs="+", type=str, help="Multiple account IDs"
    )
    parser.add_argument(
        "--churn-driver",
        type=str,
        choices=[
            "payment_issues", "disengagement", "support_frustration",
            "price_sensitivity", "content_gap", "technical_issues",
        ],
        help="Pre-specified churn driver (skips diagnosis)",
    )
    parser.add_argument(
        "--output",
        choices=["markdown", "html", "plaintext"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.account_id and not args.account_ids:
        parser.error("Must specify --account-id or --account-ids")

    # Set up logging
    try:
        from rich.console import Console
        from rich.logging import RichHandler

        console = Console()
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
    except ImportError:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
        )

    try:
        result = draft_intervention(
            account_id=args.account_id,
            account_ids=args.account_ids,
            churn_driver=args.churn_driver,
        )

        # Render output
        from .email_renderer import (
            render_as_html,
            render_as_markdown,
            render_as_plaintext,
            render_comparison,
        )

        if result.emails:
            if args.output == "html":
                for email in result.emails:
                    print(render_as_html(email))
                    print("\n---\n")
            elif args.output == "plaintext":
                for email in result.emails:
                    print(render_as_plaintext(email))
                    print("\n---\n")
            else:
                print(render_comparison(result.emails))

            print(f"\nStrategy: {result.strategy.name}")
            print(f"Churn driver: {result.churn_driver}")
            print(f"Confidence: {result.confidence:.0%}")
        else:
            print("No emails were drafted.")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Drafter failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
