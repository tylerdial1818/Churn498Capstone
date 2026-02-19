"""
DDP (Detect → Diagnose → Prescribe) pipeline graph definition.

Assembles the LangGraph StateGraph from node functions defined in ddp_nodes.py.
Provides factory and convenience functions, plus a CLI entry point.

Usage:
    python -m src.agents.pipelines.ddp_pipeline
    python -m src.agents.pipelines.ddp_pipeline --verbose
"""

import argparse
import json
import logging
import sys

from langgraph.graph import END, StateGraph

from ..config import AgentConfig
from ..state import RetainAgentState, create_initial_state
from ..utils import safe_json_serialize
from .ddp_nodes import (
    detection_node,
    diagnosis_node,
    prescription_node,
    review_node,
    route_by_phase,
    supervisor_node,
)

logger = logging.getLogger("retain.agents.ddp")


def create_ddp_pipeline(
    config: AgentConfig | None = None,
) -> StateGraph:
    """Create and compile the DDP pipeline graph.

    Args:
        config: Agent configuration. Uses defaults if None.

    Returns:
        Compiled LangGraph StateGraph ready to invoke.
    """
    config = config or AgentConfig()

    # Build graph
    graph = StateGraph(RetainAgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("detect", detection_node)
    graph.add_node("diagnose", diagnosis_node)
    graph.add_node("prescribe", prescription_node)
    graph.add_node("review", review_node)

    # Set entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to the appropriate phase node
    graph.add_conditional_edges(
        "supervisor",
        route_by_phase,
        {
            "detect": "detect",
            "diagnose": "diagnose",
            "prescribe": "prescribe",
            "review": "review",
            "__end__": END,
        },
    )

    # All phase nodes return to supervisor
    graph.add_edge("detect", "supervisor")
    graph.add_edge("diagnose", "supervisor")
    graph.add_edge("prescribe", "supervisor")
    graph.add_edge("review", "supervisor")

    # Compile with interrupt_before review for human-in-the-loop
    compiled = graph.compile(interrupt_before=["review"])

    logger.info("DDP pipeline compiled successfully")
    return compiled


def run_ddp_pipeline(
    config: AgentConfig | None = None,
    verbose: bool = False,
) -> dict:
    """Run the full DDP pipeline end-to-end.

    Creates the graph, invokes with the standard initial message,
    and returns the final state.

    Args:
        config: Agent configuration.
        verbose: If True, log intermediate results.

    Returns:
        Final pipeline state dict.
    """
    config = config or AgentConfig()

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info("Starting DDP pipeline")

    graph = create_ddp_pipeline(config)
    initial_state = create_initial_state(config)

    # Run until interrupt (review node)
    state = graph.invoke(initial_state)

    # Auto-approve for non-interactive runs
    if state.get("current_phase") == "review":
        logger.info("Auto-approving prescription (non-interactive mode)")
        state["human_feedback"] = "approve"
        state = graph.invoke(state)

    logger.info("DDP pipeline complete")

    if verbose:
        _print_results(state)

    return state


def _print_results(state: dict) -> None:
    """Print pipeline results using rich formatting."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        console.print("\n")
        console.print(Panel("[bold green]DDP Pipeline Complete[/bold green]"))

        # Detection results
        detection = state.get("detection_results")
        if detection:
            console.print("\n[bold cyan]Detection Results:[/bold cyan]")
            console.print(json.dumps(safe_json_serialize(detection), indent=2))

        # Diagnosis results
        diagnosis = state.get("diagnosis_results")
        if diagnosis:
            console.print("\n[bold cyan]Diagnosis Results:[/bold cyan]")
            console.print(json.dumps(safe_json_serialize(diagnosis), indent=2))

        # Prescription results
        prescription = state.get("prescription_results")
        if prescription:
            console.print("\n[bold cyan]Prescription Results:[/bold cyan]")
            console.print(json.dumps(safe_json_serialize(prescription), indent=2))

        # Final report (last message content)
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content") and last.content:
                console.print("\n[bold cyan]Final Report:[/bold cyan]")
                console.print(last.content)

        # Errors
        errors = state.get("errors", [])
        if errors:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in errors:
                console.print(f"  • {error}")

    except ImportError:
        # Fallback without rich
        print("\n=== DDP Pipeline Results ===")
        print(json.dumps(safe_json_serialize(state), indent=2, default=str))


def main() -> int:
    """CLI entry point for the DDP pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Detect → Diagnose → Prescribe churn analysis pipeline",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging and print intermediate results",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default="2025-12-01",
        help="Reference date for analysis (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use",
    )

    args = parser.parse_args()

    config = AgentConfig(
        model_name=args.model,
        reference_date=args.reference_date,
    )

    try:
        from rich.console import Console
        from rich.logging import RichHandler

        console = Console()
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
    except ImportError:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
        )

    try:
        run_ddp_pipeline(config, verbose=True)
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
