"""
Shared LangGraph state schema for the Retain agent system.

Defines the TypedDict used across all agent nodes in the DDP pipeline.
Uses LangGraph's annotation-based reducers for message and error accumulation.
"""

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from .config import AgentConfig


class RetainAgentState:
    """
    LangGraph state schema for the Retain agent system.

    This is a TypedDict-compatible class used as the state schema for
    LangGraph's StateGraph. Fields with Annotated reducers accumulate
    values across node invocations.
    """

    __annotations__ = {
        # Conversation history — uses LangGraph's add_messages reducer
        "messages": Annotated[list[BaseMessage], add_messages],
        # Agent configuration
        "config": AgentConfig,
        # Current pipeline phase
        "current_phase": Literal[
            "idle",
            "detection",
            "diagnosis",
            "prescription",
            "review",
            "complete",
        ],
        # Node output slots
        "detection_results": dict | None,
        "diagnosis_results": dict | None,
        "prescription_results": dict | None,
        # Human-in-the-loop
        "human_feedback": str | None,
        # Error accumulation — uses operator.add reducer (appends lists)
        "errors": Annotated[list[str], operator.add],
        # Metadata — timing, model version, reference date, etc.
        "metadata": dict[str, Any],
    }


def create_initial_state(
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Create an initial state dict for the DDP pipeline.

    Args:
        config: Agent configuration. Uses defaults if None.

    Returns:
        Initial state dictionary matching RetainAgentState schema.
    """
    return {
        "messages": [],
        "config": config or AgentConfig(),
        "current_phase": "idle",
        "detection_results": None,
        "diagnosis_results": None,
        "prescription_results": None,
        "human_feedback": None,
        "errors": [],
        "metadata": {},
    }
