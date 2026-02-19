"""
Node implementations for the DDP (Detect → Diagnose → Prescribe) pipeline.

Each node is a function that takes RetainAgentState and returns a partial
state update dict. Agent nodes implement an inner tool-calling loop.
"""

import json
import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..config import AgentConfig
from ..prompts import (
    DETECTION_AGENT_PROMPT,
    DIAGNOSIS_AGENT_PROMPT,
    PRESCRIPTION_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
)
from ..state import RetainAgentState
from ..tools import DETECTION_TOOLS, DIAGNOSIS_TOOLS, PRESCRIPTION_TOOLS
from ..utils import safe_json_serialize

logger = logging.getLogger("retain.agents.ddp")


# =============================================================================
# Helper: parse structured JSON from LLM response
# =============================================================================


def _extract_json_from_response(content: str) -> dict | None:
    """Extract JSON from an LLM response that may contain markdown fences.

    Args:
        content: LLM response text.

    Returns:
        Parsed dict, or None if no valid JSON found.
    """
    # Try parsing the whole content as JSON
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code fences
    import re

    json_blocks = re.findall(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
    for block in json_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # Try finding JSON-like content between { and }
    brace_match = re.search(r"\{.*\}", content, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _run_agent_loop(
    llm: ChatAnthropic,
    tools: list,
    messages: list,
    max_iterations: int = 15,
) -> list:
    """Run the tool-calling loop for an agent node.

    Args:
        llm: LLM with tools bound.
        tools: List of tool objects.
        messages: Current message list.
        max_iterations: Safety limit on tool-calling rounds.

    Returns:
        Updated message list.
    """
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages = [*messages, response]

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info(f"Calling tool: {tool_name}({list(tool_args.keys())})")

            if tool_name in tool_map:
                try:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_result = f"Tool error: {e}"
                    logger.error(f"Tool {tool_name} failed: {e}")
            else:
                tool_result = f"Unknown tool: {tool_name}"
                logger.warning(f"LLM called unknown tool: {tool_name}")

            messages = [
                *messages,
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                ),
            ]

    return messages


# =============================================================================
# Node Functions
# =============================================================================


def supervisor_node(state: dict) -> dict:
    """Supervisor node — routes phases and synthesizes results.

    Reads the current phase and either advances the pipeline or
    produces the final synthesis report.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with phase transition and/or messages.
    """
    config: AgentConfig = state["config"]
    current_phase = state["current_phase"]
    messages = list(state["messages"])

    logger.info(f"Supervisor: current_phase={current_phase}")

    match current_phase:
        case "idle":
            messages = [
                *messages,
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(
                    content=(
                        "Run the full churn analysis pipeline. "
                        f"Use reference_date={config.reference_date}. "
                        "Start with detection."
                    )
                ),
            ]
            return {
                "messages": messages,
                "current_phase": "detection",
                "metadata": {"reference_date": config.reference_date},
            }

        case "detection":
            detection = state.get("detection_results")
            if detection:
                summary = json.dumps(safe_json_serialize(detection), indent=2)
                messages = [
                    *messages,
                    HumanMessage(
                        content=(
                            f"Detection complete. Results:\n```json\n{summary}\n```\n\n"
                            "Proceed to diagnosis. Investigate the highest-risk cohort."
                        )
                    ),
                ]
            return {"messages": messages, "current_phase": "diagnosis"}

        case "diagnosis":
            diagnosis = state.get("diagnosis_results")
            if diagnosis:
                summary = json.dumps(safe_json_serialize(diagnosis), indent=2)
                messages = [
                    *messages,
                    HumanMessage(
                        content=(
                            f"Diagnosis complete. Results:\n```json\n{summary}\n```\n\n"
                            "Proceed to prescription. Recommend interventions."
                        )
                    ),
                ]
            return {"messages": messages, "current_phase": "prescription"}

        case "prescription":
            return {"messages": messages, "current_phase": "review"}

        case "review":
            feedback = state.get("human_feedback")
            if feedback and feedback.lower() == "reject":
                logger.info("Pipeline rejected by human reviewer")
                return {
                    "messages": messages,
                    "current_phase": "complete",
                    "metadata": {**state.get("metadata", {}), "status": "rejected"},
                }
            elif feedback and feedback.lower() != "approve":
                # Modification requested — send back to prescribe
                messages = [
                    *messages,
                    HumanMessage(
                        content=f"Revision requested: {feedback}. Please update the prescription."
                    ),
                ]
                return {"messages": messages, "current_phase": "prescription"}
            else:
                return {"messages": messages, "current_phase": "complete"}

        case "complete":
            # Synthesize final report
            detection = state.get("detection_results", {})
            diagnosis = state.get("diagnosis_results", {})
            prescription = state.get("prescription_results", {})

            llm = ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            synthesis_prompt = (
                "Synthesize a final churn analysis report from these results.\n\n"
                f"**Detection:**\n```json\n{json.dumps(safe_json_serialize(detection), indent=2)}\n```\n\n"
                f"**Diagnosis:**\n```json\n{json.dumps(safe_json_serialize(diagnosis), indent=2)}\n```\n\n"
                f"**Prescription:**\n```json\n{json.dumps(safe_json_serialize(prescription), indent=2)}\n```\n\n"
                "Format as a well-structured markdown report with sections: "
                "Executive Summary, Risk Overview, Root Cause Analysis, "
                "Recommended Interventions, and Next Steps."
            )

            messages = [
                *messages,
                HumanMessage(content=synthesis_prompt),
            ]

            response = llm.invoke(messages)
            messages = [*messages, response]

            logger.info("Supervisor: final report generated")
            return {
                "messages": messages,
                "metadata": {**state.get("metadata", {}), "status": "complete"},
            }

    return {"messages": messages}


def detection_node(state: dict) -> dict:
    """Detection node — runs batch scoring and cohort segmentation.

    Scores all active accounts, analyzes risk distribution, and segments
    high-risk accounts into cohorts by feature similarity.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with detection_results and messages.
    """
    config: AgentConfig = state["config"]
    messages = list(state["messages"])
    errors: list[str] = []

    logger.info("Detection node: starting")

    try:
        llm = ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Inject detection system prompt
        detection_messages = [
            SystemMessage(content=DETECTION_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"Run the full detection workflow using reference_date={config.reference_date}. "
                    f"Use min_cohort_size={config.min_cohort_size}. "
                    "Score all accounts, then segment high-risk ones into cohorts. "
                    "Return your results as the specified JSON format."
                )
            ),
        ]

        # Run agent loop with detection tools
        detection_messages = _run_agent_loop(
            llm, DETECTION_TOOLS, detection_messages
        )

        # Parse structured results from last assistant message
        last_content = detection_messages[-1].content
        detection_results = _extract_json_from_response(last_content)

        if detection_results is None:
            logger.warning("Could not parse detection results as JSON")
            detection_results = {
                "raw_response": last_content,
                "parse_error": "Could not extract structured JSON",
            }

    except Exception as e:
        logger.error(f"Detection node failed: {e}")
        errors.append(f"Detection failed: {e}")
        detection_results = {"error": str(e)}

    logger.info("Detection node: complete")
    messages = [*messages, HumanMessage(content="Detection phase complete.")]

    return {
        "messages": messages,
        "current_phase": "detection",
        "detection_results": detection_results,
        "errors": errors,
    }


def diagnosis_node(state: dict) -> dict:
    """Diagnosis node — investigates root causes for the highest-risk cohort.

    For sample accounts from the highest-risk cohort, runs SHAP explanations,
    pulls support/payment/viewing data, and identifies the primary root cause.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with diagnosis_results and messages.
    """
    config: AgentConfig = state["config"]
    messages = list(state["messages"])
    detection_results = state.get("detection_results", {})
    errors: list[str] = []

    logger.info("Diagnosis node: starting")

    try:
        llm = ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Build context from detection results
        detection_summary = json.dumps(
            safe_json_serialize(detection_results), indent=2
        )

        diagnosis_messages = [
            SystemMessage(content=DIAGNOSIS_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"Here are the detection results:\n```json\n{detection_summary}\n```\n\n"
                    "Investigate the highest-risk cohort. For 3-5 sample accounts, "
                    "run SHAP explanations, check support tickets, payment history, "
                    "and viewing behavior. Identify the primary root cause.\n\n"
                    "Return your results as the specified JSON format."
                )
            ),
        ]

        diagnosis_messages = _run_agent_loop(
            llm, DIAGNOSIS_TOOLS, diagnosis_messages
        )

        last_content = diagnosis_messages[-1].content
        diagnosis_results = _extract_json_from_response(last_content)

        if diagnosis_results is None:
            logger.warning("Could not parse diagnosis results as JSON")
            diagnosis_results = {
                "raw_response": last_content,
                "parse_error": "Could not extract structured JSON",
            }

    except Exception as e:
        logger.error(f"Diagnosis node failed: {e}")
        errors.append(f"Diagnosis failed: {e}")
        diagnosis_results = {"error": str(e)}

    logger.info("Diagnosis node: complete")
    messages = [*messages, HumanMessage(content="Diagnosis phase complete.")]

    return {
        "messages": messages,
        "current_phase": "diagnosis",
        "diagnosis_results": diagnosis_results,
        "errors": errors,
    }


def prescription_node(state: dict) -> dict:
    """Prescription node — recommends interventions and drafts emails.

    Maps the diagnosed root cause to intervention strategies, drafts
    email templates, and estimates business impact.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with prescription_results and messages.
    """
    config: AgentConfig = state["config"]
    messages = list(state["messages"])
    diagnosis_results = state.get("diagnosis_results", {})
    detection_results = state.get("detection_results", {})
    errors: list[str] = []

    logger.info("Prescription node: starting")

    try:
        llm = ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        diagnosis_summary = json.dumps(
            safe_json_serialize(diagnosis_results), indent=2
        )
        detection_summary = json.dumps(
            safe_json_serialize(detection_results), indent=2
        )

        prescription_messages = [
            SystemMessage(content=PRESCRIPTION_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"**Detection results:**\n```json\n{detection_summary}\n```\n\n"
                    f"**Diagnosis results:**\n```json\n{diagnosis_summary}\n```\n\n"
                    "Based on the diagnosed root cause, recommend an intervention strategy, "
                    "draft 2-3 email templates, and estimate the business impact.\n\n"
                    "Return your results as the specified JSON format."
                )
            ),
        ]

        prescription_messages = _run_agent_loop(
            llm, PRESCRIPTION_TOOLS, prescription_messages
        )

        last_content = prescription_messages[-1].content
        prescription_results = _extract_json_from_response(last_content)

        if prescription_results is None:
            logger.warning("Could not parse prescription results as JSON")
            prescription_results = {
                "raw_response": last_content,
                "parse_error": "Could not extract structured JSON",
            }

    except Exception as e:
        logger.error(f"Prescription node failed: {e}")
        errors.append(f"Prescription failed: {e}")
        prescription_results = {"error": str(e)}

    logger.info("Prescription node: complete")
    messages = [*messages, HumanMessage(content="Prescription phase complete.")]

    return {
        "messages": messages,
        "current_phase": "prescription",
        "prescription_results": prescription_results,
        "errors": errors,
    }


def review_node(state: dict) -> dict:
    """Review node — human-in-the-loop checkpoint.

    Presents prescription results and waits for human feedback.
    This node uses LangGraph's interrupt mechanism.

    Args:
        state: Current pipeline state.

    Returns:
        Partial state update with review phase.
    """
    logger.info("Review node: awaiting human feedback")

    prescription = state.get("prescription_results", {})
    feedback = state.get("human_feedback")

    messages = list(state["messages"])

    if feedback is None:
        # First pass — present results for review
        summary = json.dumps(safe_json_serialize(prescription), indent=2)
        messages = [
            *messages,
            HumanMessage(
                content=(
                    f"**Prescription for Review:**\n```json\n{summary}\n```\n\n"
                    "Please review and respond with 'approve', 'reject', "
                    "or provide modification instructions."
                )
            ),
        ]

    return {
        "messages": messages,
        "current_phase": "review",
    }


# =============================================================================
# Routing Function
# =============================================================================


def route_by_phase(state: dict) -> str:
    """Conditional edge that routes based on current_phase.

    Args:
        state: Current pipeline state.

    Returns:
        Name of the next node to execute.
    """
    phase = state.get("current_phase", "idle")

    match phase:
        case "idle":
            return "detect"
        case "detection":
            return "diagnose"
        case "diagnosis":
            return "prescribe"
        case "prescription":
            return "review"
        case "review":
            return "supervisor"
        case "complete":
            return "__end__"
        case _:
            logger.warning(f"Unknown phase: {phase}, routing to __end__")
            return "__end__"
