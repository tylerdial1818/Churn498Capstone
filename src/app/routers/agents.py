"""Agent API endpoints — trigger and monitor pipeline runs."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_demo_mode
from ..schemas import AgentStatusResponse, AgentTriggerResponse
from ..services import agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.post("/trigger/{pipeline}", response_model=AgentTriggerResponse)
async def trigger_agent(
    pipeline: str,
    demo_mode: bool = Depends(get_demo_mode),
) -> AgentTriggerResponse:
    """Trigger an agent pipeline run."""
    data = agent_service.trigger_agent_run(pipeline, demo_mode=demo_mode)
    return AgentTriggerResponse(**data)


@router.get("/status/{run_id}", response_model=AgentStatusResponse)
async def get_agent_status(
    run_id: str,
) -> AgentStatusResponse:
    """Check status of an agent run."""
    data = agent_service.get_agent_status(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentStatusResponse(**data)
