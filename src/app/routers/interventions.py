"""Intervention API endpoints — CRUD for drafted retention emails."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_db_engine, get_demo_mode
from ..schemas import (
    BatchDraftRequest,
    ContentUpdateRequest,
    DraftRequest,
    InterventionDraft,
    StatusUpdateRequest,
)
from ..services import intervention_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interventions", tags=["Interventions"])


@router.get("", response_model=list[InterventionDraft])
async def get_interventions(
    status: str | None = None,
    demo_mode: bool = Depends(get_demo_mode),
) -> list[InterventionDraft]:
    """Get intervention drafts, optionally filtered by status."""
    engine = None if demo_mode else get_db_engine()
    data = intervention_service.get_interventions(engine, demo_mode, status)
    return [InterventionDraft(**d) for d in data]


@router.get("/{intervention_id}", response_model=InterventionDraft)
async def get_intervention(
    intervention_id: str,
    demo_mode: bool = Depends(get_demo_mode),
) -> InterventionDraft:
    """Get a single intervention by ID."""
    engine = None if demo_mode else get_db_engine()
    data = intervention_service.get_intervention(
        engine, intervention_id, demo_mode
    )
    if data is None:
        raise HTTPException(status_code=404, detail="Intervention not found")
    return InterventionDraft(**data)


@router.post("/draft", response_model=InterventionDraft)
async def draft_intervention(
    request: DraftRequest,
    demo_mode: bool = Depends(get_demo_mode),
) -> InterventionDraft:
    """Create a new intervention draft for an account."""
    engine = None if demo_mode else get_db_engine()
    data = intervention_service.create_intervention(
        engine, request.account_id, request.strategy, demo_mode
    )
    return InterventionDraft(**data)


@router.post("/batch-draft", response_model=list[InterventionDraft])
async def batch_draft_interventions(
    request: BatchDraftRequest,
    demo_mode: bool = Depends(get_demo_mode),
) -> list[InterventionDraft]:
    """Create intervention drafts for multiple accounts."""
    engine = None if demo_mode else get_db_engine()
    data = intervention_service.batch_create_interventions(
        engine, request.account_ids, request.strategy, demo_mode
    )
    return [InterventionDraft(**d) for d in data]


@router.patch("/{intervention_id}/status", response_model=InterventionDraft)
async def update_status(
    intervention_id: str,
    request: StatusUpdateRequest,
    demo_mode: bool = Depends(get_demo_mode),
) -> InterventionDraft:
    """Update intervention status (approve/reject)."""
    engine = None if demo_mode else get_db_engine()
    data = intervention_service.update_intervention_status(
        engine, intervention_id, request.status, demo_mode
    )
    if data is None:
        raise HTTPException(status_code=404, detail="Intervention not found")
    return InterventionDraft(**data)


@router.patch("/{intervention_id}/content", response_model=InterventionDraft)
async def update_content(
    intervention_id: str,
    request: ContentUpdateRequest,
    demo_mode: bool = Depends(get_demo_mode),
) -> InterventionDraft:
    """Update intervention email content."""
    engine = None if demo_mode else get_db_engine()
    data = intervention_service.update_intervention_content(
        engine, intervention_id, request.subject, request.body, demo_mode
    )
    if data is None:
        raise HTTPException(status_code=404, detail="Intervention not found")
    return InterventionDraft(**data)
