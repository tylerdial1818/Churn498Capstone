"""Prescription API endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_db_engine, get_demo_mode
from ..schemas import PrescriptionGroup
from ..services import prescription_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prescriptions", tags=["Prescriptions"])


@router.get("", response_model=list[PrescriptionGroup])
async def get_prescriptions(
    demo_mode: bool = Depends(get_demo_mode),
) -> list[PrescriptionGroup]:
    """Get all prescription groups."""
    engine = None if demo_mode else get_db_engine()
    data = prescription_service.get_prescriptions(engine, demo_mode)
    return [PrescriptionGroup(**g) for g in data]


@router.get("/{strategy}", response_model=PrescriptionGroup)
async def get_prescription_by_strategy(
    strategy: str,
    demo_mode: bool = Depends(get_demo_mode),
) -> PrescriptionGroup:
    """Get a single prescription group by strategy."""
    engine = None if demo_mode else get_db_engine()
    data = prescription_service.get_prescription_by_strategy(
        engine, strategy, demo_mode
    )
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy}' not found",
        )
    return PrescriptionGroup(**data)
