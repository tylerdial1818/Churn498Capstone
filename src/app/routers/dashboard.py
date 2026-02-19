"""Dashboard API endpoints."""

import logging

from fastapi import APIRouter, Depends

from ..dependencies import get_db_engine, get_demo_mode
from ..schemas import (
    ActiveInactiveDistribution,
    AgentInsight,
    KPIResponse,
    RiskDistribution,
    TrendPoint,
)
from ..services import dashboard_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/kpis", response_model=KPIResponse)
async def get_kpis(
    demo_mode: bool = Depends(get_demo_mode),
) -> KPIResponse:
    """Get dashboard KPIs."""
    engine = None if demo_mode else get_db_engine()
    data = dashboard_service.get_kpis(engine, demo_mode)
    return KPIResponse(**data)


@router.get("/trends", response_model=list[TrendPoint])
async def get_trends(
    demo_mode: bool = Depends(get_demo_mode),
) -> list[TrendPoint]:
    """Get monthly trend data."""
    engine = None if demo_mode else get_db_engine()
    data = dashboard_service.get_trends(engine, demo_mode)
    return [TrendPoint(**t) for t in data]


@router.get("/risk-distribution", response_model=RiskDistribution)
async def get_risk_distribution(
    demo_mode: bool = Depends(get_demo_mode),
) -> RiskDistribution:
    """Get risk tier distribution."""
    engine = None if demo_mode else get_db_engine()
    data = dashboard_service.get_risk_distribution(engine, demo_mode)
    return RiskDistribution(**data)


@router.get("/active-inactive", response_model=ActiveInactiveDistribution)
async def get_active_inactive(
    demo_mode: bool = Depends(get_demo_mode),
) -> ActiveInactiveDistribution:
    """Get active vs inactive subscriber distribution."""
    engine = None if demo_mode else get_db_engine()
    data = dashboard_service.get_active_inactive(engine, demo_mode)
    return ActiveInactiveDistribution(**data)


@router.get("/executive-summary", response_model=AgentInsight)
async def get_executive_summary(
    demo_mode: bool = Depends(get_demo_mode),
) -> AgentInsight:
    """Get AI executive summary."""
    engine = None if demo_mode else get_db_engine()
    data = dashboard_service.get_executive_summary(engine, demo_mode)
    return AgentInsight(**data)
