"""Analytics API endpoints."""

import logging

from fastapi import APIRouter, Depends

from ..dependencies import get_db_engine, get_demo_mode
from ..schemas import (
    AnalyticsOverview,
    DriftStatus,
    ModelMetrics,
    SHAPFeature,
    SegmentBreakdown,
    TrendPoint,
)
from ..services import analytics_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview", response_model=AnalyticsOverview)
async def get_overview(
    demo_mode: bool = Depends(get_demo_mode),
) -> AnalyticsOverview:
    """Get analytics overview bundle."""
    engine = None if demo_mode else get_db_engine()
    data = analytics_service.get_overview(engine, demo_mode)
    return AnalyticsOverview(**data)


@router.get("/churn-trends", response_model=list[TrendPoint])
async def get_churn_trends(
    months: int = 12,
    demo_mode: bool = Depends(get_demo_mode),
) -> list[TrendPoint]:
    """Get churn trend time series."""
    engine = None if demo_mode else get_db_engine()
    data = analytics_service.get_churn_trends(engine, demo_mode, months)
    return [TrendPoint(**t) for t in data]


@router.get("/segments", response_model=SegmentBreakdown)
async def get_segments(
    demo_mode: bool = Depends(get_demo_mode),
) -> SegmentBreakdown:
    """Get segment breakdowns."""
    engine = None if demo_mode else get_db_engine()
    data = analytics_service.get_segments(engine, demo_mode)
    return SegmentBreakdown(**data)


@router.get("/model-performance", response_model=ModelMetrics)
async def get_model_performance(
    demo_mode: bool = Depends(get_demo_mode),
) -> ModelMetrics:
    """Get model performance metrics."""
    engine = None if demo_mode else get_db_engine()
    data = analytics_service.get_model_performance(engine, demo_mode)
    return ModelMetrics(**data)


@router.get("/drift", response_model=DriftStatus)
async def get_drift_status(
    demo_mode: bool = Depends(get_demo_mode),
) -> DriftStatus:
    """Get drift monitoring status."""
    engine = None if demo_mode else get_db_engine()
    data = analytics_service.get_drift_status(engine, demo_mode)
    return DriftStatus(**data)


@router.get("/shap-global", response_model=list[SHAPFeature])
async def get_shap_global(
    demo_mode: bool = Depends(get_demo_mode),
) -> list[SHAPFeature]:
    """Get global SHAP feature importance."""
    engine = None if demo_mode else get_db_engine()
    data = analytics_service.get_shap_global(engine, demo_mode)
    return [SHAPFeature(**f) for f in data]
