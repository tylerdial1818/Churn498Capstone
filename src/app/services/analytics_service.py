"""
Analytics service — overview, trends, segments, model performance, drift.
"""

import logging

from sqlalchemy import Engine

from ..fixtures.demo_data import (
    get_fixture_analytics_overview,
    get_fixture_churn_trends,
    get_fixture_drift_status,
    get_fixture_model_performance,
    get_fixture_segments,
    get_fixture_shap_global,
)

logger = logging.getLogger(__name__)


def get_overview(engine: Engine | None, demo_mode: bool = False) -> dict:
    """Get analytics overview bundle."""
    if demo_mode or engine is None:
        return get_fixture_analytics_overview()
    return get_fixture_analytics_overview()


def get_churn_trends(
    engine: Engine | None,
    demo_mode: bool = False,
    months: int = 12,
) -> list[dict]:
    """Get churn trend time series."""
    if demo_mode or engine is None:
        return get_fixture_churn_trends(months)
    return get_fixture_churn_trends(months)


def get_segments(engine: Engine | None, demo_mode: bool = False) -> dict:
    """Get segment breakdowns."""
    if demo_mode or engine is None:
        return get_fixture_segments()
    return get_fixture_segments()


def get_model_performance(
    engine: Engine | None, demo_mode: bool = False
) -> dict:
    """Get model performance metrics."""
    if demo_mode or engine is None:
        return get_fixture_model_performance()

    try:
        from src.models.registry import ModelRegistry

        registry = ModelRegistry()
        metadata = registry.get_model_metadata()
        if metadata and "metrics" in metadata:
            return metadata["metrics"]
    except Exception as e:
        logger.warning(f"Could not load model metrics from registry: {e}")

    return get_fixture_model_performance()


def get_drift_status(
    engine: Engine | None, demo_mode: bool = False
) -> dict:
    """Get drift monitoring status."""
    if demo_mode or engine is None:
        return get_fixture_drift_status()
    return get_fixture_drift_status()


def get_shap_global(
    engine: Engine | None, demo_mode: bool = False
) -> list[dict]:
    """Get global SHAP feature importance."""
    if demo_mode or engine is None:
        return get_fixture_shap_global()
    return get_fixture_shap_global()
