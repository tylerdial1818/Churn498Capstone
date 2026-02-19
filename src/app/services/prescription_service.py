"""
Prescription service — strategy grouping and account prescriptions.
"""

import logging

from sqlalchemy import Engine

from ..fixtures.demo_data import (
    get_fixture_prescription_by_strategy,
    get_fixture_prescriptions,
)

logger = logging.getLogger(__name__)


def get_prescriptions(
    engine: Engine | None, demo_mode: bool = False
) -> list[dict]:
    """Get all prescription groups."""
    if demo_mode or engine is None:
        return get_fixture_prescriptions()

    try:
        from src.agents.intervention.strategies import (
            STRATEGY_REGISTRY,
            select_strategy,
        )

        # In live mode, would query predictions + apply select_strategy
        # For now, return fixtures as the live query requires scoring data
        return get_fixture_prescriptions()
    except Exception as e:
        logger.error(f"get_prescriptions failed: {e}")
        return get_fixture_prescriptions()


def get_prescription_by_strategy(
    engine: Engine | None,
    strategy: str,
    demo_mode: bool = False,
) -> dict | None:
    """Get a single prescription group by strategy."""
    if demo_mode or engine is None:
        return get_fixture_prescription_by_strategy(strategy)

    groups = get_prescriptions(engine, demo_mode)
    return next((g for g in groups if g["strategy"] == strategy), None)
