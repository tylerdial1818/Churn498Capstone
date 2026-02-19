"""
Dashboard service — wraps existing modules for KPIs, trends, risk distribution.
"""

import logging

import pandas as pd
from sqlalchemy import Engine, text

from ..fixtures.demo_data import (
    get_fixture_active_inactive,
    get_fixture_executive_summary,
    get_fixture_kpis,
    get_fixture_risk_distribution,
    get_fixture_trends,
)

logger = logging.getLogger(__name__)


def get_kpis(engine: Engine | None, demo_mode: bool = False) -> dict:
    """Get dashboard KPIs."""
    if demo_mode or engine is None:
        return get_fixture_kpis()

    try:
        with engine.connect() as conn:
            total = pd.read_sql(
                text("SELECT COUNT(*) as cnt FROM accounts"), conn
            ).iloc[0]["cnt"]

            active = pd.read_sql(
                text(
                    "SELECT COUNT(*) as cnt FROM subscriptions "
                    "WHERE status = 'active'"
                ),
                conn,
            ).iloc[0]["cnt"]

            churned = total - active

            # Risk counts from predictions table
            try:
                risk_df = pd.read_sql(
                    text(
                        "SELECT risk_tier, COUNT(*) as cnt "
                        "FROM predictions GROUP BY risk_tier"
                    ),
                    conn,
                )
                risk_map = dict(zip(risk_df["risk_tier"], risk_df["cnt"]))
                high_risk = risk_map.get("high", 0)
            except Exception:
                high_risk = 0

            churn_rate = churned / total if total > 0 else 0
            avg_mrr = 14.0
            at_risk_mrr = high_risk * avg_mrr

            return {
                "total_accounts": int(total),
                "active_subscribers": int(active),
                "churned_accounts": int(churned),
                "churn_rate_30d": round(churn_rate, 4),
                "high_risk_count": int(high_risk),
                "at_risk_mrr": round(at_risk_mrr, 2),
                "cac": 45.00,
                "retention_cost_per_save": 12.50,
            }
    except Exception as e:
        logger.error(f"get_kpis failed, falling back to fixtures: {e}")
        return get_fixture_kpis()


def get_trends(engine: Engine | None, demo_mode: bool = False) -> list[dict]:
    """Get monthly trend data."""
    if demo_mode or engine is None:
        return get_fixture_trends()

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    TO_CHAR(start_date, 'YYYY-MM') as month,
                    COUNT(*) FILTER (WHERE status = 'active') as signups,
                    COUNT(*) FILTER (WHERE status IN ('canceled', 'expired')) as cancellations
                FROM subscriptions
                GROUP BY TO_CHAR(start_date, 'YYYY-MM')
                ORDER BY month
            """)
            df = pd.read_sql(query, conn)

        trends = []
        for _, row in df.iterrows():
            s = int(row["signups"])
            c = int(row["cancellations"])
            total = s + c
            trends.append({
                "month": row["month"],
                "signups": s,
                "cancellations": c,
                "net_growth": s - c,
                "churn_rate": round(c / total, 4) if total > 0 else 0,
            })
        return trends
    except Exception as e:
        logger.error(f"get_trends failed, falling back to fixtures: {e}")
        return get_fixture_trends()


def get_risk_distribution(
    engine: Engine | None, demo_mode: bool = False
) -> dict:
    """Get risk tier distribution from predictions."""
    if demo_mode or engine is None:
        return get_fixture_risk_distribution()

    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT risk_tier, COUNT(*) as cnt "
                    "FROM predictions GROUP BY risk_tier"
                ),
                conn,
            )
        risk_map = dict(zip(df["risk_tier"], df["cnt"]))
        return {
            "low": int(risk_map.get("low", 0)),
            "medium": int(risk_map.get("medium", 0)),
            "high": int(risk_map.get("high", 0)),
        }
    except Exception as e:
        logger.error(f"get_risk_distribution failed: {e}")
        return get_fixture_risk_distribution()


def get_active_inactive(
    engine: Engine | None, demo_mode: bool = False
) -> dict:
    """Get active vs inactive subscriber distribution."""
    if demo_mode or engine is None:
        return get_fixture_active_inactive()

    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT "
                    "  COUNT(*) FILTER (WHERE status = 'active') as active, "
                    "  COUNT(*) FILTER (WHERE status != 'active') as inactive "
                    "FROM subscriptions"
                ),
                conn,
            )
        a = int(df.iloc[0]["active"])
        i = int(df.iloc[0]["inactive"])
        total = a + i
        return {
            "active": a,
            "inactive": i,
            "active_pct": round(a / total * 100, 1) if total > 0 else 0,
            "inactive_pct": round(i / total * 100, 1) if total > 0 else 0,
        }
    except Exception as e:
        logger.error(f"get_active_inactive failed: {e}")
        return get_fixture_active_inactive()


def get_executive_summary(
    engine: Engine | None, demo_mode: bool = False
) -> dict:
    """Get AI executive summary."""
    if demo_mode or engine is None:
        return get_fixture_executive_summary()

    # In live mode, return fixture summary (LLM call would be async)
    return get_fixture_executive_summary()
