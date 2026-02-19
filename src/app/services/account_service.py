"""
Account service — at-risk accounts, details, SHAP explanations.
"""

import logging

import pandas as pd
from sqlalchemy import Engine, text

from ..fixtures.demo_data import (
    get_fixture_account_detail,
    get_fixture_account_shap,
    get_fixture_at_risk_accounts,
)

logger = logging.getLogger(__name__)


def get_at_risk_accounts(
    engine: Engine | None,
    demo_mode: bool = False,
    risk_tier: str | None = None,
    plan_type: str | None = None,
    sort_by: str = "churn_probability",
    page: int = 1,
    per_page: int = 20,
) -> dict:
    """Get paginated at-risk accounts."""
    if demo_mode or engine is None:
        return get_fixture_at_risk_accounts(
            risk_tier=risk_tier,
            plan_type=plan_type,
            sort_by=sort_by,
            page=page,
            per_page=per_page,
        )

    try:
        with engine.connect() as conn:
            # Build query with filters
            conditions = ["p.risk_tier IN ('high', 'medium')"]
            params: dict = {}

            if risk_tier and risk_tier != "all":
                conditions.append("p.risk_tier = :risk_tier")
                params["risk_tier"] = risk_tier

            if plan_type and plan_type != "all":
                conditions.append("s.plan_type = :plan_type")
                params["plan_type"] = plan_type

            where_clause = " AND ".join(conditions)

            sort_col = {
                "churn_probability": "p.churn_probability DESC",
                "tenure_days": "tenure_days DESC",
                "last_payment": "last_payment_days ASC",
            }.get(sort_by, "p.churn_probability DESC")

            count_query = text(f"""
                SELECT COUNT(*) as total
                FROM predictions p
                JOIN accounts a ON p.account_id = a.account_id
                JOIN subscriptions s ON a.account_id = s.account_id
                WHERE {where_clause}
            """)
            total = pd.read_sql(count_query, conn, params=params).iloc[0]["total"]

            offset = (page - 1) * per_page
            params["limit"] = per_page
            params["offset"] = offset

            query = text(f"""
                SELECT
                    a.account_id, a.email,
                    p.churn_probability, p.risk_tier,
                    s.plan_type,
                    EXTRACT(DAY FROM NOW() - a.signup_date)::int as tenure_days,
                    0 as last_payment_days,
                    0 as last_stream_days,
                    0 as open_tickets
                FROM predictions p
                JOIN accounts a ON p.account_id = a.account_id
                JOIN subscriptions s ON a.account_id = s.account_id
                WHERE {where_clause}
                ORDER BY {sort_col}
                LIMIT :limit OFFSET :offset
            """)
            df = pd.read_sql(query, conn, params=params)

        items = []
        for _, row in df.iterrows():
            items.append({
                "account_id": row["account_id"],
                "email": row["email"],
                "churn_probability": round(float(row["churn_probability"]), 4),
                "risk_tier": row["risk_tier"],
                "plan_type": row["plan_type"],
                "tenure_days": int(row["tenure_days"]),
                "last_payment_days": int(row["last_payment_days"]),
                "last_stream_days": int(row["last_stream_days"]),
                "open_tickets": int(row["open_tickets"]),
                "top_drivers": [],
            })

        return {
            "items": items,
            "total": int(total),
            "page": page,
            "per_page": per_page,
        }
    except Exception as e:
        logger.error(f"get_at_risk_accounts failed: {e}")
        return get_fixture_at_risk_accounts(
            risk_tier=risk_tier, plan_type=plan_type,
            sort_by=sort_by, page=page, per_page=per_page,
        )


def get_account_detail(
    engine: Engine | None,
    account_id: str,
    demo_mode: bool = False,
) -> dict | None:
    """Get detailed account information."""
    if demo_mode or engine is None:
        return get_fixture_account_detail(account_id)

    try:
        detail = get_fixture_account_detail(account_id)
        if detail:
            return detail

        # Fallback: query database
        with engine.connect() as conn:
            query = text("""
                SELECT a.account_id, a.email, a.signup_date, a.country, a.age, a.gender,
                       s.plan_type, s.status as subscription_status
                FROM accounts a
                JOIN subscriptions s ON a.account_id = s.account_id
                WHERE a.account_id = :account_id
                LIMIT 1
            """)
            df = pd.read_sql(query, conn, params={"account_id": account_id})

        if df.empty:
            return None

        row = df.iloc[0]
        return {
            "account_id": row["account_id"],
            "email": row["email"],
            "signup_date": str(row["signup_date"]),
            "country": row["country"],
            "age": int(row["age"]),
            "gender": row["gender"],
            "plan_type": row["plan_type"],
            "subscription_status": row["subscription_status"],
            "tenure_days": 0,
            "churn_probability": 0.0,
            "risk_tier": "low",
            "top_drivers": [],
            "shap_values": None,
            "payment_history": [],
            "ticket_history": [],
            "last_payment_days": 0,
            "last_stream_days": 0,
            "open_tickets": 0,
            "agent_narrative": "",
        }
    except Exception as e:
        logger.error(f"get_account_detail failed: {e}")
        return get_fixture_account_detail(account_id)


def get_shap_explanation(
    engine: Engine | None,
    account_id: str,
    demo_mode: bool = False,
) -> dict[str, float] | None:
    """Get SHAP explanation for an account."""
    if demo_mode or engine is None:
        return get_fixture_account_shap(account_id)

    # In live mode, could call explain_account_prediction tool
    return get_fixture_account_shap(account_id)
