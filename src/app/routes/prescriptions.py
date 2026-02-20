"""
Prescriptions route — strategy-matched recommendations for at-risk accounts.

GET /api/prescriptions — deterministic strategy selection + Analysis Agent narrative.
"""

import json
import logging

from fastapi import APIRouter, Depends, Query

from src.agents.config import AgentConfig

from ..agent_schemas import PrescriptionRecommendation, PrescriptionResponse
from ..dependencies import get_demo_mode

logger = logging.getLogger("retain.app.routes.prescriptions")

router = APIRouter(tags=["Prescriptions"])


@router.get("/prescriptions", response_model=PrescriptionResponse)
async def get_prescriptions(
    reference_date: str = Query("2025-12-01", description="Reference date"),
    demo_mode: bool = Depends(get_demo_mode),
) -> PrescriptionResponse:
    """Prescription recommendations for at-risk accounts.

    Uses deterministic strategy selection from the intervention module,
    plus the Analysis Agent for a summary narrative.
    """
    recommendations: list[PrescriptionRecommendation] = []
    strategy_dist: dict[str, int] = {}
    narrative_text = ""

    try:
        from src.agents.tools import get_account_risk_scores

        # Get high-risk accounts
        scores_raw = get_account_risk_scores.invoke({"risk_tier": "high"})

        if "No risk scores" not in scores_raw and "Error" not in scores_raw:
            # Parse account IDs from markdown table
            accounts = _parse_accounts_from_table(scores_raw)

            if accounts:
                from src.agents.intervention.strategies import (
                    STRATEGY_REGISTRY,
                    select_strategy,
                )

                # For each account (cap at 50), select strategy
                for acct in accounts[:50]:
                    try:
                        account_context = {
                            "tenure_days": acct.get("tenure_days", 365),
                            "subscription_status": "active",
                            "plan_type": acct.get("plan_type", "Regular"),
                        }

                        # Default root cause mapping based on available data
                        root_cause = acct.get("root_cause", "disengagement")
                        strategy = select_strategy(
                            root_cause=root_cause,
                            account_context=account_context,
                        )

                        prob = float(acct.get("churn_probability", 0.7))

                        # Estimate save probability based on strategy priority
                        save_prob = 0.10 + (strategy.priority * 0.03)

                        recommendations.append(PrescriptionRecommendation(
                            account_id=acct["account_id"],
                            churn_probability=prob,
                            risk_tier=acct.get("risk_tier", "high"),
                            recommended_strategy=strategy.name,
                            strategy_description=strategy.description,
                            typical_offer=strategy.typical_offer,
                            priority=strategy.priority,
                            estimated_save_probability=round(save_prob, 2),
                        ))

                        # Count by strategy
                        strategy_dist[strategy.name] = (
                            strategy_dist.get(strategy.name, 0) + 1
                        )
                    except Exception as e:
                        logger.warning(
                            f"Strategy selection failed for "
                            f"{acct.get('account_id')}: {e}"
                        )

    except Exception as e:
        logger.warning(f"Failed to get prescriptions: {e}")

    # Analysis narrative
    try:
        from src.agents.analysis.analyzer import run_analysis

        analysis = run_analysis(
            page_context="prescription_summary",
            reference_date=reference_date,
        )
        narrative_text = "\n\n".join(
            s.body for s in analysis.sections
        ) if analysis.sections else ""
    except Exception as e:
        logger.warning(f"Prescription narrative failed: {e}")
        narrative_text = "Narrative generation temporarily unavailable."

    return PrescriptionResponse(
        recommendations=recommendations,
        total_actionable=len(recommendations),
        narrative=narrative_text,
        strategy_distribution=strategy_dist,
    )


def _parse_accounts_from_table(markdown_table: str) -> list[dict]:
    """Parse account data from markdown table."""
    accounts: list[dict] = []
    lines = markdown_table.strip().split("\n")

    data_lines = [
        line for line in lines
        if line.strip() and not line.startswith("| ---") and "|" in line
    ]

    if len(data_lines) < 2:
        return accounts

    headers = [h.strip() for h in data_lines[0].split("|") if h.strip()]
    for line in data_lines[1:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) >= len(headers):
            row = dict(zip(headers, cells))
            accounts.append(row)

    return accounts
