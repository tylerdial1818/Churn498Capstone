"""
At-risk accounts route — high-risk list with Early Warning alerts.

GET /api/at-risk — accounts + optional early warning report.
GET /api/at-risk/{account_id} — single account detail with SHAP.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from src.agents.config import AgentConfig

from ..agent_schemas import (
    AlertGroupResponse,
    AtRiskAccount,
    AtRiskAccountDetail,
    AtRiskResponse,
    EarlyWarningAlertResponse,
)
from ..dependencies import get_demo_mode

logger = logging.getLogger("retain.app.routes.at_risk")

router = APIRouter(tags=["At Risk"])


@router.get("/at-risk", response_model=AtRiskResponse)
async def get_at_risk_accounts(
    reference_date: str = Query("2025-12-01", description="Reference date"),
    include_early_warning: bool = Query(
        True, description="Include Early Warning Agent report"
    ),
    demo_mode: bool = Depends(get_demo_mode),
) -> AtRiskResponse:
    """At-risk accounts list with optional Early Warning Agent report.

    Combines scored high-risk accounts with risk transition alerts.
    The Analysis Agent runs in 'at_risk_detail' mode for the narrative.
    """
    accounts: list[AtRiskAccount] = []
    early_warning_response: EarlyWarningAlertResponse | None = None
    narrative_text = ""

    try:
        # Get high-risk scores
        from src.agents.tools import get_account_risk_scores

        scores_raw = get_account_risk_scores.invoke({"risk_tier": "high"})

        # Parse markdown table into account objects
        if "No risk scores" not in scores_raw and "Error" not in scores_raw:
            accounts = _parse_risk_scores_to_accounts(scores_raw)

    except Exception as e:
        logger.warning(f"Failed to get risk scores: {e}")

    # Early Warning
    if include_early_warning:
        try:
            from src.agents.early_warning.detector import run_early_warning

            ew_report = run_early_warning(
                reference_date=reference_date,
            )

            alert_groups = [
                AlertGroupResponse(
                    root_cause=g.root_cause,
                    count=len(g.accounts),
                    priority=g.priority,
                    representative_accounts=g.representative_account_ids[:5],
                    evidence_summary=g.evidence_summary,
                    recommended_action=g.recommended_action,
                )
                for g in ew_report.alert_groups
            ]

            early_warning_response = EarlyWarningAlertResponse(
                headline=ew_report.headline,
                total_escalated=ew_report.total_escalated,
                total_new_high_risk=ew_report.total_new_high_risk,
                total_improved=ew_report.total_improved,
                alert_groups=alert_groups,
                narrative=ew_report.narrative,
            )
        except Exception as e:
            logger.warning(f"Early warning failed: {e}")

    # Analysis narrative
    try:
        from src.agents.analysis.analyzer import run_analysis

        analysis = run_analysis(
            page_context="at_risk_detail",
            reference_date=reference_date,
        )
        narrative_text = "\n\n".join(
            s.body for s in analysis.sections
        ) if analysis.sections else ""
    except Exception as e:
        logger.warning(f"Analysis narrative failed: {e}")
        narrative_text = "Narrative generation temporarily unavailable."

    return AtRiskResponse(
        accounts=accounts,
        total_high_risk=len(accounts),
        early_warning=early_warning_response,
        narrative=narrative_text,
    )


@router.get("/at-risk/{account_id}", response_model=AtRiskAccountDetail)
async def get_at_risk_account_detail(
    account_id: str,
    demo_mode: bool = Depends(get_demo_mode),
) -> AtRiskAccountDetail:
    """Detailed view of a single at-risk account with SHAP explanation."""
    from src.agents.utils import validate_account_id

    if not validate_account_id(account_id):
        raise HTTPException(status_code=422, detail="Invalid account ID format")

    try:
        from src.agents.tools import (
            explain_account_prediction,
            get_account_payment_history,
            get_account_profile,
            get_account_risk_scores,
            get_account_support_history,
            get_account_viewing_summary,
        )

        # Get score
        score_raw = get_account_risk_scores.invoke(
            {"account_ids": [account_id]}
        )
        if "No risk scores" in score_raw:
            raise HTTPException(status_code=404, detail="Account not found")

        # Get SHAP
        shap_raw = explain_account_prediction.invoke(
            {"account_id": account_id}
        )
        shap_features = _parse_shap_explanation(shap_raw)

        # Get profile
        profile_raw = get_account_profile.invoke(
            {"account_id": account_id}
        )
        profile = _parse_profile(profile_raw)

        return AtRiskAccountDetail(
            account_id=account_id,
            email=profile.get("email", ""),
            churn_probability=profile.get("churn_probability", 0.0),
            risk_tier=profile.get("risk_tier", "unknown"),
            plan_type=profile.get("plan_type", "unknown"),
            tenure_days=profile.get("tenure_days", 0),
            shap_explanation=shap_features,
            profile=profile,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account detail failed for {account_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _parse_risk_scores_to_accounts(
    markdown_table: str,
) -> list[AtRiskAccount]:
    """Parse a markdown table of risk scores into AtRiskAccount objects."""
    accounts: list[AtRiskAccount] = []
    lines = markdown_table.strip().split("\n")

    # Skip header and separator
    data_lines = [
        line for line in lines
        if line.strip() and not line.startswith("| ---") and "|" in line
    ]

    if len(data_lines) < 2:
        return accounts

    headers = [
        h.strip() for h in data_lines[0].split("|") if h.strip()
    ]
    for line in data_lines[1:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) >= len(headers):
            row = dict(zip(headers, cells))
            try:
                accounts.append(AtRiskAccount(
                    account_id=row.get("account_id", ""),
                    email=row.get("email", ""),
                    churn_probability=float(
                        row.get("churn_probability", "0")
                    ),
                    risk_tier=row.get("risk_tier", "high"),
                    plan_type=row.get("plan_type", "unknown"),
                    tenure_days=int(float(
                        row.get("tenure_days", "0")
                    )),
                    primary_risk_driver=row.get(
                        "primary_risk_driver", None
                    ),
                    last_scored_at=row.get("scored_at", ""),
                ))
            except (ValueError, KeyError):
                continue

    return accounts


def _parse_shap_explanation(raw: str) -> list[dict]:
    """Parse SHAP explanation text into structured features."""
    features = []
    for line in raw.split("\n"):
        if line.strip().startswith("- **"):
            try:
                parts = line.split("**")
                if len(parts) >= 3:
                    feature = parts[1].strip()
                    rest = parts[2].strip()
                    # Extract SHAP value
                    if "SHAP:" in rest:
                        shap_str = rest.split("SHAP:")[1].strip().rstrip(")")
                        shap_val = float(shap_str)
                        features.append({
                            "feature": feature,
                            "shap_value": shap_val,
                        })
            except (ValueError, IndexError):
                continue
    return features


def _parse_profile(raw: str) -> dict:
    """Parse profile text into a dictionary."""
    profile: dict = {}
    for line in raw.split("\n"):
        if line.strip().startswith("- **"):
            try:
                parts = line.split("**")
                if len(parts) >= 3:
                    key = parts[1].strip().rstrip(":")
                    value = parts[2].strip().lstrip(": ")
                    profile[key] = value
            except (ValueError, IndexError):
                continue
    return profile
