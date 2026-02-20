"""
Analytics route — full stats with AI-generated narrative.

GET /api/analytics — Analysis Agent in analytics_deep_dive mode.
Returns both raw data (for charts) and narrative (for explanations).
"""

import json
import logging

from fastapi import APIRouter, Depends, Query

from src.agents.config import AgentConfig

from ..agent_schemas import (
    AnalysisNarrativeResponse,
    AnalyticsResponse,
    NarrativeSectionResponse,
)
from ..dependencies import get_demo_mode

logger = logging.getLogger("retain.app.routes.analytics")

router = APIRouter(tags=["Analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    reference_date: str = Query("2025-12-01", description="Reference date"),
    lookback_days: int = Query(90, description="Days to look back"),
    demo_mode: bool = Depends(get_demo_mode),
) -> AnalyticsResponse:
    """Full analytics dashboard with AI-generated narrative.

    The Analysis Agent runs in 'analytics_deep_dive' mode.
    Returns both raw data (for charts) and narrative (for explanations).
    """
    try:
        from src.agents.analysis.analyzer import run_analysis

        analysis = run_analysis(
            page_context="analytics_deep_dive",
            reference_date=reference_date,
        )

        # Build narrative response
        sections = [
            NarrativeSectionResponse(
                heading=s.heading,
                body=s.body,
                sentiment=s.sentiment,
            )
            for s in analysis.sections
        ]

        narrative_response = AnalysisNarrativeResponse(
            headline=analysis.headline,
            sections=sections,
            overall_sentiment=analysis.overall_sentiment,
            key_callouts=analysis.key_callouts[:4],
        )

        # Extract supplementary data from metadata if available
        raw_kpis = analysis.raw_kpis or {}

        # Get supplementary data by calling tools directly
        engagement_trends: list[dict] = []
        support_metrics: dict = {}
        payment_metrics: dict = {}
        subscription_distribution: dict = {}
        cohort_analysis: list[dict] = []

        try:
            from src.agents.tools import (
                get_churn_cohort_analysis,
                get_engagement_trends,
                get_payment_health,
                get_subscription_distribution,
                get_support_health,
            )

            # Call each tool and parse JSON results
            eng_raw = get_engagement_trends.invoke({
                "reference_date": reference_date,
                "lookback_days": lookback_days,
            })
            try:
                eng_data = json.loads(eng_raw)
                engagement_trends = eng_data.get("weekly_data", [])
            except (json.JSONDecodeError, TypeError):
                pass

            sup_raw = get_support_health.invoke({
                "reference_date": reference_date,
                "lookback_days": lookback_days,
            })
            try:
                support_metrics = json.loads(sup_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            pay_raw = get_payment_health.invoke({
                "reference_date": reference_date,
                "lookback_days": lookback_days,
            })
            try:
                payment_metrics = json.loads(pay_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            sub_raw = get_subscription_distribution.invoke({
                "reference_date": reference_date,
            })
            try:
                subscription_distribution = json.loads(sub_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            coh_raw = get_churn_cohort_analysis.invoke({
                "reference_date": reference_date,
            })
            try:
                coh_data = json.loads(coh_raw)
                cohort_analysis = coh_data.get("cohorts", [])
            except (json.JSONDecodeError, TypeError):
                pass

        except Exception as e:
            logger.warning(f"Supplementary data collection failed: {e}")

        return AnalyticsResponse(
            narrative=narrative_response,
            raw_kpis=raw_kpis,
            engagement_trends=engagement_trends,
            support_metrics=support_metrics,
            payment_metrics=payment_metrics,
            subscription_distribution=subscription_distribution,
            cohort_analysis=cohort_analysis,
        )

    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")
        return AnalyticsResponse(
            narrative=AnalysisNarrativeResponse(
                headline="Analytics temporarily unavailable",
                sections=[],
                overall_sentiment="healthy",
                key_callouts=[],
            ),
            raw_kpis={},
            engagement_trends=[],
            support_metrics={},
            payment_metrics={},
            subscription_distribution={},
            cohort_analysis=[],
        )
