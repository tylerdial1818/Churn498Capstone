"""
Dashboard route — executive summary with AI-generated narrative.

GET /api/dashboard — Analysis Agent in executive_summary mode.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query

from src.agents.config import AgentConfig

from ..agent_schemas import DashboardResponse, KPIValue
from ..dependencies import get_demo_mode

logger = logging.getLogger("retain.app.routes.dashboard")

router = APIRouter(tags=["Dashboard"])


def _get_agent_config() -> AgentConfig:
    return AgentConfig()


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    reference_date: str = Query("2025-12-01", description="Reference date"),
    demo_mode: bool = Depends(get_demo_mode),
) -> DashboardResponse:
    """Executive dashboard — top-line KPIs with AI-generated narrative.

    The Analysis Agent runs in 'executive_summary' mode to produce a
    natural language headline, narrative, and key callouts. Raw KPI
    values are included for chart rendering.
    """
    try:
        from src.agents.analysis.analyzer import run_analysis
        from src.agents.analysis.narratives import KPI_REGISTRY, assess_kpi_health

        narrative = run_analysis(
            page_context="executive_summary",
            reference_date=reference_date,
        )

        # Build KPIValue list from raw_kpis
        kpis: list[KPIValue] = []
        for name, value in narrative.raw_kpis.items():
            if name in KPI_REGISTRY:
                defn = KPI_REGISTRY[name]
                kpis.append(KPIValue(
                    name=name,
                    value=float(value),
                    unit=defn.unit,
                    health=assess_kpi_health(name, float(value)),
                    description=defn.description,
                ))

        # Extract key values
        raw = narrative.raw_kpis
        high_risk_count = int(raw.get("high_risk_count", 0))
        churn_rate = float(raw.get("churn_rate_30d", 0.0))
        monthly_revenue = float(raw.get("monthly_recurring_revenue", 0.0))
        retention_rate = float(raw.get("retention_rate_30d", 0.0))

        # Combine section bodies for the narrative
        narrative_text = "\n\n".join(
            s.body for s in narrative.sections
        ) if narrative.sections else ""

        return DashboardResponse(
            kpis=kpis,
            high_risk_count=high_risk_count,
            churn_rate=churn_rate,
            monthly_revenue=monthly_revenue,
            retention_rate=retention_rate,
            headline=narrative.headline,
            narrative=narrative_text,
            key_callouts=narrative.key_callouts[:4],
            overall_sentiment=narrative.overall_sentiment,
            generated_at=narrative.generated_at,
        )

    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        # Return partial result with error info
        return DashboardResponse(
            kpis=[],
            high_risk_count=0,
            churn_rate=0.0,
            monthly_revenue=0.0,
            retention_rate=0.0,
            headline="Dashboard temporarily unavailable",
            narrative=f"Analysis agent encountered an error: {e}",
            key_callouts=[],
            overall_sentiment="healthy",
            generated_at=datetime.now().isoformat(),
        )
