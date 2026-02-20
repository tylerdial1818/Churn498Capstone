"""
Analysis Agent for dashboard narration.

Produces natural language explanations of KPIs and charts
for different front-end page contexts.
"""

from .analyzer import create_analysis_agent, run_analysis
from .narratives import (
    AnalysisNarrative,
    KPIDefinition,
    KPI_REGISTRY,
    NarrativeSection,
    PageContext,
    assess_kpi_health,
    compute_overall_sentiment,
    format_narrative_markdown,
)

__all__ = [
    "PageContext",
    "KPIDefinition",
    "KPI_REGISTRY",
    "NarrativeSection",
    "AnalysisNarrative",
    "assess_kpi_health",
    "compute_overall_sentiment",
    "format_narrative_markdown",
    "create_analysis_agent",
    "run_analysis",
]
