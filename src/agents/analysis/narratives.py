"""
Narrative result types, KPI definitions, and page-context configs.

All functions in this module are deterministic — no LLM calls.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..utils import safe_json_serialize

logger = logging.getLogger("retain.agents.analysis")


class PageContext(str, Enum):
    """Which front-end page is requesting the narrative."""

    EXECUTIVE_SUMMARY = "executive_summary"
    ANALYTICS_DEEP_DIVE = "analytics_deep_dive"
    AT_RISK_DETAIL = "at_risk_detail"
    PRESCRIPTION_SUMMARY = "prescription_summary"


@dataclass(frozen=True)
class KPIDefinition:
    """Metadata for a single KPI so the agent knows how to contextualize it."""

    name: str
    description: str
    healthy_range: str              # e.g., "< 5%", "> 90%", "$50-80"
    warning_threshold: str          # e.g., "> 8%"
    unit: str                       # "percent", "count", "currency", "days", "hours"


# Registry of all KPIs the analysis agent should understand
KPI_REGISTRY: dict[str, KPIDefinition] = {
    "churn_rate_30d": KPIDefinition(
        name="churn_rate_30d",
        description="30-day churn rate as percentage of subscriber base",
        healthy_range="< 5%",
        warning_threshold="> 8%",
        unit="percent",
    ),
    "retention_rate_30d": KPIDefinition(
        name="retention_rate_30d",
        description="30-day retention rate as percentage",
        healthy_range="> 92%",
        warning_threshold="< 90%",
        unit="percent",
    ),
    "high_risk_count": KPIDefinition(
        name="high_risk_count",
        description="Number of accounts in the high-risk tier",
        healthy_range="< 5% of active base",
        warning_threshold="> 10%",
        unit="count",
    ),
    "monthly_recurring_revenue": KPIDefinition(
        name="monthly_recurring_revenue",
        description="Total monthly recurring revenue from active subscriptions",
        healthy_range="context-dependent",
        warning_threshold="declining trend",
        unit="currency",
    ),
    "avg_customer_lifetime_days": KPIDefinition(
        name="avg_customer_lifetime_days",
        description="Average tenure of active customers in days",
        healthy_range="> 365",
        warning_threshold="< 180",
        unit="days",
    ),
    "payment_failure_rate": KPIDefinition(
        name="payment_failure_rate",
        description="Percentage of payment transactions that failed",
        healthy_range="< 3%",
        warning_threshold="> 5%",
        unit="percent",
    ),
    "avg_resolution_hours": KPIDefinition(
        name="avg_resolution_hours",
        description="Average support ticket resolution time in hours",
        healthy_range="< 24",
        warning_threshold="> 48",
        unit="hours",
    ),
    "engagement_watch_hours_weekly": KPIDefinition(
        name="engagement_watch_hours_weekly",
        description="Average weekly watch hours across the platform",
        healthy_range="no fixed range",
        warning_threshold="declining trend",
        unit="hours",
    ),
}


@dataclass
class NarrativeSection:
    """One section of the analysis narrative."""

    heading: str
    body: str                       # plain English paragraph(s)
    kpis_referenced: list[str]      # which KPIs this section discusses
    sentiment: str                  # "positive", "neutral", "concerning", "critical"
    data_points: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisNarrative:
    """Complete output of the Analysis Agent for a given page context."""

    page_context: str               # which page requested this
    generated_at: str               # ISO timestamp
    reference_date: str
    headline: str                   # one-sentence summary
    sections: list[NarrativeSection]
    overall_sentiment: str          # "healthy", "watch_closely", "action_needed"
    key_callouts: list[str]         # 2-4 bullet-point highlights
    raw_kpis: dict[str, Any]        # all computed KPI values for the front-end
    metadata: dict = field(default_factory=dict)
    success: bool = True
    errors: list[str] = field(default_factory=list)


def assess_kpi_health(name: str, value: float) -> str:
    """Assess health of a KPI by comparing to thresholds.

    Args:
        name: KPI name matching KPI_REGISTRY keys.
        value: Current KPI value.

    Returns:
        "healthy", "warning", or "critical".
    """
    definition = KPI_REGISTRY.get(name)
    if not definition:
        return "healthy"

    # Parse threshold rules based on KPI
    if name == "churn_rate_30d":
        if value > 8:
            return "critical"
        elif value > 5:
            return "warning"
        return "healthy"

    elif name == "retention_rate_30d":
        if value < 88:
            return "critical"
        elif value < 90:
            return "warning"
        return "healthy"

    elif name == "high_risk_count":
        # This is a count, not a percentage — threshold depends on context
        # Treat > 5000 as critical, > 2000 as warning for a 60K base
        if value > 5000:
            return "critical"
        elif value > 2000:
            return "warning"
        return "healthy"

    elif name == "avg_customer_lifetime_days":
        if value < 120:
            return "critical"
        elif value < 180:
            return "warning"
        return "healthy"

    elif name == "payment_failure_rate":
        if value > 8:
            return "critical"
        elif value > 5:
            return "warning"
        return "healthy"

    elif name == "avg_resolution_hours":
        if value > 72:
            return "critical"
        elif value > 48:
            return "warning"
        return "healthy"

    # Default for KPIs without specific thresholds
    return "healthy"


def compute_overall_sentiment(section_sentiments: list[str]) -> str:
    """Compute overall sentiment from section sentiments.

    Args:
        section_sentiments: List of sentiment strings from sections.

    Returns:
        "healthy", "watch_closely", or "action_needed".
    """
    if "critical" in section_sentiments:
        return "action_needed"
    if "concerning" in section_sentiments:
        return "watch_closely"
    return "healthy"


def format_narrative_markdown(narrative: AnalysisNarrative) -> str:
    """Render an AnalysisNarrative as markdown.

    Args:
        narrative: AnalysisNarrative to format.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        f"# Analysis: {narrative.page_context.replace('_', ' ').title()}",
        f"**Reference Date**: {narrative.reference_date}",
        f"**Overall Sentiment**: {narrative.overall_sentiment}",
        "",
        f"## {narrative.headline}",
        "",
    ]

    if narrative.key_callouts:
        lines.append("### Key Highlights")
        for callout in narrative.key_callouts:
            lines.append(f"- {callout}")
        lines.append("")

    for section in narrative.sections:
        lines.append(f"### {section.heading}")
        lines.append(f"*Sentiment: {section.sentiment}*")
        lines.append("")
        lines.append(section.body)
        lines.append("")

    if narrative.errors:
        lines.append("### Errors")
        for err in narrative.errors:
            lines.append(f"- {err}")

    return "\n".join(lines)
