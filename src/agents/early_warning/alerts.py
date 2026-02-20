"""
Alert dataclasses, grouping/ranking logic, and formatting.

All functions in this module are deterministic — no LLM calls.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..utils import safe_json_serialize

logger = logging.getLogger("retain.agents.early_warning")

# Valid root cause categories (matches DDP pipeline)
VALID_ROOT_CAUSES = frozenset({
    "payment_issues",
    "disengagement",
    "support_frustration",
    "price_sensitivity",
    "content_gap",
    "technical_issues",
})


@dataclass
class RiskTransition:
    """A single account's risk tier change between scoring runs."""

    account_id: str
    previous_tier: str              # "low", "medium", "high"
    current_tier: str
    previous_probability: float
    current_probability: float
    probability_delta: float        # current - previous
    direction: str                  # "escalated", "improved", "stable"

    def __post_init__(self) -> None:
        tier_order = {"low": 0, "medium": 1, "high": 2}
        prev_rank = tier_order.get(self.previous_tier, 0)
        curr_rank = tier_order.get(self.current_tier, 0)

        if not self.direction:
            if curr_rank > prev_rank:
                self.direction = "escalated"
            elif curr_rank < prev_rank:
                self.direction = "improved"
            else:
                self.direction = "stable"

        if self.probability_delta == 0.0:
            self.probability_delta = round(
                self.current_probability - self.previous_probability, 4
            )


@dataclass
class AlertGroup:
    """A cluster of escalated accounts sharing a common root cause."""

    root_cause: str                 # matches DDP categories
    accounts: list[RiskTransition]
    representative_account_ids: list[str]  # 3-5 accounts to highlight
    evidence_summary: str           # LLM-generated explanation
    priority: int                   # 1-5
    recommended_action: str         # brief suggested next step

    def __post_init__(self) -> None:
        if self.root_cause not in VALID_ROOT_CAUSES:
            logger.warning(
                f"Unknown root cause '{self.root_cause}', "
                f"valid: {VALID_ROOT_CAUSES}"
            )


@dataclass
class EarlyWarningReport:
    """Complete output of the Early Warning Agent."""

    reference_date: str
    previous_date: str
    total_accounts_scored: int
    total_escalated: int            # accounts that moved to higher tier
    total_new_high_risk: int        # accounts that entered high-risk tier
    total_improved: int             # accounts that moved to lower tier
    alert_groups: list[AlertGroup]  # escalated accounts by root cause
    headline: str                   # summary headline
    narrative: str                  # 2-3 paragraph summary
    model_health: str               # brief note on model drift status
    metadata: dict = field(default_factory=dict)
    success: bool = True
    errors: list[str] = field(default_factory=list)


def classify_transitions(
    transitions: list[RiskTransition],
) -> dict[str, list[RiskTransition]]:
    """Group transitions by direction (escalated, improved, stable).

    Args:
        transitions: List of RiskTransition objects.

    Returns:
        Dict mapping direction to list of transitions.
    """
    groups: dict[str, list[RiskTransition]] = {
        "escalated": [],
        "improved": [],
        "stable": [],
    }
    for t in transitions:
        groups.setdefault(t.direction, []).append(t)
    return groups


def compute_alert_priority(group_size: int, avg_delta: float) -> int:
    """Compute priority score for an alert group.

    Larger groups and bigger probability deltas get higher priority.

    Args:
        group_size: Number of accounts in the group.
        avg_delta: Average probability delta for the group.

    Returns:
        Priority score 1-5 (5 = highest urgency).
    """
    # Score based on size (0-2.5 points)
    if group_size >= 50:
        size_score = 2.5
    elif group_size >= 20:
        size_score = 2.0
    elif group_size >= 10:
        size_score = 1.5
    elif group_size >= 5:
        size_score = 1.0
    else:
        size_score = 0.5

    # Score based on delta magnitude (0-2.5 points)
    abs_delta = abs(avg_delta)
    if abs_delta >= 0.30:
        delta_score = 2.5
    elif abs_delta >= 0.20:
        delta_score = 2.0
    elif abs_delta >= 0.10:
        delta_score = 1.5
    elif abs_delta >= 0.05:
        delta_score = 1.0
    else:
        delta_score = 0.5

    total = size_score + delta_score
    # Map to 1-5
    if total >= 4.5:
        return 5
    elif total >= 3.5:
        return 4
    elif total >= 2.5:
        return 3
    elif total >= 1.5:
        return 2
    else:
        return 1


def _recommended_action(root_cause: str) -> str:
    """Get recommended action for a root cause category."""
    actions = {
        "payment_issues": (
            "Trigger payment recovery workflow: retry reminders, "
            "alternative payment method prompts, grace period extension."
        ),
        "disengagement": (
            "Launch re-engagement campaign: personalized content "
            "recommendations, 'we miss you' outreach, new release alerts."
        ),
        "support_frustration": (
            "Escalate to VIP support queue: proactive resolution, "
            "service credit, dedicated account manager assignment."
        ),
        "price_sensitivity": (
            "Offer targeted discount: limited-time price lock, "
            "feature comparison showing value, loyalty reward."
        ),
        "content_gap": (
            "Send curated content digest: genre expansion "
            "suggestions, upcoming releases matching preferences."
        ),
        "technical_issues": (
            "Initiate proactive tech support: device-specific "
            "troubleshooting, app update notification, known issue fix."
        ),
    }
    return actions.get(root_cause, "Review account details and determine intervention.")


def compose_headline(
    total_escalated: int,
    total_new_high_risk: int,
    alert_groups: list[AlertGroup],
) -> str:
    """Compose a headline string for the report.

    Args:
        total_escalated: Total escalated accounts.
        total_new_high_risk: Accounts entering high-risk tier.
        alert_groups: Grouped alerts.

    Returns:
        Headline string with specific numbers.
    """
    if total_escalated == 0:
        return "All clear — no accounts escalated to higher risk tiers this period."

    # Find top cause
    if alert_groups:
        top_group = max(alert_groups, key=lambda g: len(g.accounts))
        top_pct = round(len(top_group.accounts) / max(total_escalated, 1) * 100)
        cause_label = top_group.root_cause.replace("_", " ")
        return (
            f"{total_escalated} accounts escalated to higher risk — "
            f"{cause_label} drives {top_pct}%"
        )

    return f"{total_escalated} accounts escalated to higher risk this period."


def compose_narrative(
    report: "EarlyWarningReport",
) -> str:
    """Compose a 2-3 paragraph narrative from structured data.

    Template-based, not LLM-generated.

    Args:
        report: The EarlyWarningReport to narrate.

    Returns:
        Narrative string.
    """
    if report.total_escalated == 0:
        return (
            f"Between {report.previous_date} and {report.reference_date}, "
            f"no accounts moved to a higher risk tier. "
            f"{report.total_improved} accounts improved to lower risk tiers. "
            f"The subscriber base appears stable."
        )

    paragraphs = []

    # Paragraph 1: Overview
    paragraphs.append(
        f"Between {report.previous_date} and {report.reference_date}, "
        f"{report.total_escalated} accounts escalated to higher risk tiers, "
        f"including {report.total_new_high_risk} that newly entered the "
        f"high-risk category. Meanwhile, {report.total_improved} accounts "
        f"showed improvement."
    )

    # Paragraph 2: Group breakdown
    if report.alert_groups:
        sorted_groups = sorted(
            report.alert_groups,
            key=lambda g: g.priority,
            reverse=True,
        )
        group_lines = []
        for g in sorted_groups:
            cause = g.root_cause.replace("_", " ")
            count = len(g.accounts)
            group_lines.append(
                f"{cause} ({count} accounts, priority {g.priority})"
            )
        paragraphs.append(
            "Root cause breakdown: " + "; ".join(group_lines) + "."
        )

    # Paragraph 3: Action needed
    if report.alert_groups:
        top = max(report.alert_groups, key=lambda g: g.priority)
        cause = top.root_cause.replace("_", " ")
        paragraphs.append(
            f"Immediate attention recommended for the {cause} group "
            f"(priority {top.priority}/5). "
            f"{top.recommended_action}"
        )

    return "\n\n".join(paragraphs)


def format_alert_report_markdown(report: EarlyWarningReport) -> str:
    """Render the full report as markdown.

    Args:
        report: EarlyWarningReport to format.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        f"# Early Warning Report",
        f"**Period**: {report.previous_date} → {report.reference_date}",
        f"**Total Scored**: {report.total_accounts_scored:,}",
        "",
        f"## {report.headline}",
        "",
        report.narrative,
        "",
    ]

    if report.alert_groups:
        lines.append("## Alert Groups")
        lines.append("")
        for i, group in enumerate(
            sorted(report.alert_groups, key=lambda g: g.priority, reverse=True),
            1,
        ):
            cause = group.root_cause.replace("_", " ").title()
            lines.append(
                f"### {i}. {cause} (Priority {group.priority}/5)"
            )
            lines.append(f"- **Accounts**: {len(group.accounts)}")
            lines.append(
                f"- **Representative IDs**: "
                f"{', '.join(group.representative_account_ids)}"
            )
            lines.append(f"- **Evidence**: {group.evidence_summary}")
            lines.append(f"- **Action**: {group.recommended_action}")
            lines.append("")

    lines.append("## Summary")
    lines.append(f"- Escalated: {report.total_escalated}")
    lines.append(f"- New High Risk: {report.total_new_high_risk}")
    lines.append(f"- Improved: {report.total_improved}")
    lines.append(f"- Model Health: {report.model_health}")
    lines.append("")

    if report.errors:
        lines.append("## Errors")
        for err in report.errors:
            lines.append(f"- {err}")

    return "\n".join(lines)
