"""
Early Warning Agent for proactive churn risk detection.

Compares scoring runs, identifies risk escalations, groups alerts
by root cause, and produces structured alert reports.
"""

from .alerts import (
    AlertGroup,
    EarlyWarningReport,
    RiskTransition,
    classify_transitions,
    compute_alert_priority,
    format_alert_report_markdown,
)
from .detector import (
    create_early_warning_agent,
    run_early_warning,
)

__all__ = [
    "RiskTransition",
    "AlertGroup",
    "EarlyWarningReport",
    "classify_transitions",
    "compute_alert_priority",
    "format_alert_report_markdown",
    "create_early_warning_agent",
    "run_early_warning",
]
