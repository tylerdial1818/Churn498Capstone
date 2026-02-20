"""
AI agent system for Retain churn prediction platform.

Provides multi-agent workflows for detecting churn risk, diagnosing
root causes, prescribing interventions, and drafting retention emails.

For CLI usage:
    python -m src.agents.pipelines.ddp_pipeline       # Run full DDP pipeline
    python -m src.agents.intervention.drafter --help   # Draft retention emails
"""

# Configuration
from .config import AgentConfig

# State
from .state import RetainAgentState, create_initial_state

# Prompts
from .prompts import (
    ANALYSIS_AGENT_PROMPT,
    DETECTION_AGENT_PROMPT,
    DIAGNOSIS_AGENT_PROMPT,
    EARLY_WARNING_AGENT_PROMPT,
    INTERVENTION_DRAFTER_PROMPT,
    PRESCRIPTION_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
)

# Tools
from .tools import (
    ALL_TOOLS,
    ANALYSIS_TOOLS,
    DETECTION_TOOLS,
    DIAGNOSIS_TOOLS,
    EARLY_WARNING_TOOLS,
    PRESCRIPTION_TOOLS,
    explain_account_prediction,
    get_account_features,
    get_account_payment_history,
    get_account_profile,
    get_account_risk_scores,
    get_account_scoring_history,
    get_account_support_history,
    get_account_viewing_summary,
    get_churn_cohort_analysis,
    get_engagement_trends,
    get_executive_kpis,
    get_feature_importance_global,
    get_model_health_status,
    get_new_high_risk_accounts,
    get_payment_health,
    get_risk_tier_transitions,
    get_risk_velocity,
    get_subscription_distribution,
    get_support_health,
    query_database_readonly,
    run_batch_scoring,
    segment_high_risk_accounts,
)

# Utils
from .utils import (
    format_dataframe_as_markdown,
    get_reference_engine,
    safe_json_serialize,
    truncate_for_context,
    validate_account_id,
)

__all__ = [
    # Config
    "AgentConfig",
    # State
    "RetainAgentState",
    "create_initial_state",
    # Prompts
    "SUPERVISOR_PROMPT",
    "DETECTION_AGENT_PROMPT",
    "DIAGNOSIS_AGENT_PROMPT",
    "PRESCRIPTION_AGENT_PROMPT",
    "INTERVENTION_DRAFTER_PROMPT",
    "EARLY_WARNING_AGENT_PROMPT",
    "ANALYSIS_AGENT_PROMPT",
    # Tools
    "ALL_TOOLS",
    "DETECTION_TOOLS",
    "DIAGNOSIS_TOOLS",
    "PRESCRIPTION_TOOLS",
    "EARLY_WARNING_TOOLS",
    "ANALYSIS_TOOLS",
    "get_account_risk_scores",
    "get_account_scoring_history",
    "run_batch_scoring",
    "get_account_features",
    "explain_account_prediction",
    "get_feature_importance_global",
    "get_account_profile",
    "get_account_support_history",
    "get_account_payment_history",
    "get_account_viewing_summary",
    "query_database_readonly",
    "segment_high_risk_accounts",
    "get_model_health_status",
    "get_risk_tier_transitions",
    "get_risk_velocity",
    "get_new_high_risk_accounts",
    "get_executive_kpis",
    "get_subscription_distribution",
    "get_engagement_trends",
    "get_support_health",
    "get_payment_health",
    "get_churn_cohort_analysis",
    # Utils
    "format_dataframe_as_markdown",
    "safe_json_serialize",
    "validate_account_id",
    "get_reference_engine",
    "truncate_for_context",
]
