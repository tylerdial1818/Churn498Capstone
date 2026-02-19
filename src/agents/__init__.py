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
    DETECTION_AGENT_PROMPT,
    DIAGNOSIS_AGENT_PROMPT,
    INTERVENTION_DRAFTER_PROMPT,
    PRESCRIPTION_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
)

# Tools
from .tools import (
    ALL_TOOLS,
    DETECTION_TOOLS,
    DIAGNOSIS_TOOLS,
    PRESCRIPTION_TOOLS,
    explain_account_prediction,
    get_account_features,
    get_account_payment_history,
    get_account_profile,
    get_account_risk_scores,
    get_account_scoring_history,
    get_account_support_history,
    get_account_viewing_summary,
    get_feature_importance_global,
    get_model_health_status,
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
    # Tools
    "ALL_TOOLS",
    "DETECTION_TOOLS",
    "DIAGNOSIS_TOOLS",
    "PRESCRIPTION_TOOLS",
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
    # Utils
    "format_dataframe_as_markdown",
    "safe_json_serialize",
    "validate_account_id",
    "get_reference_engine",
    "truncate_for_context",
]
