"""
Pydantic v2 response models for agent-backed API endpoints.

These schemas define the data contracts between the AI agent layer
and the front-end for the /api/ routes.
"""

from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Dashboard
# =============================================================================


class KPIValue(BaseModel):
    """A single KPI with health assessment."""

    name: str = Field(description="KPI identifier")
    value: float = Field(description="Current KPI value")
    unit: str = Field(description="Unit: percent, count, currency, days, hours")
    health: str = Field(description="Health status: healthy, warning, critical")
    description: str = Field(description="Human-readable description")


class DashboardResponse(BaseModel):
    """Executive dashboard with AI-generated narrative."""

    kpis: list[KPIValue] = Field(description="KPI values with health assessments")
    high_risk_count: int = Field(description="Accounts in high-risk tier")
    churn_rate: float = Field(description="30-day churn rate percentage")
    monthly_revenue: float = Field(description="Monthly recurring revenue")
    retention_rate: float = Field(description="30-day retention rate percentage")
    headline: str = Field(description="AI-generated headline summary")
    narrative: str = Field(description="Executive summary narrative")
    key_callouts: list[str] = Field(description="2-4 bullet highlights")
    overall_sentiment: str = Field(
        description="Overall health: healthy, watch_closely, action_needed"
    )
    generated_at: str = Field(description="ISO timestamp of generation")


# =============================================================================
# At-Risk
# =============================================================================


class AtRiskAccount(BaseModel):
    """A single at-risk account."""

    account_id: str
    email: str
    churn_probability: float
    risk_tier: str
    plan_type: str
    tenure_days: int
    primary_risk_driver: str | None = None
    last_scored_at: str


class AlertGroupResponse(BaseModel):
    """A group of accounts sharing a root cause."""

    root_cause: str = Field(description="One of the 6 risk categories")
    count: int = Field(description="Number of accounts in this group")
    priority: int = Field(description="Priority 1-5 (5=highest)")
    representative_accounts: list[str] = Field(
        description="3-5 representative account IDs"
    )
    evidence_summary: str = Field(description="Why this group matters")
    recommended_action: str = Field(description="Suggested intervention")


class EarlyWarningAlertResponse(BaseModel):
    """Early Warning Agent alert report."""

    headline: str
    total_escalated: int
    total_new_high_risk: int
    total_improved: int
    alert_groups: list[AlertGroupResponse]
    narrative: str


class RiskTransitionSummary(BaseModel):
    """Summary of risk tier transitions."""

    from_tier: str
    to_tier: str
    count: int
    account_ids: list[str]


class AtRiskResponse(BaseModel):
    """At-risk accounts with optional early warning alerts."""

    accounts: list[AtRiskAccount]
    total_high_risk: int
    early_warning: EarlyWarningAlertResponse | None = None
    narrative: str = Field(description="AI narrative for at-risk detail page")


class AtRiskAccountDetail(BaseModel):
    """Detailed view of a single at-risk account."""

    account_id: str
    email: str
    churn_probability: float
    risk_tier: str
    plan_type: str
    tenure_days: int
    shap_explanation: list[dict[str, Any]] = Field(
        default_factory=list,
        description="SHAP feature contributions",
    )
    profile: dict[str, Any] = Field(
        default_factory=dict,
        description="Full account profile",
    )
    support_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent support tickets",
    )
    payment_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent payments",
    )
    viewing_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Streaming behavior summary",
    )


# =============================================================================
# Analytics
# =============================================================================


class NarrativeSectionResponse(BaseModel):
    """One section of the analysis narrative."""

    heading: str
    body: str
    sentiment: str = Field(
        description="Section sentiment: positive, neutral, concerning, critical"
    )


class AnalysisNarrativeResponse(BaseModel):
    """AI-generated narrative for a dashboard page."""

    headline: str
    sections: list[NarrativeSectionResponse]
    overall_sentiment: str = Field(
        description="Overall: healthy, watch_closely, action_needed"
    )
    key_callouts: list[str] = Field(description="2-4 bullet highlights")


class AnalyticsResponse(BaseModel):
    """Full analytics with AI narrative and raw data for charts."""

    narrative: AnalysisNarrativeResponse
    raw_kpis: dict[str, Any] = Field(
        description="All numeric values for chart rendering"
    )
    engagement_trends: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Weekly engagement data points",
    )
    support_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Support ticket health metrics",
    )
    payment_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Payment health metrics",
    )
    subscription_distribution: dict[str, Any] = Field(
        default_factory=dict,
        description="Subscription plan breakdown",
    )
    cohort_analysis: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Churn by signup cohort",
    )


# =============================================================================
# Prescriptions
# =============================================================================


class PrescriptionRecommendation(BaseModel):
    """A single account's intervention recommendation."""

    account_id: str
    churn_probability: float
    risk_tier: str
    recommended_strategy: str = Field(description="Strategy name")
    strategy_description: str
    typical_offer: str | None = None
    priority: int = Field(description="Strategy priority 1-5")
    estimated_save_probability: float = Field(
        description="Estimated likelihood of saving the account"
    )


class PrescriptionResponse(BaseModel):
    """Prescription recommendations for at-risk accounts."""

    recommendations: list[PrescriptionRecommendation]
    total_actionable: int
    narrative: str = Field(description="AI summary of prescription landscape")
    strategy_distribution: dict[str, int] = Field(
        description="Count per strategy type"
    )


# =============================================================================
# Interventions
# =============================================================================


class DraftRequest(BaseModel):
    """Request to draft retention emails."""

    account_id: str | None = None
    account_ids: list[str] | None = None
    churn_driver: str | None = None


class DraftedEmailResponse(BaseModel):
    """A single drafted email variant."""

    variant: str = Field(description="A or B")
    subject: str
    preview_text: str
    body: str
    cta_text: str
    tone: str


class InterventionDraftResponse(BaseModel):
    """Drafted retention emails from the Intervention Drafter agent."""

    churn_driver: str
    strategy_name: str
    emails: list[DraftedEmailResponse]
    account_context_summary: str
    confidence: float = Field(description="Agent confidence 0-1")


class ExportRequest(BaseModel):
    """Request to render and export an email."""

    email_variant: str = Field(description="A or B")
    format: str = Field(description="html, plaintext, or markdown")
    integration: str | None = Field(
        default=None,
        description="Target: email, hubspot, salesforce, marketo, braze, or None",
    )


class ExportResponse(BaseModel):
    """Rendered email with optional integration payload."""

    rendered_content: str
    format: str
    integration_payload: dict[str, Any] | None = None
    integration_instructions: str | None = None


class IntegrationInfo(BaseModel):
    """Metadata about a supported integration."""

    name: str
    description: str
    setup_url: str
    required_config: list[str]
