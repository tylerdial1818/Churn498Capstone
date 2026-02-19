"""
Pydantic response and request models for Retain API.

Every endpoint has a typed schema. TypeScript mirrors live in
frontend/src/api/types.ts.
"""

from pydantic import BaseModel, ConfigDict


# =============================================================================
# Dashboard
# =============================================================================


class KPIResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_accounts: int
    active_subscribers: int
    churned_accounts: int
    churn_rate_30d: float
    high_risk_count: int
    at_risk_mrr: float
    cac: float
    retention_cost_per_save: float


class TrendPoint(BaseModel):
    month: str
    signups: int
    cancellations: int
    net_growth: int
    churn_rate: float


class RiskDistribution(BaseModel):
    low: int
    medium: int
    high: int


class ActiveInactiveDistribution(BaseModel):
    active: int
    inactive: int
    active_pct: float
    inactive_pct: float


class AgentInsight(BaseModel):
    agent_name: str
    content: str
    generated_at: str
    status: str


# =============================================================================
# Accounts
# =============================================================================


class AccountAtRisk(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    account_id: str
    email: str
    churn_probability: float
    risk_tier: str
    plan_type: str
    tenure_days: int
    last_payment_days: int
    last_stream_days: int
    open_tickets: int
    top_drivers: list[str]
    shap_values: dict[str, float] | None = None


class PaginatedAccounts(BaseModel):
    items: list[AccountAtRisk]
    total: int
    page: int
    per_page: int


class PaymentRecord(BaseModel):
    payment_id: str
    payment_date: str
    amount: float
    currency: str
    payment_method: str
    status: str
    failure_reason: str | None = None


class TicketRecord(BaseModel):
    ticket_id: str
    created_at: str
    category: str
    priority: str
    resolved_at: str | None = None


class AccountDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    account_id: str
    email: str
    signup_date: str
    country: str
    age: int
    gender: str
    plan_type: str
    subscription_status: str
    tenure_days: int
    churn_probability: float
    risk_tier: str
    top_drivers: list[str]
    shap_values: dict[str, float] | None = None
    payment_history: list[PaymentRecord]
    ticket_history: list[TicketRecord]
    last_payment_days: int
    last_stream_days: int
    open_tickets: int
    agent_narrative: str


# =============================================================================
# Analytics
# =============================================================================


class SHAPFeature(BaseModel):
    feature: str
    importance: float
    direction: str


class PlanBreakdown(BaseModel):
    plan_type: str
    total: int
    churned: int
    churn_rate: float


class SegmentBreakdown(BaseModel):
    by_plan: list[PlanBreakdown]
    by_tenure: list[dict]
    by_payment_method: list[dict]


class ModelMetrics(BaseModel):
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    calibration_error: float | None = None
    prediction_distribution: list[dict] | None = None


class DriftFeature(BaseModel):
    feature: str
    psi: float
    status: str


class DriftStatus(BaseModel):
    overall_status: str
    features: list[DriftFeature]
    checked_at: str


class AnalyticsOverview(BaseModel):
    kpis: KPIResponse
    risk_distribution: RiskDistribution
    top_shap_features: list[SHAPFeature]
    plan_breakdown: list[PlanBreakdown]


# =============================================================================
# Prescriptions
# =============================================================================


class PrescriptionGroup(BaseModel):
    strategy: str
    display_name: str
    account_count: int
    estimated_mrr: float
    accounts: list[AccountAtRisk]


# =============================================================================
# Interventions
# =============================================================================


class InterventionDraft(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    account_id: str
    strategy: str
    status: str
    subject: str
    body_html: str
    body_plaintext: str
    agent_rationale: str
    created_at: str
    updated_at: str


class DraftRequest(BaseModel):
    account_id: str
    strategy: str


class BatchDraftRequest(BaseModel):
    account_ids: list[str]
    strategy: str


class StatusUpdateRequest(BaseModel):
    status: str


class ContentUpdateRequest(BaseModel):
    subject: str | None = None
    body: str | None = None


# =============================================================================
# Agents
# =============================================================================


class AgentTriggerResponse(BaseModel):
    run_id: str
    status: str


class AgentStatusResponse(BaseModel):
    run_id: str
    status: str
    result: dict | None = None


# =============================================================================
# System
# =============================================================================


class HealthResponse(BaseModel):
    status: str
    demo_mode: bool
    db_connected: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
