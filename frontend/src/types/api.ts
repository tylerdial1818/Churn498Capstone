/* API response types — mirrors backend schemas.py */

export interface KPIResponse {
  total_accounts: number;
  active_subscribers: number;
  churned_accounts: number;
  churn_rate_30d: number;
  high_risk_count: number;
  at_risk_mrr: number;
  cac: number;
  retention_cost_per_save: number;
}

export interface TrendPoint {
  label: string;
  value: number;
  predicted?: number;
}

export interface RiskDistribution {
  low: number;
  medium: number;
  high: number;
}

export interface ActiveInactiveDistribution {
  active: number;
  inactive: number;
  recent_churn_30d: number;
}

export interface AgentInsight {
  title: string;
  content: string;
  timestamp?: string;
}

export interface AccountAtRisk {
  account_id: string;
  email: string;
  churn_probability: number;
  risk_tier: "high" | "medium" | "low";
  plan_type: string;
  tenure_days: number;
  last_payment_days: number;
  last_stream_days: number;
  open_tickets: number;
  top_drivers: string[];
  shap_values?: SHAPFeature[] | null;
}

export interface PaginatedAccounts {
  items: AccountAtRisk[];
  total: number;
  page: number;
  per_page: number;
}

export interface PaymentRecord {
  date: string;
  amount: number;
  status: string;
  method: string;
}

export interface TicketRecord {
  id: string;
  date: string;
  subject: string;
  status: string;
  priority: string;
}

export interface AccountDetail extends AccountAtRisk {
  recent_payments: PaymentRecord[];
  recent_tickets: TicketRecord[];
  watch_hours_30d: number;
  watch_hours_90d: number;
  sessions_30d: number;
  content_categories: string[];
}

export interface SHAPFeature {
  feature: string;
  importance: number;
  direction: "positive" | "negative";
}

export interface PlanBreakdown {
  plan: string;
  count: number;
  churn_rate: number;
}

export interface SegmentBreakdown {
  by_plan: PlanBreakdown[];
  by_tenure: PlanBreakdown[];
}

export interface ModelMetrics {
  auc_roc: number;
  precision: number;
  recall: number;
  f1_score: number;
  log_loss: number;
  calibration_error: number;
  last_trained: string;
  training_samples: number;
}

export interface DriftFeature {
  feature: string;
  psi: number;
  status: "ok" | "warning" | "drift";
}

export interface DriftStatus {
  overall_status: "ok" | "warning" | "drift";
  features: DriftFeature[];
  last_checked: string;
}

export interface AnalyticsOverview {
  kpis: KPIResponse;
  risk_distribution: RiskDistribution;
  top_shap_features: SHAPFeature[];
}

export interface PrescriptionGroup {
  strategy: string;
  display_name: string;
  account_count: number;
  estimated_mrr: number;
  accounts: AccountAtRisk[];
}

export interface InterventionDraft {
  id: string;
  account_id: string;
  strategy: string;
  status: "pending" | "approved" | "sent" | "rejected";
  subject: string;
  body_html: string;
  body_plaintext: string;
  agent_rationale: string;
  created_at: string;
  updated_at: string;
}

export interface AgentTriggerResponse {
  run_id: string;
  pipeline: string;
  status: string;
  message: string;
}

export interface AgentStatusResponse {
  run_id: string;
  pipeline: string;
  status: string;
  started_at: string;
  completed_at?: string;
  result?: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  demo_mode: boolean;
  db_connected: boolean;
}
