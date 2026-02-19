"""Tests for Pydantic schema validation."""

from src.app.schemas import (
    AccountAtRisk,
    AccountDetail,
    AgentInsight,
    AnalyticsOverview,
    DriftFeature,
    DriftStatus,
    InterventionDraft,
    KPIResponse,
    ModelMetrics,
    PlanBreakdown,
    RiskDistribution,
    SHAPFeature,
    TrendPoint,
)


class TestSchemaValidation:
    """Test that Pydantic schemas validate correctly."""

    def test_kpi_response_validates(self):
        kpi = KPIResponse(
            total_accounts=60000,
            active_subscribers=49684,
            churned_accounts=10316,
            churn_rate_30d=0.049,
            high_risk_count=2420,
            at_risk_mrr=33880.0,
            cac=45.0,
            retention_cost_per_save=12.5,
        )
        assert kpi.total_accounts == 60000

    def test_trend_point_validates(self):
        t = TrendPoint(
            month="2024-01",
            signups=2810,
            cancellations=385,
            net_growth=2425,
            churn_rate=0.031,
        )
        assert t.month == "2024-01"

    def test_risk_distribution_validates(self):
        r = RiskDistribution(low=53760, medium=3820, high=2420)
        assert r.low + r.medium + r.high == 60000

    def test_account_at_risk_validates(self):
        a = AccountAtRisk(
            account_id="ACC_001",
            email="test@test.com",
            churn_probability=0.85,
            risk_tier="high",
            plan_type="Regular",
            tenure_days=200,
            last_payment_days=10,
            last_stream_days=15,
            open_tickets=2,
            top_drivers=["payment_failure_rate", "tenure_days"],
        )
        assert a.churn_probability == 0.85
        assert a.shap_values is None

    def test_account_at_risk_with_shap(self):
        a = AccountAtRisk(
            account_id="ACC_001",
            email="test@test.com",
            churn_probability=0.85,
            risk_tier="high",
            plan_type="Regular",
            tenure_days=200,
            last_payment_days=10,
            last_stream_days=15,
            open_tickets=2,
            top_drivers=["payment_failure_rate"],
            shap_values={"payment_failure_rate": 0.15},
        )
        assert a.shap_values["payment_failure_rate"] == 0.15

    def test_agent_insight_validates(self):
        ai = AgentInsight(
            agent_name="Analysis Agent",
            content="Test content",
            generated_at="2025-12-01T10:00:00",
            status="complete",
        )
        assert ai.agent_name == "Analysis Agent"

    def test_intervention_draft_validates(self):
        i = InterventionDraft(
            id="test-uuid",
            account_id="ACC_001",
            strategy="payment_recovery",
            status="pending",
            subject="Test subject",
            body_html="<p>Test</p>",
            body_plaintext="Test",
            agent_rationale="Test rationale",
            created_at="2025-12-01T10:00:00",
            updated_at="2025-12-01T10:00:00",
        )
        assert i.status == "pending"

    def test_shap_feature_validates(self):
        f = SHAPFeature(
            feature="payment_failure_rate",
            importance=0.142,
            direction="positive",
        )
        assert f.importance == 0.142

    def test_plan_breakdown_validates(self):
        p = PlanBreakdown(
            plan_type="Regular",
            total=38400,
            churned=7104,
            churn_rate=0.058,
        )
        assert p.churn_rate == 0.058

    def test_model_metrics_validates(self):
        m = ModelMetrics(
            auc_roc=0.891,
            precision=0.823,
            recall=0.756,
            f1_score=0.788,
            accuracy=0.862,
        )
        assert 0 <= m.auc_roc <= 1
        assert m.calibration_error is None

    def test_drift_feature_validates(self):
        d = DriftFeature(feature="tenure_days", psi=0.015, status="stable")
        assert d.psi == 0.015

    def test_drift_status_validates(self):
        ds = DriftStatus(
            overall_status="healthy",
            features=[
                DriftFeature(feature="tenure_days", psi=0.015, status="stable"),
            ],
            checked_at="2025-12-01T08:00:00",
        )
        assert len(ds.features) == 1

    def test_analytics_overview_validates(self):
        ao = AnalyticsOverview(
            kpis=KPIResponse(
                total_accounts=60000,
                active_subscribers=49684,
                churned_accounts=10316,
                churn_rate_30d=0.049,
                high_risk_count=2420,
                at_risk_mrr=33880.0,
                cac=45.0,
                retention_cost_per_save=12.5,
            ),
            risk_distribution=RiskDistribution(low=53760, medium=3820, high=2420),
            top_shap_features=[],
            plan_breakdown=[],
        )
        assert ao.kpis.total_accounts == 60000

    def test_account_detail_validates(self):
        ad = AccountDetail(
            account_id="ACC_001",
            email="test@test.com",
            signup_date="2025-03-15",
            country="US",
            age=34,
            gender="F",
            plan_type="Regular",
            subscription_status="active",
            tenure_days=200,
            churn_probability=0.85,
            risk_tier="high",
            top_drivers=["payment_failure_rate"],
            payment_history=[],
            ticket_history=[],
            last_payment_days=10,
            last_stream_days=15,
            open_tickets=2,
            agent_narrative="Test narrative",
        )
        assert ad.subscription_status == "active"
