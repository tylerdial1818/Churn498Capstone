"""
Tests for the agent-backed FastAPI endpoints (Phase 4).

All tests mock agent convenience functions to avoid DB/LLM dependencies.

Run with: pytest tests/test_app_agents.py -v
"""

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["DEMO_MODE"] = "true"

from src.app.main import create_app


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        yield c


# =============================================================================
# Mock fixtures for agent results
# =============================================================================


def _mock_analysis_narrative(page_context="executive_summary"):
    """Build a mock AnalysisNarrative-like object."""
    section = MagicMock()
    section.heading = "Churn Overview"
    section.body = "The 30-day churn rate stands at 4.9%, within healthy range."
    section.sentiment = "neutral"
    section.kpis_referenced = ["churn_rate_30d"]

    narrative = MagicMock()
    narrative.page_context = page_context
    narrative.generated_at = "2025-12-01T00:00:00"
    narrative.reference_date = "2025-12-01"
    narrative.headline = "Platform health is stable with 4.9% churn rate"
    narrative.sections = [section]
    narrative.overall_sentiment = "healthy"
    narrative.key_callouts = [
        "Churn rate 4.9% — within healthy range",
        "2,420 accounts in high-risk tier",
    ]
    narrative.raw_kpis = {
        "churn_rate_30d": 4.9,
        "retention_rate_30d": 95.1,
        "high_risk_count": 2420,
        "monthly_recurring_revenue": 33880.0,
        "avg_customer_lifetime_days": 400.0,
    }
    narrative.metadata = {}
    narrative.success = True
    narrative.errors = []
    return narrative


def _mock_early_warning_report():
    """Build a mock EarlyWarningReport-like object."""
    group = MagicMock()
    group.root_cause = "payment_issues"
    group.accounts = [MagicMock()] * 15
    group.representative_account_ids = ["ACC_00000001", "ACC_00000002", "ACC_00000003"]
    group.evidence_summary = "Payment failures across 15 accounts"
    group.priority = 4
    group.recommended_action = "Trigger payment recovery workflow"

    report = MagicMock()
    report.headline = "47 accounts escalated — payment issues drives 32%"
    report.total_escalated = 47
    report.total_new_high_risk = 30
    report.total_improved = 12
    report.alert_groups = [group]
    report.narrative = "Between Nov and Dec, 47 accounts escalated."
    report.model_health = "Healthy"
    report.success = True
    report.errors = []
    return report


def _mock_intervention_result():
    """Build a mock InterventionResult-like object."""
    email_a = MagicMock()
    email_a.variant = "A"
    email_a.subject = "A special offer just for you"
    email_a.preview_text = "Save 20% on your next 3 months"
    email_a.body = "Hi there, we appreciate your loyalty..."
    email_a.cta_text = "Claim Your Discount"
    email_a.cta_url = "https://retain.example.com/redeem"
    email_a.tone = "value-focused"
    email_a.personalization_fields_used = ["name", "plan_type"]

    email_b = MagicMock()
    email_b.variant = "B"
    email_b.subject = "We'd love to keep you"
    email_b.preview_text = "Your exclusive deal inside"
    email_b.body = "We noticed your subscription..."
    email_b.cta_text = "Activate 20% Off"
    email_b.cta_url = "https://retain.example.com/redeem"
    email_b.tone = "empathetic"
    email_b.personalization_fields_used = ["name"]

    strategy = MagicMock()
    strategy.name = "win_back_discount"
    strategy.description = "Price-sensitive discount offer"
    strategy.typical_offer = "20% off next 3 months"

    result = MagicMock()
    result.account_id = "ACC_00000001"
    result.churn_driver = "price_sensitivity"
    result.strategy = strategy
    result.emails = [email_a, email_b]
    result.account_context_summary = "Long-term Basic plan subscriber showing price sensitivity"
    result.confidence = 0.82
    return result


# =============================================================================
# TestDashboardEndpoint
# =============================================================================


class TestDashboardEndpoint:
    """Tests for GET /api/dashboard."""

    @patch("src.agents.analysis.analyzer.run_analysis")
    def test_dashboard_returns_200(self, mock_analysis, client):
        mock_analysis.return_value = _mock_analysis_narrative()
        resp = client.get("/api/dashboard")
        assert resp.status_code == 200

    @patch("src.agents.analysis.analyzer.run_analysis")
    def test_dashboard_has_kpis(self, mock_analysis, client):
        mock_analysis.return_value = _mock_analysis_narrative()
        data = client.get("/api/dashboard").json()
        assert "kpis" in data
        assert isinstance(data["kpis"], list)

    @patch("src.agents.analysis.analyzer.run_analysis")
    def test_dashboard_has_narrative(self, mock_analysis, client):
        mock_analysis.return_value = _mock_analysis_narrative()
        data = client.get("/api/dashboard").json()
        assert data["headline"]
        assert data["narrative"]
        assert data["overall_sentiment"] in ("healthy", "watch_closely", "action_needed")

    @patch("src.agents.analysis.analyzer.run_analysis")
    def test_dashboard_kpi_health_values(self, mock_analysis, client):
        mock_analysis.return_value = _mock_analysis_narrative()
        data = client.get("/api/dashboard").json()
        for kpi in data["kpis"]:
            assert kpi["health"] in ("healthy", "warning", "critical")


# =============================================================================
# TestAtRiskEndpoint
# =============================================================================


class TestAtRiskEndpoint:
    """Tests for GET /api/at-risk."""

    @patch("src.agents.analysis.analyzer.run_analysis")
    @patch("src.agents.early_warning.detector.run_early_warning")
    @patch("src.agents.tools.get_account_risk_scores")
    def test_at_risk_returns_accounts(
        self, mock_scores, mock_ew, mock_analysis, client
    ):
        mock_scores.invoke = MagicMock(
            return_value="| account_id | churn_probability | risk_tier |\n| --- | --- | --- |\n| ACC_00000001 | 0.85 | high |"
        )
        mock_ew.return_value = _mock_early_warning_report()
        mock_analysis.return_value = _mock_analysis_narrative("at_risk_detail")

        resp = client.get("/api/at-risk")
        assert resp.status_code == 200
        data = resp.json()
        assert "accounts" in data
        assert "total_high_risk" in data

    @patch("src.agents.analysis.analyzer.run_analysis")
    @patch("src.agents.early_warning.detector.run_early_warning")
    @patch("src.agents.tools.get_account_risk_scores")
    def test_at_risk_with_early_warning(
        self, mock_scores, mock_ew, mock_analysis, client
    ):
        mock_scores.invoke = MagicMock(return_value="No risk scores found")
        mock_ew.return_value = _mock_early_warning_report()
        mock_analysis.return_value = _mock_analysis_narrative("at_risk_detail")

        data = client.get("/api/at-risk?include_early_warning=true").json()
        assert data["early_warning"] is not None
        assert data["early_warning"]["total_escalated"] == 47

    @patch("src.agents.analysis.analyzer.run_analysis")
    @patch("src.agents.tools.get_account_risk_scores")
    def test_at_risk_without_early_warning(
        self, mock_scores, mock_analysis, client
    ):
        mock_scores.invoke = MagicMock(return_value="No risk scores found")
        mock_analysis.return_value = _mock_analysis_narrative("at_risk_detail")

        data = client.get("/api/at-risk?include_early_warning=false").json()
        assert data["early_warning"] is None


# =============================================================================
# TestAnalyticsEndpoint
# =============================================================================


class TestAnalyticsEndpoint:
    """Tests for GET /api/analytics."""

    @patch("src.agents.tools.get_churn_cohort_analysis")
    @patch("src.agents.tools.get_subscription_distribution")
    @patch("src.agents.tools.get_payment_health")
    @patch("src.agents.tools.get_support_health")
    @patch("src.agents.tools.get_engagement_trends")
    @patch("src.agents.analysis.analyzer.run_analysis")
    def test_analytics_returns_raw_kpis(
        self, mock_analysis, mock_eng, mock_sup, mock_pay, mock_sub, mock_coh, client
    ):
        mock_analysis.return_value = _mock_analysis_narrative("analytics_deep_dive")
        mock_eng.invoke = MagicMock(return_value='{"weekly_data": []}')
        mock_sup.invoke = MagicMock(return_value='{}')
        mock_pay.invoke = MagicMock(return_value='{}')
        mock_sub.invoke = MagicMock(return_value='{}')
        mock_coh.invoke = MagicMock(return_value='{"cohorts": []}')

        data = client.get("/api/analytics").json()
        assert "raw_kpis" in data
        assert isinstance(data["raw_kpis"], dict)

    @patch("src.agents.tools.get_churn_cohort_analysis")
    @patch("src.agents.tools.get_subscription_distribution")
    @patch("src.agents.tools.get_payment_health")
    @patch("src.agents.tools.get_support_health")
    @patch("src.agents.tools.get_engagement_trends")
    @patch("src.agents.analysis.analyzer.run_analysis")
    def test_analytics_has_narrative_sections(
        self, mock_analysis, mock_eng, mock_sup, mock_pay, mock_sub, mock_coh, client
    ):
        mock_analysis.return_value = _mock_analysis_narrative("analytics_deep_dive")
        mock_eng.invoke = MagicMock(return_value='{"weekly_data": []}')
        mock_sup.invoke = MagicMock(return_value='{}')
        mock_pay.invoke = MagicMock(return_value='{}')
        mock_sub.invoke = MagicMock(return_value='{}')
        mock_coh.invoke = MagicMock(return_value='{"cohorts": []}')

        data = client.get("/api/analytics").json()
        assert "narrative" in data
        assert "sections" in data["narrative"]
        assert "overall_sentiment" in data["narrative"]

    @patch("src.agents.tools.get_churn_cohort_analysis")
    @patch("src.agents.tools.get_subscription_distribution")
    @patch("src.agents.tools.get_payment_health")
    @patch("src.agents.tools.get_support_health")
    @patch("src.agents.tools.get_engagement_trends")
    @patch("src.agents.analysis.analyzer.run_analysis")
    def test_analytics_engagement_trends_format(
        self, mock_analysis, mock_eng, mock_sup, mock_pay, mock_sub, mock_coh, client
    ):
        mock_analysis.return_value = _mock_analysis_narrative("analytics_deep_dive")
        mock_eng.invoke = MagicMock(
            return_value='{"weekly_data": [{"week": "2025-10-06", "total_watch_hours": 1200}]}'
        )
        mock_sup.invoke = MagicMock(return_value='{}')
        mock_pay.invoke = MagicMock(return_value='{}')
        mock_sub.invoke = MagicMock(return_value='{}')
        mock_coh.invoke = MagicMock(return_value='{"cohorts": []}')

        data = client.get("/api/analytics").json()
        assert isinstance(data["engagement_trends"], list)


# =============================================================================
# TestPrescriptionsEndpoint
# =============================================================================


class TestPrescriptionsEndpoint:
    """Tests for GET /api/prescriptions."""

    @patch("src.agents.analysis.analyzer.run_analysis")
    @patch("src.agents.intervention.strategies.select_strategy")
    @patch("src.agents.intervention.strategies.STRATEGY_REGISTRY")
    @patch("src.agents.tools.get_account_risk_scores")
    def test_prescriptions_returns_recommendations(
        self, mock_scores, mock_registry, mock_select, mock_analysis, client
    ):
        mock_scores.invoke = MagicMock(
            return_value=(
                "| account_id | churn_probability | risk_tier | plan_type |\n"
                "| --- | --- | --- | --- |\n"
                "| ACC_00000001 | 0.85 | high | Basic |"
            )
        )
        mock_strategy = MagicMock()
        mock_strategy.name = "engagement_reignite"
        mock_strategy.description = "Re-engage dormant users"
        mock_strategy.typical_offer = "Free premium trial"
        mock_strategy.priority = 3
        mock_select.return_value = mock_strategy
        mock_analysis.return_value = _mock_analysis_narrative("prescription_summary")

        data = client.get("/api/prescriptions").json()
        assert "recommendations" in data
        assert "total_actionable" in data
        assert isinstance(data["strategy_distribution"], dict)

    @patch("src.agents.analysis.analyzer.run_analysis")
    @patch("src.agents.tools.get_account_risk_scores")
    def test_prescriptions_has_narrative(
        self, mock_scores, mock_analysis, client
    ):
        mock_scores.invoke = MagicMock(return_value="No risk scores found")
        mock_analysis.return_value = _mock_analysis_narrative("prescription_summary")

        data = client.get("/api/prescriptions").json()
        assert "narrative" in data
        assert data["narrative"]  # not empty


# =============================================================================
# TestInterventionsEndpoint
# =============================================================================


class TestInterventionsEndpoint:
    """Tests for intervention draft, export, and integrations."""

    @patch("src.agents.intervention.drafter.draft_intervention")
    def test_draft_single_account(self, mock_draft, client):
        mock_draft.return_value = _mock_intervention_result()
        resp = client.post(
            "/api/interventions/draft",
            json={"account_id": "ACC_00000001"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["churn_driver"] == "price_sensitivity"
        assert data["strategy_name"] == "win_back_discount"
        assert len(data["emails"]) == 2

    @patch("src.agents.intervention.drafter.draft_intervention")
    def test_draft_with_churn_driver(self, mock_draft, client):
        mock_draft.return_value = _mock_intervention_result()
        resp = client.post(
            "/api/interventions/draft",
            json={
                "account_id": "ACC_00000001",
                "churn_driver": "price_sensitivity",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == pytest.approx(0.82)

    @patch("src.agents.intervention.drafter.draft_intervention")
    def test_draft_cohort(self, mock_draft, client):
        mock_draft.return_value = _mock_intervention_result()
        resp = client.post(
            "/api/interventions/draft",
            json={"account_ids": ["ACC_00000001", "ACC_00000002"]},
        )
        assert resp.status_code == 200

    @patch("src.agents.intervention.email_renderer.render_as_html")
    @patch("src.agents.intervention.drafter.draft_intervention")
    def test_export_html(self, mock_draft, mock_render, client):
        result = _mock_intervention_result()
        mock_draft.return_value = result

        # Draft first to populate cache
        client.post(
            "/api/interventions/draft",
            json={"account_id": "ACC_00000001"},
        )

        mock_render.return_value = "<html><body>Email content</body></html>"
        resp = client.post(
            "/api/interventions/export",
            json={"email_variant": "A", "format": "html"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "html"
        assert data["rendered_content"]

    @patch("src.agents.intervention.email_renderer.render_as_plaintext")
    @patch("src.agents.intervention.drafter.draft_intervention")
    def test_export_plaintext(self, mock_draft, mock_render, client):
        result = _mock_intervention_result()
        mock_draft.return_value = result

        client.post(
            "/api/interventions/draft",
            json={"account_id": "ACC_00000001"},
        )

        mock_render.return_value = "Plain text email content"
        resp = client.post(
            "/api/interventions/export",
            json={"email_variant": "A", "format": "plaintext"},
        )
        assert resp.status_code == 200

    @patch("src.agents.intervention.email_renderer.render_as_html")
    @patch("src.agents.intervention.drafter.draft_intervention")
    def test_export_hubspot_payload(self, mock_draft, mock_render, client):
        result = _mock_intervention_result()
        mock_draft.return_value = result

        client.post(
            "/api/interventions/draft",
            json={"account_id": "ACC_00000001"},
        )

        mock_render.return_value = "<html>Email</html>"
        resp = client.post(
            "/api/interventions/export",
            json={
                "email_variant": "A",
                "format": "html",
                "integration": "hubspot",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["integration_payload"] is not None
        assert "emailId" in data["integration_payload"]

    @patch("src.agents.intervention.email_renderer.render_as_html")
    @patch("src.agents.intervention.drafter.draft_intervention")
    def test_export_salesforce_payload(self, mock_draft, mock_render, client):
        result = _mock_intervention_result()
        mock_draft.return_value = result

        client.post(
            "/api/interventions/draft",
            json={"account_id": "ACC_00000001"},
        )

        mock_render.return_value = "<html>Email</html>"
        resp = client.post(
            "/api/interventions/export",
            json={
                "email_variant": "A",
                "format": "html",
                "integration": "salesforce",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["integration_payload"] is not None
        assert "definitionKey" in data["integration_payload"]

    @patch("src.agents.intervention.email_renderer.render_as_markdown")
    @patch("src.agents.intervention.drafter.draft_intervention")
    def test_export_no_integration(self, mock_draft, mock_render, client):
        result = _mock_intervention_result()
        mock_draft.return_value = result

        client.post(
            "/api/interventions/draft",
            json={"account_id": "ACC_00000001"},
        )

        mock_render.return_value = "# Email markdown"
        resp = client.post(
            "/api/interventions/export",
            json={"email_variant": "A", "format": "markdown"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["integration_payload"] is None
        assert data["integration_instructions"] is not None

    def test_list_integrations(self, client):
        resp = client.get("/api/interventions/integrations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5
        names = {item["name"] for item in data}
        assert names == {"email", "hubspot", "salesforce", "marketo", "braze"}
        for item in data:
            assert "description" in item
            assert "required_config" in item


# =============================================================================
# TestErrorHandling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and graceful degradation."""

    @patch("src.agents.analysis.analyzer.run_analysis")
    def test_agent_failure_returns_partial_result(self, mock_analysis, client):
        """Dashboard returns partial result on agent failure, not 500."""
        mock_analysis.side_effect = RuntimeError("LLM unavailable")
        resp = client.get("/api/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "unavailable" in data["headline"].lower()

    def test_export_without_draft_returns_404(self, client):
        """Export before drafting returns 404."""
        # Clear cache
        from src.app.routes.interventions import _last_draft_result
        _last_draft_result.clear()

        resp = client.post(
            "/api/interventions/export",
            json={"email_variant": "X", "format": "html"},
        )
        assert resp.status_code == 404
