"""Tests for /api/v1/dashboard endpoints."""


class TestDashboardEndpoints:
    """Dashboard endpoint tests against demo-mode fixture data."""

    def test_get_kpis_returns_200(self, client):
        r = client.get("/api/v1/dashboard/kpis")
        assert r.status_code == 200

    def test_get_kpis_schema_valid(self, client):
        data = client.get("/api/v1/dashboard/kpis").json()
        assert data["total_accounts"] == 60000
        assert data["active_subscribers"] == 49684
        assert data["churned_accounts"] == 10316
        assert isinstance(data["churn_rate_30d"], float)
        assert isinstance(data["high_risk_count"], int)
        assert isinstance(data["at_risk_mrr"], float)
        assert isinstance(data["cac"], float)
        assert isinstance(data["retention_cost_per_save"], float)

    def test_get_kpis_accounts_sum(self, client):
        data = client.get("/api/v1/dashboard/kpis").json()
        assert data["active_subscribers"] + data["churned_accounts"] == data["total_accounts"]

    def test_get_trends_returns_list(self, client):
        r = client.get("/api/v1/dashboard/trends")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_trends_point_schema(self, client):
        data = client.get("/api/v1/dashboard/trends").json()
        point = data[0]
        assert "month" in point
        assert "signups" in point
        assert "cancellations" in point
        assert "net_growth" in point
        assert "churn_rate" in point

    def test_get_trends_net_growth_calculated(self, client):
        data = client.get("/api/v1/dashboard/trends").json()
        for point in data:
            assert point["net_growth"] == point["signups"] - point["cancellations"]

    def test_get_risk_distribution_returns_200(self, client):
        r = client.get("/api/v1/dashboard/risk-distribution")
        assert r.status_code == 200

    def test_get_risk_distribution_sums_to_total(self, client):
        data = client.get("/api/v1/dashboard/risk-distribution").json()
        total = data["low"] + data["medium"] + data["high"]
        kpis = client.get("/api/v1/dashboard/kpis").json()
        assert total == kpis["total_accounts"]

    def test_get_active_inactive_returns_200(self, client):
        r = client.get("/api/v1/dashboard/active-inactive")
        assert r.status_code == 200

    def test_get_active_inactive_sums_to_total(self, client):
        data = client.get("/api/v1/dashboard/active-inactive").json()
        kpis = client.get("/api/v1/dashboard/kpis").json()
        assert data["active"] + data["inactive"] == kpis["total_accounts"]

    def test_get_active_inactive_percentages(self, client):
        data = client.get("/api/v1/dashboard/active-inactive").json()
        assert isinstance(data["active_pct"], float)
        assert isinstance(data["inactive_pct"], float)

    def test_get_executive_summary_returns_200(self, client):
        r = client.get("/api/v1/dashboard/executive-summary")
        assert r.status_code == 200

    def test_get_executive_summary_has_agent_fields(self, client):
        data = client.get("/api/v1/dashboard/executive-summary").json()
        assert "agent_name" in data
        assert "content" in data
        assert "generated_at" in data
        assert "status" in data
        assert data["status"] == "complete"
        assert data["agent_name"] == "Analysis Agent"
