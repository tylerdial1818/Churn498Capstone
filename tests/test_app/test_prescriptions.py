"""Tests for /api/v1/prescriptions endpoints."""


class TestPrescriptionEndpoints:
    """Prescription endpoint tests against demo-mode fixture data."""

    def test_get_prescriptions_returns_200(self, client):
        r = client.get("/api/v1/prescriptions")
        assert r.status_code == 200

    def test_get_prescriptions_returns_groups(self, client):
        data = client.get("/api/v1/prescriptions").json()
        assert isinstance(data, list)
        assert len(data) == 4

    def test_get_prescriptions_group_schema(self, client):
        data = client.get("/api/v1/prescriptions").json()
        group = data[0]
        assert "strategy" in group
        assert "display_name" in group
        assert "account_count" in group
        assert "estimated_mrr" in group
        assert "accounts" in group

    def test_get_prescriptions_groups_have_accounts(self, client):
        data = client.get("/api/v1/prescriptions").json()
        for group in data:
            assert isinstance(group["accounts"], list)
            assert group["account_count"] > 0

    def test_get_prescriptions_expected_strategies(self, client):
        data = client.get("/api/v1/prescriptions").json()
        strategies = {g["strategy"] for g in data}
        assert "payment_recovery" in strategies
        assert "engagement_reignite" in strategies
        assert "vip_support_rescue" in strategies
        assert "content_discovery" in strategies

    def test_get_prescription_by_strategy_valid(self, client):
        r = client.get("/api/v1/prescriptions/payment_recovery")
        assert r.status_code == 200
        data = r.json()
        assert data["strategy"] == "payment_recovery"
        assert data["display_name"] == "Payment Recovery"

    def test_get_prescription_by_strategy_has_accounts(self, client):
        data = client.get("/api/v1/prescriptions/payment_recovery").json()
        assert len(data["accounts"]) > 0

    def test_get_prescription_by_strategy_invalid_404(self, client):
        r = client.get("/api/v1/prescriptions/nonexistent_strategy")
        assert r.status_code == 404
