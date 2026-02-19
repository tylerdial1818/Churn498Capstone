"""Tests for /api/v1/accounts endpoints."""


class TestAccountEndpoints:
    """Account endpoint tests against demo-mode fixture data."""

    def test_get_at_risk_returns_paginated(self, client):
        r = client.get("/api/v1/accounts/at-risk")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) > 0

    def test_get_at_risk_default_pagination(self, client):
        data = client.get("/api/v1/accounts/at-risk").json()
        assert data["page"] == 1
        assert data["per_page"] == 20

    def test_get_at_risk_custom_page_size(self, client):
        data = client.get("/api/v1/accounts/at-risk?per_page=5").json()
        assert data["per_page"] == 5
        assert len(data["items"]) <= 5

    def test_get_at_risk_filter_by_risk_tier_high(self, client):
        data = client.get("/api/v1/accounts/at-risk?risk_tier=high").json()
        for item in data["items"]:
            assert item["risk_tier"] == "high"

    def test_get_at_risk_filter_by_risk_tier_medium(self, client):
        data = client.get("/api/v1/accounts/at-risk?risk_tier=medium").json()
        for item in data["items"]:
            assert item["risk_tier"] == "medium"

    def test_get_at_risk_accounts_sorted_by_churn(self, client):
        data = client.get("/api/v1/accounts/at-risk").json()
        probs = [a["churn_probability"] for a in data["items"]]
        assert probs == sorted(probs, reverse=True)

    def test_get_at_risk_account_schema(self, client):
        data = client.get("/api/v1/accounts/at-risk").json()
        account = data["items"][0]
        assert "account_id" in account
        assert "email" in account
        assert "churn_probability" in account
        assert "risk_tier" in account
        assert "plan_type" in account
        assert "tenure_days" in account
        assert "top_drivers" in account
        assert isinstance(account["top_drivers"], list)

    def test_get_account_detail_valid_id(self, client):
        r = client.get("/api/v1/accounts/ACC_00004217")
        assert r.status_code == 200
        data = r.json()
        assert data["account_id"] == "ACC_00004217"
        assert "payment_history" in data
        assert "ticket_history" in data
        assert "agent_narrative" in data

    def test_get_account_detail_has_extended_fields(self, client):
        data = client.get("/api/v1/accounts/ACC_00004217").json()
        assert "signup_date" in data
        assert "country" in data
        assert "age" in data
        assert "subscription_status" in data

    def test_get_account_detail_invalid_id_404(self, client):
        r = client.get("/api/v1/accounts/NONEXISTENT")
        assert r.status_code == 404

    def test_get_shap_returns_dict(self, client):
        r = client.get("/api/v1/accounts/ACC_00004217/shap")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert len(data) > 0
        for key, value in data.items():
            assert isinstance(key, str)
            assert isinstance(value, float)

    def test_get_shap_invalid_id_404(self, client):
        r = client.get("/api/v1/accounts/NONEXISTENT/shap")
        assert r.status_code == 404
