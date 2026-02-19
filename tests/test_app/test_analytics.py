"""Tests for /api/v1/analytics endpoints."""


class TestAnalyticsEndpoints:
    """Analytics endpoint tests against demo-mode fixture data."""

    def test_get_overview_returns_200(self, client):
        r = client.get("/api/v1/analytics/overview")
        assert r.status_code == 200

    def test_get_overview_returns_bundle(self, client):
        data = client.get("/api/v1/analytics/overview").json()
        assert "kpis" in data
        assert "risk_distribution" in data
        assert "top_shap_features" in data
        assert "plan_breakdown" in data

    def test_get_overview_kpis_nested(self, client):
        data = client.get("/api/v1/analytics/overview").json()
        kpis = data["kpis"]
        assert kpis["total_accounts"] == 60000
        assert isinstance(kpis["churn_rate_30d"], float)

    def test_get_churn_trends_returns_200(self, client):
        r = client.get("/api/v1/analytics/churn-trends")
        assert r.status_code == 200

    def test_get_churn_trends_default_12_months(self, client):
        data = client.get("/api/v1/analytics/churn-trends").json()
        assert isinstance(data, list)
        assert len(data) == 12

    def test_get_churn_trends_custom_months(self, client):
        data = client.get("/api/v1/analytics/churn-trends?months=6").json()
        assert len(data) == 6

    def test_get_segments_returns_200(self, client):
        r = client.get("/api/v1/analytics/segments")
        assert r.status_code == 200

    def test_get_segments_has_plan_breakdown(self, client):
        data = client.get("/api/v1/analytics/segments").json()
        assert "by_plan" in data
        assert "by_tenure" in data
        assert "by_payment_method" in data
        assert len(data["by_plan"]) > 0

    def test_get_segments_plan_schema(self, client):
        data = client.get("/api/v1/analytics/segments").json()
        plan = data["by_plan"][0]
        assert "plan_type" in plan
        assert "total" in plan
        assert "churned" in plan
        assert "churn_rate" in plan

    def test_get_model_performance_returns_200(self, client):
        r = client.get("/api/v1/analytics/model-performance")
        assert r.status_code == 200

    def test_get_model_performance_has_metrics(self, client):
        data = client.get("/api/v1/analytics/model-performance").json()
        assert "auc_roc" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
        assert "accuracy" in data
        assert 0 <= data["auc_roc"] <= 1
        assert 0 <= data["precision"] <= 1

    def test_get_drift_returns_200(self, client):
        r = client.get("/api/v1/analytics/drift")
        assert r.status_code == 200

    def test_get_drift_has_features(self, client):
        data = client.get("/api/v1/analytics/drift").json()
        assert "overall_status" in data
        assert "features" in data
        assert "checked_at" in data
        assert len(data["features"]) > 0

    def test_get_drift_feature_schema(self, client):
        data = client.get("/api/v1/analytics/drift").json()
        feat = data["features"][0]
        assert "feature" in feat
        assert "psi" in feat
        assert "status" in feat

    def test_get_shap_global_returns_200(self, client):
        r = client.get("/api/v1/analytics/shap-global")
        assert r.status_code == 200

    def test_get_shap_global_returns_features(self, client):
        data = client.get("/api/v1/analytics/shap-global").json()
        assert isinstance(data, list)
        assert len(data) == 10
        feat = data[0]
        assert "feature" in feat
        assert "importance" in feat
        assert "direction" in feat

    def test_get_shap_global_sorted_by_importance(self, client):
        data = client.get("/api/v1/analytics/shap-global").json()
        importances = [f["importance"] for f in data]
        assert importances == sorted(importances, reverse=True)
