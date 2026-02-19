"""Tests for /api/v1/health endpoint."""


class TestHealthEndpoint:
    """Health check endpoint tests."""

    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"
        assert data["demo_mode"] is True
        assert isinstance(data["db_connected"], bool)
