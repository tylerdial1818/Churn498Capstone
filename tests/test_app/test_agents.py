"""Tests for /api/v1/agents endpoints."""


class TestAgentEndpoints:
    """Agent endpoint tests against demo-mode fixture data."""

    def test_trigger_agent_returns_200(self, client):
        r = client.post("/api/v1/agents/trigger/ddp")
        assert r.status_code == 200

    def test_trigger_agent_has_run_id(self, client):
        data = client.post("/api/v1/agents/trigger/ddp").json()
        assert "run_id" in data
        assert "status" in data
        assert isinstance(data["run_id"], str)

    def test_get_agent_status(self, client):
        trigger = client.post("/api/v1/agents/trigger/ddp").json()
        run_id = trigger["run_id"]
        r = client.get(f"/api/v1/agents/status/{run_id}")
        assert r.status_code == 200

    def test_get_agent_status_invalid_404(self, client):
        r = client.get("/api/v1/agents/status/nonexistent-run-id")
        assert r.status_code == 404
