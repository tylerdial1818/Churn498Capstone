"""Tests for /api/v1/interventions endpoints."""


class TestInterventionEndpoints:
    """Intervention endpoint tests against demo-mode fixture data."""

    def test_get_interventions_returns_200(self, client):
        r = client.get("/api/v1/interventions")
        assert r.status_code == 200

    def test_get_interventions_returns_list(self, client):
        data = client.get("/api/v1/interventions").json()
        assert isinstance(data, list)
        assert len(data) >= 4  # fixture has 4 interventions

    def test_get_interventions_schema(self, client):
        data = client.get("/api/v1/interventions").json()
        item = data[0]
        assert "id" in item
        assert "account_id" in item
        assert "strategy" in item
        assert "status" in item
        assert "subject" in item
        assert "body_html" in item
        assert "body_plaintext" in item
        assert "agent_rationale" in item
        assert "created_at" in item
        assert "updated_at" in item

    def test_get_interventions_filter_by_pending(self, client):
        data = client.get("/api/v1/interventions?status=pending").json()
        for item in data:
            assert item["status"] == "pending"

    def test_get_interventions_filter_by_approved(self, client):
        data = client.get("/api/v1/interventions?status=approved").json()
        for item in data:
            assert item["status"] == "approved"

    def test_get_interventions_filter_by_sent(self, client):
        data = client.get("/api/v1/interventions?status=sent").json()
        for item in data:
            assert item["status"] == "sent"

    def test_get_intervention_by_id(self, client):
        items = client.get("/api/v1/interventions").json()
        first_id = items[0]["id"]
        r = client.get(f"/api/v1/interventions/{first_id}")
        assert r.status_code == 200
        assert r.json()["id"] == first_id

    def test_get_intervention_invalid_id_404(self, client):
        r = client.get("/api/v1/interventions/nonexistent-uuid")
        assert r.status_code == 404

    def test_create_intervention_returns_draft(self, client):
        r = client.post(
            "/api/v1/interventions/draft",
            json={"account_id": "ACC_00004217", "strategy": "payment_recovery"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["account_id"] == "ACC_00004217"
        assert data["strategy"] == "payment_recovery"
        assert data["status"] == "pending"
        assert "id" in data
        assert "subject" in data
        assert "body_html" in data

    def test_batch_draft_creates_multiple(self, client):
        r = client.post(
            "/api/v1/interventions/batch-draft",
            json={
                "account_ids": ["ACC_00012843", "ACC_00008391"],
                "strategy": "engagement_reignite",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["account_id"] == "ACC_00012843"
        assert data[1]["account_id"] == "ACC_00008391"

    def test_approve_intervention_updates_status(self, client):
        # Create a new intervention to approve
        create_r = client.post(
            "/api/v1/interventions/draft",
            json={"account_id": "ACC_00003674", "strategy": "engagement_reignite"},
        )
        intervention_id = create_r.json()["id"]

        r = client.patch(
            f"/api/v1/interventions/{intervention_id}/status",
            json={"status": "approved"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "approved"

    def test_reject_intervention_updates_status(self, client):
        create_r = client.post(
            "/api/v1/interventions/draft",
            json={"account_id": "ACC_00027156", "strategy": "payment_recovery"},
        )
        intervention_id = create_r.json()["id"]

        r = client.patch(
            f"/api/v1/interventions/{intervention_id}/status",
            json={"status": "rejected"},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "rejected"

    def test_update_content_persists(self, client):
        create_r = client.post(
            "/api/v1/interventions/draft",
            json={"account_id": "ACC_00015829", "strategy": "engagement_reignite"},
        )
        intervention_id = create_r.json()["id"]

        r = client.patch(
            f"/api/v1/interventions/{intervention_id}/content",
            json={"subject": "Updated subject line", "body": "Updated body content"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["subject"] == "Updated subject line"

    def test_update_status_invalid_id_404(self, client):
        r = client.patch(
            "/api/v1/interventions/nonexistent-uuid/status",
            json={"status": "approved"},
        )
        assert r.status_code == 404

    def test_update_content_invalid_id_404(self, client):
        r = client.patch(
            "/api/v1/interventions/nonexistent-uuid/content",
            json={"subject": "test"},
        )
        assert r.status_code == 404
