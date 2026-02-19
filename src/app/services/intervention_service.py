"""
Intervention service — CRUD for drafted retention emails.
"""

import logging
import uuid
from datetime import datetime

from sqlalchemy import Engine, text

from ..fixtures.demo_data import (
    create_fixture_intervention,
    get_fixture_intervention,
    get_fixture_interventions,
)

logger = logging.getLogger(__name__)

# In-memory store for demo mode interventions
_demo_interventions: list[dict] = []


def _get_demo_store() -> list[dict]:
    """Get demo intervention store, initializing from fixtures if empty."""
    global _demo_interventions
    if not _demo_interventions:
        _demo_interventions = get_fixture_interventions()
    return _demo_interventions


def get_interventions(
    engine: Engine | None,
    demo_mode: bool = False,
    status: str | None = None,
) -> list[dict]:
    """Get intervention drafts, optionally filtered by status."""
    if demo_mode or engine is None:
        store = _get_demo_store()
        if status:
            return [i for i in store if i["status"] == status]
        return list(store)

    try:
        with engine.connect() as conn:
            if status:
                query = text(
                    "SELECT * FROM interventions WHERE status = :status "
                    "ORDER BY created_at DESC"
                )
                rows = conn.execute(query, {"status": status}).mappings().all()
            else:
                query = text(
                    "SELECT * FROM interventions ORDER BY created_at DESC"
                )
                rows = conn.execute(query).mappings().all()

        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_interventions failed: {e}")
        return get_fixture_interventions(status)


def get_intervention(
    engine: Engine | None,
    intervention_id: str,
    demo_mode: bool = False,
) -> dict | None:
    """Get a single intervention by ID."""
    if demo_mode or engine is None:
        store = _get_demo_store()
        return next((i for i in store if i["id"] == intervention_id), None)

    try:
        with engine.connect() as conn:
            query = text("SELECT * FROM interventions WHERE id = :id")
            row = conn.execute(
                query, {"id": intervention_id}
            ).mappings().first()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"get_intervention failed: {e}")
        return get_fixture_intervention(intervention_id)


def create_intervention(
    engine: Engine | None,
    account_id: str,
    strategy: str,
    demo_mode: bool = False,
) -> dict:
    """Create a new intervention draft."""
    if demo_mode or engine is None:
        draft = create_fixture_intervention(account_id, strategy)
        _get_demo_store().insert(0, draft)
        return draft

    try:
        from src.agents.intervention.drafter import draft_intervention
        from src.agents.intervention.email_renderer import (
            render_as_html,
            render_as_plaintext,
        )

        result = draft_intervention(
            account_id=account_id,
            churn_driver=_strategy_to_driver(strategy),
        )

        draft_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        email = result.emails[0] if result.emails else None
        subject = email.subject if email else ""
        body_html = render_as_html(email) if email else ""
        body_plaintext = render_as_plaintext(email) if email else ""

        draft = {
            "id": draft_id,
            "account_id": account_id,
            "strategy": strategy,
            "status": "pending",
            "subject": subject,
            "body_html": body_html,
            "body_plaintext": body_plaintext,
            "agent_rationale": result.account_context_summary,
            "created_at": now,
            "updated_at": now,
        }

        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO interventions
                    (id, account_id, strategy, status, subject,
                     body_html, body_plaintext, agent_rationale)
                    VALUES (:id, :account_id, :strategy, :status, :subject,
                            :body_html, :body_plaintext, :agent_rationale)
                """),
                draft,
            )
            conn.commit()

        return draft
    except Exception as e:
        logger.error(f"create_intervention failed: {e}")
        draft = create_fixture_intervention(account_id, strategy)
        return draft


def batch_create_interventions(
    engine: Engine | None,
    account_ids: list[str],
    strategy: str,
    demo_mode: bool = False,
) -> list[dict]:
    """Create interventions for multiple accounts."""
    results = []
    for aid in account_ids:
        draft = create_intervention(engine, aid, strategy, demo_mode)
        results.append(draft)
    return results


def update_intervention_status(
    engine: Engine | None,
    intervention_id: str,
    new_status: str,
    demo_mode: bool = False,
) -> dict | None:
    """Update intervention status (approve/reject)."""
    if demo_mode or engine is None:
        store = _get_demo_store()
        for item in store:
            if item["id"] == intervention_id:
                item["status"] = new_status
                item["updated_at"] = datetime.now().isoformat()
                return item
        return None

    try:
        now = datetime.now().isoformat()
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE interventions
                    SET status = :status, updated_at = :updated_at
                    WHERE id = :id
                """),
                {"id": intervention_id, "status": new_status, "updated_at": now},
            )
            conn.commit()
        return get_intervention(engine, intervention_id, demo_mode)
    except Exception as e:
        logger.error(f"update_intervention_status failed: {e}")
        return None


def update_intervention_content(
    engine: Engine | None,
    intervention_id: str,
    subject: str | None = None,
    body: str | None = None,
    demo_mode: bool = False,
) -> dict | None:
    """Update intervention email content."""
    if demo_mode or engine is None:
        store = _get_demo_store()
        for item in store:
            if item["id"] == intervention_id:
                if subject is not None:
                    item["subject"] = subject
                if body is not None:
                    item["body_html"] = body
                    item["body_plaintext"] = body
                item["updated_at"] = datetime.now().isoformat()
                return item
        return None

    try:
        now = datetime.now().isoformat()
        updates = ["updated_at = :updated_at"]
        params: dict = {"id": intervention_id, "updated_at": now}

        if subject is not None:
            updates.append("subject = :subject")
            params["subject"] = subject
        if body is not None:
            updates.append("body_html = :body")
            updates.append("body_plaintext = :body_plain")
            params["body"] = body
            params["body_plain"] = body

        with engine.connect() as conn:
            conn.execute(
                text(f"UPDATE interventions SET {', '.join(updates)} WHERE id = :id"),
                params,
            )
            conn.commit()
        return get_intervention(engine, intervention_id, demo_mode)
    except Exception as e:
        logger.error(f"update_intervention_content failed: {e}")
        return None


def _strategy_to_driver(strategy: str) -> str:
    """Map strategy name to churn driver for the drafter."""
    mapping = {
        "payment_recovery": "payment_issues",
        "engagement_reignite": "disengagement",
        "vip_support_rescue": "support_frustration",
        "win_back_discount": "price_sensitivity",
        "content_discovery": "content_gap",
    }
    return mapping.get(strategy, "disengagement")
