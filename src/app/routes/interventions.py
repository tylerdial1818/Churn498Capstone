"""
Interventions route — draft, export, and list integrations.

POST /api/interventions/draft — draft retention emails via Intervention Drafter.
POST /api/interventions/export — render email + build integration payload.
GET  /api/interventions/integrations — list supported integration targets.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.agents.config import AgentConfig

from ..agent_schemas import (
    DraftedEmailResponse,
    DraftRequest,
    ExportRequest,
    ExportResponse,
    IntegrationInfo,
    InterventionDraftResponse,
)
from ..dependencies import get_demo_mode

logger = logging.getLogger("retain.app.routes.interventions")

router = APIRouter(tags=["Interventions"])

# Module-level cache for most recent draft (for export)
_last_draft_result: dict[str, Any] = {}


@router.post(
    "/interventions/draft",
    response_model=InterventionDraftResponse,
)
async def draft_intervention_email(
    request: DraftRequest,
    demo_mode: bool = Depends(get_demo_mode),
) -> InterventionDraftResponse:
    """Draft personalized retention emails using the Intervention Drafter agent.

    Provide either a single account_id or a list of account_ids.
    Optionally specify a churn_driver to skip root cause diagnosis.
    """
    global _last_draft_result

    try:
        from src.agents.intervention.drafter import draft_intervention

        result = draft_intervention(
            account_id=request.account_id,
            account_ids=request.account_ids,
            churn_driver=request.churn_driver,
        )

        emails = [
            DraftedEmailResponse(
                variant=e.variant,
                subject=e.subject,
                preview_text=e.preview_text,
                body=e.body,
                cta_text=e.cta_text,
                tone=e.tone,
            )
            for e in result.emails
        ]

        # Cache for export
        _last_draft_result = {
            "emails": result.emails,
            "strategy": result.strategy,
        }

        return InterventionDraftResponse(
            churn_driver=result.churn_driver,
            strategy_name=result.strategy.name,
            emails=emails,
            account_context_summary=result.account_context_summary,
            confidence=result.confidence,
        )

    except Exception as e:
        logger.error(f"Draft intervention failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Intervention drafting failed: {e}",
        )


@router.post(
    "/interventions/export",
    response_model=ExportResponse,
)
async def export_intervention(
    request: ExportRequest,
    demo_mode: bool = Depends(get_demo_mode),
) -> ExportResponse:
    """Render a drafted email in the specified format and optionally
    prepare an integration payload for external systems.
    """
    from src.agents.intervention.email_renderer import (
        render_as_html,
        render_as_markdown,
        render_as_plaintext,
    )

    # Find the email variant from cache
    cached_emails = _last_draft_result.get("emails", [])
    email = None
    for e in cached_emails:
        if e.variant == request.email_variant:
            email = e
            break

    if email is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Email variant '{request.email_variant}' not found. "
                "Draft an intervention first."
            ),
        )

    # Render in requested format
    if request.format == "html":
        rendered = render_as_html(email)
    elif request.format == "plaintext":
        rendered = render_as_plaintext(email)
    elif request.format == "markdown":
        rendered = render_as_markdown(email)
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format: {request.format}",
        )

    # Build integration payload
    integration_payload: dict[str, Any] | None = None
    integration_instructions: str | None = None

    if request.integration:
        integration_payload = _build_integration_payload(
            email, rendered, request.integration, request.format
        )
    else:
        integration_instructions = (
            "To send this email manually:\n"
            "1. Copy the rendered content above\n"
            "2. Paste into your email client or marketing platform\n"
            "3. Replace any placeholder values (e.g., {name})\n"
            "4. Review the subject line and CTA before sending"
        )

    return ExportResponse(
        rendered_content=rendered,
        format=request.format,
        integration_payload=integration_payload,
        integration_instructions=integration_instructions,
    )


@router.get(
    "/interventions/integrations",
    response_model=list[IntegrationInfo],
)
async def list_available_integrations() -> list[IntegrationInfo]:
    """List supported integration targets with descriptions and setup info."""
    return [
        IntegrationInfo(
            name="email",
            description="Standard email via SMTP or mailto",
            setup_url="",
            required_config=["smtp_host", "smtp_port", "from_address"],
        ),
        IntegrationInfo(
            name="hubspot",
            description="HubSpot single-send transactional email",
            setup_url="https://developers.hubspot.com/docs/api/marketing/transactional-emails",
            required_config=["api_key", "email_id"],
        ),
        IntegrationInfo(
            name="salesforce",
            description="Salesforce Marketing Cloud triggered send",
            setup_url="https://developer.salesforce.com/docs/marketing/marketing-cloud/guide/triggered-sends.html",
            required_config=["client_id", "client_secret", "send_id"],
        ),
        IntegrationInfo(
            name="marketo",
            description="Marketo trigger campaign via REST API",
            setup_url="https://developers.marketo.com/rest-api/",
            required_config=["munchkin_id", "client_id", "client_secret", "campaign_id"],
        ),
        IntegrationInfo(
            name="braze",
            description="Braze campaign trigger with personalization",
            setup_url="https://www.braze.com/docs/api/endpoints/messaging/send_messages/post_send_triggered_campaigns/",
            required_config=["api_key", "campaign_id"],
        ),
    ]


def _build_integration_payload(
    email: Any,
    rendered_content: str,
    integration: str,
    format: str,
) -> dict[str, Any]:
    """Build a structured payload for the target integration.

    These are template stubs with correct structure and populated content,
    with placeholder values for API keys and IDs.
    """
    subject = email.subject
    body = rendered_content

    if integration == "email":
        return {
            "to": "{recipient_email}",
            "from": "retention@retain.example.com",
            "subject": subject,
            "html_body": body if format == "html" else None,
            "text_body": body if format == "plaintext" else None,
            "headers": {
                "X-Campaign": "retention-outreach",
                "Reply-To": "support@retain.example.com",
            },
        }

    elif integration == "hubspot":
        return {
            "emailId": "{hubspot_email_id}",
            "message": {
                "to": "{recipient_email}",
                "subject": subject,
            },
            "customProperties": {
                "email_body": body,
                "cta_text": email.cta_text,
                "tone": email.tone,
            },
            "contactProperties": {
                "churn_risk_contacted": "true",
            },
        }

    elif integration == "salesforce":
        return {
            "definitionKey": "{salesforce_send_id}",
            "recipients": [{
                "contactKey": "{subscriber_key}",
                "to": "{recipient_email}",
                "attributes": {
                    "Subject": subject,
                    "Body": body,
                    "CTAText": email.cta_text,
                },
            }],
        }

    elif integration == "marketo":
        return {
            "input": {
                "leads": [{"id": "{lead_id}"}],
                "tokens": [
                    {"name": "{{my.Subject}}", "value": subject},
                    {"name": "{{my.Body}}", "value": body},
                    {"name": "{{my.CTAText}}", "value": email.cta_text},
                ],
            },
        }

    elif integration == "braze":
        return {
            "campaign_id": "{braze_campaign_id}",
            "recipients": [{
                "external_user_id": "{external_id}",
                "trigger_properties": {
                    "subject": subject,
                    "body": body,
                    "cta_text": email.cta_text,
                    "tone": email.tone,
                },
            }],
        }

    else:
        return {"error": f"Unsupported integration: {integration}"}
