"""
Email rendering utilities for retention interventions.

Converts DraftedEmail objects into markdown, HTML, and plaintext formats.
HTML uses inline styles only for email client compatibility.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .drafter import DraftedEmail

logger = logging.getLogger("retain.agents.intervention")


def render_as_markdown(email: "DraftedEmail") -> str:
    """Render an email as formatted markdown.

    Args:
        email: DraftedEmail to render.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        f"## Variant {email.variant}: {email.subject}",
        "",
        f"*Preview: {email.preview_text}*",
        "",
        f"**Tone:** {email.tone}",
        "",
        "---",
        "",
        email.greeting,
        "",
        email.body,
        "",
        f"**[{email.cta_text}]({email.cta_url})**",
        "",
        email.closing,
        "",
        "---",
        "",
    ]

    if email.personalization_fields_used:
        lines.append(
            f"*Personalization fields: {', '.join(email.personalization_fields_used)}*"
        )

    return "\n".join(lines)


def render_as_html(email: "DraftedEmail", template: str = "default") -> str:
    """Render an email as inline-styled HTML for email clients.

    Uses only inline CSS — no <style> blocks or external stylesheets.
    Single column layout with header, body, CTA button, and footer.

    Args:
        email: DraftedEmail to render.
        template: Template name (reserved for future use).

    Returns:
        HTML string ready for email clients.
    """
    # Convert body paragraphs (split on double newlines)
    body_paragraphs = email.body.split("\n\n")
    body_html = "".join(
        f'<p style="margin: 0 0 16px 0; font-size: 16px; line-height: 1.6; '
        f'color: #333333;">{p.strip()}</p>'
        for p in body_paragraphs
        if p.strip()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{email.subject}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f4f4f4; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #f4f4f4;">
<tr>
<td align="center" style="padding: 40px 20px;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden;">

<!-- Header -->
<tr>
<td style="background-color: #1a1a2e; padding: 30px 40px; text-align: center;">
<h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 600;">Retain</h1>
</td>
</tr>

<!-- Body -->
<tr>
<td style="padding: 40px;">
<p style="margin: 0 0 20px 0; font-size: 18px; color: #1a1a2e;">{email.greeting}</p>

{body_html}

<!-- CTA Button -->
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin: 30px 0;">
<tr>
<td align="center">
<a href="{email.cta_url}" style="display: inline-block; padding: 14px 32px; background-color: #e94560; color: #ffffff; text-decoration: none; font-size: 16px; font-weight: 600; border-radius: 6px;">{email.cta_text}</a>
</td>
</tr>
</table>

<p style="margin: 0; font-size: 16px; line-height: 1.6; color: #333333;">{email.closing}</p>
</td>
</tr>

<!-- Footer -->
<tr>
<td style="background-color: #f8f8f8; padding: 20px 40px; text-align: center;">
<p style="margin: 0; font-size: 12px; color: #999999;">You're receiving this because you're a Retain subscriber.</p>
</td>
</tr>

</table>
</td>
</tr>
</table>
</body>
</html>"""

    return html


def render_as_plaintext(email: "DraftedEmail") -> str:
    """Render an email as zero-HTML plaintext.

    Args:
        email: DraftedEmail to render.

    Returns:
        Plain text string with no HTML tags.
    """
    lines = [
        f"Subject: {email.subject}",
        "",
        email.greeting,
        "",
        email.body,
        "",
        f"{email.cta_text}: {email.cta_url}",
        "",
        email.closing,
    ]

    return "\n".join(lines)


def render_comparison(emails: list["DraftedEmail"]) -> str:
    """Render a side-by-side A/B comparison as a markdown table.

    Args:
        emails: List of DraftedEmail variants (typically 2).

    Returns:
        Markdown-formatted comparison.
    """
    if not emails:
        return "*No emails to compare.*"

    if len(emails) == 1:
        return render_as_markdown(emails[0])

    # Build comparison table
    fields = [
        ("Subject", "subject"),
        ("Preview", "preview_text"),
        ("Tone", "tone"),
        ("Greeting", "greeting"),
        ("CTA", "cta_text"),
    ]

    lines = ["# Email A/B Comparison", ""]

    # Header row
    headers = "| Field | " + " | ".join(f"Variant {e.variant}" for e in emails) + " |"
    separator = "| --- | " + " | ".join("---" for _ in emails) + " |"
    lines.extend([headers, separator])

    for label, attr in fields:
        values = " | ".join(
            str(getattr(e, attr, ""))[:60] for e in emails
        )
        lines.append(f"| {label} | {values} |")

    lines.append("")

    # Full body for each variant
    for email in emails:
        lines.extend([
            f"## Variant {email.variant} — Full Body",
            "",
            email.body,
            "",
            f"**[{email.cta_text}]({email.cta_url})**",
            "",
            email.closing,
            "",
            "---",
            "",
        ])

    return "\n".join(lines)
