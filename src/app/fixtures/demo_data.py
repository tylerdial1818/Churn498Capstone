"""
Deterministic fixture data for demo mode.

Provides realistic data matching actual distributions so the app works
without a live database or LLM connection. Every service method has a
matching fixture function.
"""

import uuid
from datetime import datetime

# =============================================================================
# Dashboard Fixtures
# =============================================================================

_MONTHLY_TRENDS = [
    {"month": "2024-01", "signups": 2810, "cancellations": 385, "churn_rate": 0.031},
    {"month": "2024-02", "signups": 2640, "cancellations": 410, "churn_rate": 0.033},
    {"month": "2024-03", "signups": 3120, "cancellations": 392, "churn_rate": 0.030},
    {"month": "2024-04", "signups": 2950, "cancellations": 428, "churn_rate": 0.033},
    {"month": "2024-05", "signups": 3210, "cancellations": 445, "churn_rate": 0.034},
    {"month": "2024-06", "signups": 2780, "cancellations": 460, "churn_rate": 0.036},
    {"month": "2024-07", "signups": 3050, "cancellations": 475, "churn_rate": 0.037},
    {"month": "2024-08", "signups": 2890, "cancellations": 510, "churn_rate": 0.040},
    {"month": "2024-09", "signups": 3340, "cancellations": 498, "churn_rate": 0.038},
    {"month": "2024-10", "signups": 3180, "cancellations": 520, "churn_rate": 0.040},
    {"month": "2024-11", "signups": 2760, "cancellations": 545, "churn_rate": 0.042},
    {"month": "2024-12", "signups": 2950, "cancellations": 580, "churn_rate": 0.044},
    {"month": "2025-01", "signups": 3410, "cancellations": 560, "churn_rate": 0.042},
    {"month": "2025-02", "signups": 3060, "cancellations": 590, "churn_rate": 0.044},
    {"month": "2025-03", "signups": 3280, "cancellations": 610, "churn_rate": 0.045},
    {"month": "2025-04", "signups": 2870, "cancellations": 625, "churn_rate": 0.046},
    {"month": "2025-05", "signups": 3150, "cancellations": 640, "churn_rate": 0.047},
    {"month": "2025-06", "signups": 2980, "cancellations": 655, "churn_rate": 0.048},
    {"month": "2025-07", "signups": 3090, "cancellations": 660, "churn_rate": 0.048},
    {"month": "2025-08", "signups": 2840, "cancellations": 670, "churn_rate": 0.049},
    {"month": "2025-09", "signups": 3220, "cancellations": 685, "churn_rate": 0.049},
    {"month": "2025-10", "signups": 2910, "cancellations": 700, "churn_rate": 0.050},
    {"month": "2025-11", "signups": 2780, "cancellations": 715, "churn_rate": 0.051},
    {"month": "2025-12", "signups": 3050, "cancellations": 690, "churn_rate": 0.049},
]

_AT_RISK_ACCOUNTS = [
    {
        "account_id": "ACC_00004217",
        "email": "maria.santos@email.com",
        "churn_probability": 0.94,
        "risk_tier": "high",
        "plan_type": "Regular",
        "tenure_days": 245,
        "last_payment_days": 38,
        "last_stream_days": 22,
        "open_tickets": 3,
        "top_drivers": ["payment_failure_rate", "days_since_last_stream", "support_ticket_count"],
    },
    {
        "account_id": "ACC_00012843",
        "email": "james.wilson@email.com",
        "churn_probability": 0.91,
        "risk_tier": "high",
        "plan_type": "Premium",
        "tenure_days": 89,
        "last_payment_days": 45,
        "last_stream_days": 31,
        "open_tickets": 2,
        "top_drivers": ["payment_failure_rate", "watch_hours_decline_pct", "tenure_days"],
    },
    {
        "account_id": "ACC_00008391",
        "email": "sarah.chen@email.com",
        "churn_probability": 0.88,
        "risk_tier": "high",
        "plan_type": "Premium-Multi-Screen",
        "tenure_days": 412,
        "last_payment_days": 12,
        "last_stream_days": 45,
        "open_tickets": 0,
        "top_drivers": ["days_since_last_stream", "watch_hours_decline_pct", "content_diversity_drop"],
    },
    {
        "account_id": "ACC_00019502",
        "email": "alex.kumar@email.com",
        "churn_probability": 0.86,
        "risk_tier": "high",
        "plan_type": "Regular",
        "tenure_days": 156,
        "last_payment_days": 8,
        "last_stream_days": 28,
        "open_tickets": 4,
        "top_drivers": ["support_ticket_count", "avg_resolution_hours", "support_escalation_rate"],
    },
    {
        "account_id": "ACC_00003674",
        "email": "emma.johnson@email.com",
        "churn_probability": 0.84,
        "risk_tier": "high",
        "plan_type": "Regular",
        "tenure_days": 67,
        "last_payment_days": 5,
        "last_stream_days": 14,
        "open_tickets": 1,
        "top_drivers": ["watch_hours_decline_pct", "session_frequency_drop", "tenure_days"],
    },
    {
        "account_id": "ACC_00027156",
        "email": "david.martinez@email.com",
        "churn_probability": 0.82,
        "risk_tier": "high",
        "plan_type": "Premium",
        "tenure_days": 534,
        "last_payment_days": 30,
        "last_stream_days": 18,
        "open_tickets": 1,
        "top_drivers": ["payment_failure_rate", "plan_downgrade_flag", "watch_hours_decline_pct"],
    },
    {
        "account_id": "ACC_00015829",
        "email": "lisa.park@email.com",
        "churn_probability": 0.79,
        "risk_tier": "high",
        "plan_type": "Regular",
        "tenure_days": 198,
        "last_payment_days": 15,
        "last_stream_days": 35,
        "open_tickets": 0,
        "top_drivers": ["days_since_last_stream", "content_diversity_drop", "watchlist_abandonment"],
    },
    {
        "account_id": "ACC_00031247",
        "email": "tom.nguyen@email.com",
        "churn_probability": 0.76,
        "risk_tier": "high",
        "plan_type": "Premium-Multi-Screen",
        "tenure_days": 310,
        "last_payment_days": 10,
        "last_stream_days": 8,
        "open_tickets": 2,
        "top_drivers": ["support_ticket_count", "payment_failure_rate", "plan_downgrade_flag"],
    },
    {
        "account_id": "ACC_00009812",
        "email": "rachel.brown@email.com",
        "churn_probability": 0.73,
        "risk_tier": "high",
        "plan_type": "Regular",
        "tenure_days": 121,
        "last_payment_days": 7,
        "last_stream_days": 19,
        "open_tickets": 0,
        "top_drivers": ["watch_hours_decline_pct", "session_frequency_drop", "device_count_drop"],
    },
    {
        "account_id": "ACC_00022485",
        "email": "michael.lee@email.com",
        "churn_probability": 0.71,
        "risk_tier": "high",
        "plan_type": "Premium",
        "tenure_days": 445,
        "last_payment_days": 5,
        "last_stream_days": 12,
        "open_tickets": 1,
        "top_drivers": ["content_diversity_drop", "days_since_last_stream", "watch_hours_decline_pct"],
    },
    # Medium risk
    {
        "account_id": "ACC_00041283",
        "email": "jennifer.davis@email.com",
        "churn_probability": 0.62,
        "risk_tier": "medium",
        "plan_type": "Regular",
        "tenure_days": 287,
        "last_payment_days": 3,
        "last_stream_days": 10,
        "open_tickets": 1,
        "top_drivers": ["watch_hours_decline_pct", "session_frequency_drop", "content_diversity_drop"],
    },
    {
        "account_id": "ACC_00038917",
        "email": "robert.taylor@email.com",
        "churn_probability": 0.58,
        "risk_tier": "medium",
        "plan_type": "Premium",
        "tenure_days": 178,
        "last_payment_days": 8,
        "last_stream_days": 6,
        "open_tickets": 0,
        "top_drivers": ["tenure_days", "watch_hours_decline_pct", "payment_method_change"],
    },
    {
        "account_id": "ACC_00044562",
        "email": "amy.white@email.com",
        "churn_probability": 0.55,
        "risk_tier": "medium",
        "plan_type": "Regular",
        "tenure_days": 365,
        "last_payment_days": 2,
        "last_stream_days": 15,
        "open_tickets": 2,
        "top_drivers": ["support_ticket_count", "days_since_last_stream", "watch_hours_decline_pct"],
    },
    {
        "account_id": "ACC_00050231",
        "email": "chris.garcia@email.com",
        "churn_probability": 0.51,
        "risk_tier": "medium",
        "plan_type": "Premium-Multi-Screen",
        "tenure_days": 520,
        "last_payment_days": 5,
        "last_stream_days": 9,
        "open_tickets": 0,
        "top_drivers": ["content_diversity_drop", "session_frequency_drop", "device_count_drop"],
    },
    {
        "account_id": "ACC_00046789",
        "email": "nicole.miller@email.com",
        "churn_probability": 0.47,
        "risk_tier": "medium",
        "plan_type": "Regular",
        "tenure_days": 142,
        "last_payment_days": 4,
        "last_stream_days": 7,
        "open_tickets": 1,
        "top_drivers": ["watch_hours_decline_pct", "tenure_days", "payment_method_change"],
    },
]

_SHAP_GLOBAL = [
    {"feature": "payment_failure_rate", "importance": 0.142, "direction": "positive"},
    {"feature": "days_since_last_stream", "importance": 0.128, "direction": "positive"},
    {"feature": "watch_hours_decline_pct", "importance": 0.115, "direction": "positive"},
    {"feature": "support_ticket_count", "importance": 0.098, "direction": "positive"},
    {"feature": "tenure_days", "importance": 0.087, "direction": "negative"},
    {"feature": "session_frequency_drop", "importance": 0.076, "direction": "positive"},
    {"feature": "content_diversity_drop", "importance": 0.065, "direction": "positive"},
    {"feature": "avg_resolution_hours", "importance": 0.054, "direction": "positive"},
    {"feature": "plan_downgrade_flag", "importance": 0.048, "direction": "positive"},
    {"feature": "device_count_drop", "importance": 0.039, "direction": "positive"},
]

_INTERVENTIONS: list[dict] = [
    {
        "id": "d3f8a1b2-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
        "account_id": "ACC_00004217",
        "strategy": "payment_recovery",
        "status": "pending",
        "subject": "Let's get your account back on track",
        "body_html": (
            "<html><body style='font-family:DM Sans,sans-serif;'>"
            "<p>Hi Maria,</p>"
            "<p>We noticed your recent payment didn't go through. We'd hate for you "
            "to lose access to your favorite shows — especially with the new season "
            "of <em>Midnight Horizon</em> dropping next week.</p>"
            "<p>Updating your payment method takes less than a minute:</p>"
            "<p><a href='https://retain.example.com/billing' "
            "style='background:#6366f1;color:white;padding:12px 24px;"
            "border-radius:8px;text-decoration:none;'>Update Payment →</a></p>"
            "<p>If you're experiencing any issues, our support team is standing by "
            "to help.</p>"
            "<p>Best,<br/>The Retain Team</p>"
            "</body></html>"
        ),
        "body_plaintext": (
            "Hi Maria,\n\n"
            "We noticed your recent payment didn't go through. We'd hate for you "
            "to lose access to your favorite shows — especially with the new season "
            "of Midnight Horizon dropping next week.\n\n"
            "Update your payment method here: https://retain.example.com/billing\n\n"
            "If you're experiencing any issues, our support team is standing by.\n\n"
            "Best,\nThe Retain Team"
        ),
        "agent_rationale": (
            "Maria's account shows 3 consecutive failed payments over the last 38 days "
            "with an active viewing history prior to the payment issues. The account has "
            "3 open support tickets, suggesting she may be aware of the issue. A direct, "
            "empathetic payment recovery email is the highest-priority intervention."
        ),
        "created_at": "2025-12-01T10:30:00",
        "updated_at": "2025-12-01T10:30:00",
    },
    {
        "id": "e4a9b2c3-5d6f-7a8b-9c0d-1e2f3a4b5c6d",
        "account_id": "ACC_00008391",
        "strategy": "engagement_reignite",
        "status": "pending",
        "subject": "We've been saving something special for you",
        "body_html": (
            "<html><body style='font-family:DM Sans,sans-serif;'>"
            "<p>Hey Sarah,</p>"
            "<p>It's been a while since your last stream, and we've been adding "
            "incredible new content in the genres you love most — thriller and "
            "sci-fi.</p>"
            "<p>Here are 3 picks we think you'll love:</p>"
            "<ul><li><strong>Quantum Drift</strong> — A mind-bending sci-fi thriller</li>"
            "<li><strong>The Last Signal</strong> — Edge-of-your-seat suspense</li>"
            "<li><strong>Neon Requiem</strong> — Cyberpunk noir at its finest</li></ul>"
            "<p><a href='https://retain.example.com/browse?for=ACC_00008391' "
            "style='background:#6366f1;color:white;padding:12px 24px;"
            "border-radius:8px;text-decoration:none;'>Start Watching →</a></p>"
            "<p>Happy streaming,<br/>The Retain Team</p>"
            "</body></html>"
        ),
        "body_plaintext": (
            "Hey Sarah,\n\n"
            "It's been a while since your last stream, and we've been adding "
            "incredible new content in the genres you love most — thriller and sci-fi.\n\n"
            "Here are 3 picks we think you'll love:\n"
            "- Quantum Drift — A mind-bending sci-fi thriller\n"
            "- The Last Signal — Edge-of-your-seat suspense\n"
            "- Neon Requiem — Cyberpunk noir at its finest\n\n"
            "Start watching: https://retain.example.com/browse?for=ACC_00008391\n\n"
            "Happy streaming,\nThe Retain Team"
        ),
        "agent_rationale": (
            "Sarah is a long-tenured Premium-Multi-Screen subscriber (412 days) "
            "whose streaming activity dropped off 45 days ago despite no payment or "
            "support issues. Content discovery is the ideal angle — she likely hasn't "
            "seen new additions matching her preferences."
        ),
        "created_at": "2025-12-01T10:35:00",
        "updated_at": "2025-12-01T10:35:00",
    },
    {
        "id": "f5b0c3d4-6e7a-8b9c-0d1e-2f3a4b5c6d7e",
        "account_id": "ACC_00019502",
        "strategy": "vip_support_rescue",
        "status": "approved",
        "subject": "We hear you — and we're making it right",
        "body_html": (
            "<html><body style='font-family:DM Sans,sans-serif;'>"
            "<p>Hi Alex,</p>"
            "<p>We know your recent experience with our support team hasn't been "
            "up to standard, and we sincerely apologize. You deserve better.</p>"
            "<p>We've assigned a dedicated support specialist to your account who "
            "will personally resolve your open tickets within the next 24 hours.</p>"
            "<p><a href='https://retain.example.com/support/vip' "
            "style='background:#6366f1;color:white;padding:12px 24px;"
            "border-radius:8px;text-decoration:none;'>Chat with Your Specialist →</a></p>"
            "<p>As a gesture of goodwill, we've also added a free month to your subscription.</p>"
            "<p>With appreciation,<br/>The Retain Team</p>"
            "</body></html>"
        ),
        "body_plaintext": (
            "Hi Alex,\n\n"
            "We know your recent experience with our support team hasn't been up to "
            "standard, and we sincerely apologize. You deserve better.\n\n"
            "We've assigned a dedicated support specialist to your account who will "
            "personally resolve your open tickets within the next 24 hours.\n\n"
            "Chat with your specialist: https://retain.example.com/support/vip\n\n"
            "As a gesture of goodwill, we've also added a free month to your subscription.\n\n"
            "With appreciation,\nThe Retain Team"
        ),
        "agent_rationale": (
            "Alex has 4 open support tickets with an average resolution time well above "
            "normal. The escalation rate is high, suggesting frustration. VIP support "
            "rescue with a personal touch and compensation is critical to retain this customer."
        ),
        "created_at": "2025-12-01T09:15:00",
        "updated_at": "2025-12-01T11:00:00",
    },
    {
        "id": "a6c1d4e5-7f8a-9b0c-1d2e-3f4a5b6c7d8e",
        "account_id": "ACC_00012843",
        "strategy": "payment_recovery",
        "status": "sent",
        "subject": "Quick fix needed for your subscription",
        "body_html": (
            "<html><body style='font-family:DM Sans,sans-serif;'>"
            "<p>Hi James,</p>"
            "<p>Your Premium subscription payment was declined on November 25th. "
            "To avoid any interruption to your service, please update your payment "
            "details.</p>"
            "<p><a href='https://retain.example.com/billing' "
            "style='background:#6366f1;color:white;padding:12px 24px;"
            "border-radius:8px;text-decoration:none;'>Fix Payment →</a></p>"
            "<p>Questions? Reply to this email — we're here to help.</p>"
            "<p>Cheers,<br/>The Retain Team</p>"
            "</body></html>"
        ),
        "body_plaintext": (
            "Hi James,\n\nYour Premium subscription payment was declined on November "
            "25th. To avoid any interruption, please update your payment details.\n\n"
            "Fix payment: https://retain.example.com/billing\n\n"
            "Questions? Reply to this email.\n\nCheers,\nThe Retain Team"
        ),
        "agent_rationale": (
            "James is a relatively new Premium subscriber (89 days) with a 45-day "
            "payment gap. The payment method likely expired or was declined. A concise, "
            "urgent payment recovery message is appropriate."
        ),
        "created_at": "2025-11-28T14:20:00",
        "updated_at": "2025-11-29T09:00:00",
    },
]


# =============================================================================
# Fixture Functions
# =============================================================================


def get_fixture_kpis() -> dict:
    """Return dashboard KPI fixture data."""
    return {
        "total_accounts": 60000,
        "active_subscribers": 49684,
        "churned_accounts": 10316,
        "churn_rate_30d": 0.049,
        "high_risk_count": 2420,
        "at_risk_mrr": 33_880.00,
        "cac": 45.00,
        "retention_cost_per_save": 12.50,
    }


def get_fixture_trends() -> list[dict]:
    """Return monthly trend fixture data."""
    trends = []
    for t in _MONTHLY_TRENDS:
        trends.append({
            **t,
            "net_growth": t["signups"] - t["cancellations"],
        })
    return trends


def get_fixture_risk_distribution() -> dict:
    """Return risk distribution fixture data."""
    return {"low": 53760, "medium": 3820, "high": 2420}


def get_fixture_active_inactive() -> dict:
    """Return active vs inactive distribution."""
    return {
        "active": 49684,
        "inactive": 10316,
        "active_pct": 82.8,
        "inactive_pct": 17.2,
    }


def get_fixture_executive_summary() -> dict:
    """Return AI executive summary fixture."""
    return {
        "agent_name": "Analysis Agent",
        "content": (
            "### Subscriber Health Summary\n\n"
            "The platform currently serves **49,684 active subscribers** across "
            "60,000 total accounts. The 30-day predicted churn rate stands at "
            "**4.9%**, up from 4.4% last quarter — a trend worth monitoring.\n\n"
            "**Key findings:**\n\n"
            "- **2,420 accounts** are flagged as high-risk, representing "
            "**$33,880 in monthly recurring revenue** at risk\n"
            "- Payment failures are the #1 churn driver this month, affecting "
            "38% of high-risk accounts\n"
            "- Engagement drop-off is accelerating among Regular plan subscribers, "
            "particularly those in the 3-6 month tenure bracket\n"
            "- Premium-Multi-Screen subscribers show the strongest retention "
            "(97.2% rate), while Regular plan churn has risen to 5.8%\n\n"
            "**Recommended focus areas:**\n\n"
            "1. Prioritize payment recovery outreach for the 920 accounts with "
            "failed payments — estimated save rate of 62% if contacted within 48 hours\n"
            "2. Deploy re-engagement campaigns for the 680 disengaged accounts "
            "before they reach the 30-day inactivity cliff\n"
            "3. Review support ticket backlog — 4 accounts have 3+ unresolved "
            "tickets and are at critical risk\n\n"
            "[View At-Risk Accounts →](/at-risk) · "
            "[See Full Analytics →](/analytics)"
        ),
        "generated_at": "2025-12-01T10:00:00",
        "status": "complete",
    }


def get_fixture_at_risk_accounts(
    risk_tier: str | None = None,
    plan_type: str | None = None,
    sort_by: str = "churn_probability",
    page: int = 1,
    per_page: int = 20,
) -> dict:
    """Return paginated at-risk accounts fixture."""
    accounts = list(_AT_RISK_ACCOUNTS)

    if risk_tier and risk_tier != "all":
        accounts = [a for a in accounts if a["risk_tier"] == risk_tier]
    if plan_type and plan_type != "all":
        accounts = [a for a in accounts if a["plan_type"] == plan_type]

    reverse = sort_by in ("churn_probability",)
    accounts.sort(key=lambda a: a.get(sort_by, 0), reverse=reverse)

    total = len(accounts)
    start = (page - 1) * per_page
    end = start + per_page
    items = accounts[start:end]

    return {"items": items, "total": total, "page": page, "per_page": per_page}


def get_fixture_account_detail(account_id: str) -> dict | None:
    """Return detailed account fixture data."""
    account = next(
        (a for a in _AT_RISK_ACCOUNTS if a["account_id"] == account_id),
        None,
    )
    if account is None:
        return None

    shap_map = {
        d: round(0.15 - i * 0.02, 4)
        for i, d in enumerate(account["top_drivers"])
    }

    return {
        **account,
        "signup_date": "2025-03-15",
        "country": "US",
        "age": 34,
        "gender": "F",
        "subscription_status": "active",
        "shap_values": shap_map,
        "payment_history": [
            {
                "payment_id": "PAY_001",
                "payment_date": "2025-11-01",
                "amount": 14.99,
                "currency": "USD",
                "payment_method": "credit_card",
                "status": "success",
                "failure_reason": None,
            },
            {
                "payment_id": "PAY_002",
                "payment_date": "2025-10-01",
                "amount": 14.99,
                "currency": "USD",
                "payment_method": "credit_card",
                "status": "failed",
                "failure_reason": "insufficient_funds",
            },
        ],
        "ticket_history": [
            {
                "ticket_id": "TKT_001",
                "created_at": "2025-11-20",
                "category": "billing",
                "priority": "high",
                "resolved_at": None,
            },
        ],
        "agent_narrative": (
            f"Account {account_id} shows elevated churn risk driven primarily by "
            f"{', '.join(account['top_drivers'][:2])}. "
            "The combination of declining engagement and unresolved support issues "
            "suggests this customer is actively considering cancellation. "
            "Immediate intervention is recommended."
        ),
    }


def get_fixture_account_shap(account_id: str) -> dict[str, float] | None:
    """Return SHAP values fixture for an account."""
    account = next(
        (a for a in _AT_RISK_ACCOUNTS if a["account_id"] == account_id),
        None,
    )
    if account is None:
        return None

    return {
        d: round(0.15 - i * 0.02, 4)
        for i, d in enumerate(account["top_drivers"])
    }


def get_fixture_analytics_overview() -> dict:
    """Return analytics overview bundle."""
    return {
        "kpis": get_fixture_kpis(),
        "risk_distribution": get_fixture_risk_distribution(),
        "top_shap_features": _SHAP_GLOBAL,
        "plan_breakdown": [
            {"plan_type": "Regular", "total": 38400, "churned": 7104, "churn_rate": 0.058},
            {"plan_type": "Premium", "total": 15600, "churned": 2652, "churn_rate": 0.040},
            {"plan_type": "Premium-Multi-Screen", "total": 6000, "churned": 560, "churn_rate": 0.028},
        ],
    }


def get_fixture_churn_trends(months: int = 12) -> list[dict]:
    """Return churn trend time series."""
    trends = get_fixture_trends()
    return trends[-months:]


def get_fixture_segments() -> dict:
    """Return segment breakdowns."""
    return {
        "by_plan": [
            {"plan_type": "Regular", "total": 38400, "churned": 7104, "churn_rate": 0.058},
            {"plan_type": "Premium", "total": 15600, "churned": 2652, "churn_rate": 0.040},
            {"plan_type": "Premium-Multi-Screen", "total": 6000, "churned": 560, "churn_rate": 0.028},
        ],
        "by_tenure": [
            {"bucket": "0-3 months", "total": 8400, "churned": 1680, "churn_rate": 0.080},
            {"bucket": "3-6 months", "total": 12000, "churned": 1920, "churn_rate": 0.064},
            {"bucket": "6-12 months", "total": 18000, "churned": 1440, "churn_rate": 0.040},
            {"bucket": "12+ months", "total": 21600, "churned": 1296, "churn_rate": 0.030},
        ],
        "by_payment_method": [
            {"method": "credit_card", "total": 36000, "churned": 5400, "churn_rate": 0.050},
            {"method": "debit_card", "total": 12000, "churned": 2160, "churn_rate": 0.060},
            {"method": "bank_transfer", "total": 6000, "churned": 780, "churn_rate": 0.043},
            {"method": "digital_wallet", "total": 6000, "churned": 600, "churn_rate": 0.033},
        ],
    }


def get_fixture_model_performance() -> dict:
    """Return model performance metrics."""
    return {
        "auc_roc": 0.891,
        "precision": 0.823,
        "recall": 0.756,
        "f1_score": 0.788,
        "accuracy": 0.862,
        "calibration_error": 0.034,
        "prediction_distribution": [
            {"bin": "0.0-0.1", "count": 28800},
            {"bin": "0.1-0.2", "count": 12600},
            {"bin": "0.2-0.3", "count": 7200},
            {"bin": "0.3-0.4", "count": 4380},
            {"bin": "0.4-0.5", "count": 2160},
            {"bin": "0.5-0.6", "count": 1680},
            {"bin": "0.6-0.7", "count": 960},
            {"bin": "0.7-0.8", "count": 1080},
            {"bin": "0.8-0.9", "count": 840},
            {"bin": "0.9-1.0", "count": 300},
        ],
    }


def get_fixture_drift_status() -> dict:
    """Return drift monitoring status."""
    return {
        "overall_status": "healthy",
        "features": [
            {"feature": "payment_failure_rate", "psi": 0.032, "status": "stable"},
            {"feature": "days_since_last_stream", "psi": 0.045, "status": "stable"},
            {"feature": "watch_hours_decline_pct", "psi": 0.028, "status": "stable"},
            {"feature": "support_ticket_count", "psi": 0.091, "status": "stable"},
            {"feature": "tenure_days", "psi": 0.015, "status": "stable"},
            {"feature": "session_frequency_drop", "psi": 0.112, "status": "warning"},
        ],
        "checked_at": "2025-12-01T08:00:00",
    }


def get_fixture_shap_global() -> list[dict]:
    """Return global SHAP feature importance."""
    return list(_SHAP_GLOBAL)


def get_fixture_prescriptions() -> list[dict]:
    """Return prescription groups."""
    high_risk = [a for a in _AT_RISK_ACCOUNTS if a["risk_tier"] == "high"]

    payment_accts = [a for a in high_risk if "payment_failure_rate" in a["top_drivers"]]
    engagement_accts = [a for a in high_risk if "days_since_last_stream" in a["top_drivers"] or "watch_hours_decline_pct" in a["top_drivers"]]
    support_accts = [a for a in high_risk if "support_ticket_count" in a["top_drivers"]]
    content_accts = [a for a in high_risk if "content_diversity_drop" in a["top_drivers"]]

    return [
        {
            "strategy": "payment_recovery",
            "display_name": "Payment Recovery",
            "account_count": max(len(payment_accts), 920),
            "estimated_mrr": 12_880.00,
            "accounts": payment_accts,
        },
        {
            "strategy": "engagement_reignite",
            "display_name": "Re-engagement Campaign",
            "account_count": max(len(engagement_accts), 680),
            "estimated_mrr": 9_520.00,
            "accounts": engagement_accts,
        },
        {
            "strategy": "vip_support_rescue",
            "display_name": "Support Escalation",
            "account_count": max(len(support_accts), 340),
            "estimated_mrr": 4_760.00,
            "accounts": support_accts,
        },
        {
            "strategy": "content_discovery",
            "display_name": "Content Recommendation",
            "account_count": max(len(content_accts), 480),
            "estimated_mrr": 6_720.00,
            "accounts": content_accts,
        },
    ]


def get_fixture_prescription_by_strategy(strategy: str) -> dict | None:
    """Return a single prescription group by strategy."""
    groups = get_fixture_prescriptions()
    return next((g for g in groups if g["strategy"] == strategy), None)


def get_fixture_interventions(status: str | None = None) -> list[dict]:
    """Return intervention drafts, optionally filtered by status."""
    items = list(_INTERVENTIONS)
    if status:
        items = [i for i in items if i["status"] == status]
    return items


def get_fixture_intervention(intervention_id: str) -> dict | None:
    """Return a single intervention by ID."""
    return next(
        (i for i in _INTERVENTIONS if i["id"] == intervention_id),
        None,
    )


def create_fixture_intervention(account_id: str, strategy: str) -> dict:
    """Create a new fixture intervention draft."""
    new_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    account = next(
        (a for a in _AT_RISK_ACCOUNTS if a["account_id"] == account_id),
        None,
    )
    email = account["email"] if account else "customer@email.com"
    name = email.split("@")[0].replace(".", " ").title()

    return {
        "id": new_id,
        "account_id": account_id,
        "strategy": strategy,
        "status": "pending",
        "subject": f"We want to keep you — here's what we can do",
        "body_html": (
            f"<html><body style='font-family:DM Sans,sans-serif;'>"
            f"<p>Hi {name},</p>"
            f"<p>We've noticed some changes in your account and want to make sure "
            f"you're getting the most out of your subscription.</p>"
            f"<p><a href='https://retain.example.com/offer/{account_id}' "
            f"style='background:#6366f1;color:white;padding:12px 24px;"
            f"border-radius:8px;text-decoration:none;'>See Your Options →</a></p>"
            f"<p>Best,<br/>The Retain Team</p>"
            f"</body></html>"
        ),
        "body_plaintext": (
            f"Hi {name},\n\n"
            f"We've noticed some changes in your account and want to make sure "
            f"you're getting the most out of your subscription.\n\n"
            f"See your options: https://retain.example.com/offer/{account_id}\n\n"
            f"Best,\nThe Retain Team"
        ),
        "agent_rationale": (
            f"Account {account_id} was flagged for {strategy} intervention based on "
            f"predictive churn analysis. The selected strategy addresses the primary "
            f"risk factors identified in the account's feature profile."
        ),
        "created_at": now,
        "updated_at": now,
    }


def get_fixture_early_warning() -> dict:
    """Return early warning agent insight for at-risk page."""
    return {
        "agent_name": "Early Warning Agent",
        "content": (
            "### Newly Flagged Accounts This Week\n\n"
            "**12 new accounts** crossed the high-risk threshold in the past 7 days, "
            "grouped by root cause:\n\n"
            "**Payment Failures (5 accounts)**\n"
            "- 3 Regular plan subscribers with expired credit cards\n"
            "- 2 Premium subscribers with insufficient funds\n\n"
            "**Engagement Drop-off (4 accounts)**\n"
            "- Average days since last stream: 28 days\n"
            "- All were previously watching 10+ hours/week\n\n"
            "**Support Escalation (3 accounts)**\n"
            "- Average 3.2 open tickets per account\n"
            "- Mean resolution time: 72 hours (vs. 24-hour SLA)\n\n"
            "**Priority action:** The 5 payment failure accounts have the highest "
            "save probability (78%) if contacted within 48 hours."
        ),
        "generated_at": "2025-12-01T08:30:00",
        "status": "complete",
    }


def get_fixture_prescription_summary() -> dict:
    """Return prescription agent summary."""
    return {
        "agent_name": "Prescription Agent",
        "content": (
            "### Intervention Recommendations\n\n"
            "Based on the current risk analysis, **2,420 accounts** require "
            "intervention across 4 strategies:\n\n"
            "| Strategy | Accounts | Est. MRR at Risk | Priority |\n"
            "|----------|----------|------------------|----------|\n"
            "| Payment Recovery | 920 | $12,880 | Critical |\n"
            "| Re-engagement | 680 | $9,520 | High |\n"
            "| Content Discovery | 480 | $6,720 | Medium |\n"
            "| Support Escalation | 340 | $4,760 | High |\n\n"
            "**Estimated overall impact:** If all interventions are executed within "
            "7 days, the projected retention rate is **68%**, preserving approximately "
            "**$23,000 in monthly revenue**.\n\n"
            "Payment recovery should be prioritized — it has the highest save rate "
            "(62%) and the shortest effective window (48 hours)."
        ),
        "generated_at": "2025-12-01T10:15:00",
        "status": "complete",
    }
