"""
Intervention strategy definitions and selection logic.

Pure Python — no LLM calls. Strategy selection is fully deterministic:
same inputs always produce the same strategy.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("retain.agents.intervention")


@dataclass
class InterventionStrategy:
    """A retention intervention approach mapped to a churn driver."""

    id: str
    name: str
    churn_driver: str
    description: str
    email_tone: str  # empathetic, urgent, value-focused, helpful, celebratory
    typical_offer: str | None
    subject_line_templates: list[str]
    cta_options: list[str]
    priority: int  # 1-5, higher = more aggressive


# =============================================================================
# Strategy Definitions
# =============================================================================


STRATEGY_REGISTRY: dict[str, InterventionStrategy] = {
    "win_back_discount": InterventionStrategy(
        id="win_back_discount",
        name="Win-Back Discount",
        churn_driver="price_sensitivity",
        description=(
            "Offer a limited-time discount to retain price-sensitive subscribers "
            "or those with payment issues that may stem from cost concerns."
        ),
        email_tone="value-focused",
        typical_offer="20% off next 3 months",
        subject_line_templates=[
            "A special offer just for you",
            "We'd love to keep you — here's 20% off",
            "Your exclusive streaming deal inside",
        ],
        cta_options=[
            "Claim Your Discount",
            "Activate 20% Off",
            "Keep Streaming for Less",
        ],
        priority=4,
    ),
    "content_discovery": InterventionStrategy(
        id="content_discovery",
        name="Content Discovery",
        churn_driver="content_gap",
        description=(
            "Surface new and relevant content for subscribers who may feel "
            "they've exhausted the catalog or whose preferences aren't being met."
        ),
        email_tone="celebratory",
        typical_offer=None,
        subject_line_templates=[
            "New picks we think you'll love",
            "Your next binge-worthy series awaits",
            "Fresh content, curated for you",
        ],
        cta_options=[
            "Explore New Titles",
            "See What's New",
            "Start Watching",
        ],
        priority=2,
    ),
    "vip_support_rescue": InterventionStrategy(
        id="vip_support_rescue",
        name="VIP Support Rescue",
        churn_driver="support_frustration",
        description=(
            "Provide VIP-level support and a service credit to subscribers "
            "frustrated by unresolved issues or poor support experiences."
        ),
        email_tone="empathetic",
        typical_offer="1 month service credit",
        subject_line_templates=[
            "We hear you — let's make this right",
            "Your dedicated support team is ready",
            "We owe you better — here's a month on us",
        ],
        cta_options=[
            "Connect with VIP Support",
            "Claim Your Credit",
            "Get Priority Help",
        ],
        priority=5,
    ),
    "engagement_reignite": InterventionStrategy(
        id="engagement_reignite",
        name="Engagement Re-Ignite",
        churn_driver="disengagement",
        description=(
            "Re-engage subscribers with declining (but non-zero) usage through "
            "personalized recommendations and premium trial offers."
        ),
        email_tone="helpful",
        typical_offer="Free premium trial",
        subject_line_templates=[
            "We picked something special for you",
            "Your personalized watchlist is ready",
            "Come back to something great",
        ],
        cta_options=[
            "See Your Recommendations",
            "Try Premium Free",
            "Resume Watching",
        ],
        priority=3,
    ),
    "payment_recovery": InterventionStrategy(
        id="payment_recovery",
        name="Payment Recovery",
        churn_driver="payment_issues",
        description=(
            "Urgent outreach to subscribers with failed payments to update "
            "their payment method and retain access to their account."
        ),
        email_tone="urgent",
        typical_offer="Update payment — keep your watchlist and history",
        subject_line_templates=[
            "Action needed: update your payment",
            "Don't lose your watchlist and history",
            "Quick fix to keep streaming",
        ],
        cta_options=[
            "Update Payment Now",
            "Fix My Account",
            "Keep My Subscription",
        ],
        priority=5,
    ),
}


# =============================================================================
# Strategy Selection
# =============================================================================


def select_strategy(
    root_cause: str,
    account_context: dict[str, Any] | None = None,
) -> InterventionStrategy:
    """Select the best intervention strategy for a given root cause.

    Selection is deterministic: same root cause + context = same strategy.
    Business rules override when applicable.

    Args:
        root_cause: Primary churn driver category.
        account_context: Optional context dict with keys like 'plan_type',
            'tenure_days', 'subscription_status', 'engagement_level'.

    Returns:
        Selected InterventionStrategy.
    """
    context = account_context or {}
    tenure_days = context.get("tenure_days", 365)
    subscription_status = context.get("subscription_status", "active")
    plan_type = context.get("plan_type", "Regular")

    logger.info(
        f"Selecting strategy for root_cause={root_cause}, "
        f"tenure_days={tenure_days}, status={subscription_status}"
    )

    # Business rule: payment_failed status always gets payment_recovery
    if subscription_status == "payment_failed":
        logger.info("Applied rule: payment_failed → payment_recovery")
        return STRATEGY_REGISTRY["payment_recovery"]

    # Map root causes to strategies
    cause_to_strategy = {
        "payment_issues": "payment_recovery",
        "price_sensitivity": "win_back_discount",
        "disengagement": "engagement_reignite",
        "content_gap": "content_discovery",
        "support_frustration": "vip_support_rescue",
        "technical_issues": "vip_support_rescue",
    }

    strategy_id = cause_to_strategy.get(root_cause)

    if strategy_id is None:
        logger.warning(
            f"Unknown root cause '{root_cause}', defaulting to engagement_reignite"
        )
        strategy_id = "engagement_reignite"

    strategy = STRATEGY_REGISTRY[strategy_id]

    # Business rule: never offer discounts to accounts < 30 days old
    if strategy_id == "win_back_discount" and tenure_days < 30:
        logger.info(
            "Applied rule: no discounts for <30 day accounts, "
            "switching to content_discovery"
        )
        strategy = STRATEGY_REGISTRY["content_discovery"]

    # Business rule: for disengaged Regular plan users, offer premium trial
    if (
        root_cause == "disengagement"
        and plan_type == "Regular"
        and strategy_id == "engagement_reignite"
    ):
        # engagement_reignite already has premium trial as offer
        logger.info("Applied rule: Regular plan disengagement → premium trial offer")

    return strategy
