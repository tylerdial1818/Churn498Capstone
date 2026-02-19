"""
Intervention drafting system for Retain.

Provides strategy selection, email drafting, and rendering utilities
for personalized retention campaigns.
"""

from .drafter import (
    DraftedEmail,
    InterventionResult,
    create_intervention_drafter,
    draft_intervention,
)
from .email_renderer import (
    render_as_html,
    render_as_markdown,
    render_as_plaintext,
    render_comparison,
)
from .strategies import (
    STRATEGY_REGISTRY,
    InterventionStrategy,
    select_strategy,
)

__all__ = [
    "InterventionStrategy",
    "STRATEGY_REGISTRY",
    "select_strategy",
    "DraftedEmail",
    "InterventionResult",
    "create_intervention_drafter",
    "draft_intervention",
    "render_as_markdown",
    "render_as_html",
    "render_as_plaintext",
    "render_comparison",
]
