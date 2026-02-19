"""
Tests for the Intervention Drafter (Phase 3).

Run with: pytest tests/test_intervention_drafter.py -v -k "TestInterventionStrategies or TestEmailRenderer"
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.intervention.drafter import DraftedEmail, InterventionResult
from src.agents.intervention.email_renderer import (
    render_as_html,
    render_as_markdown,
    render_as_plaintext,
    render_comparison,
)
from src.agents.intervention.strategies import (
    STRATEGY_REGISTRY,
    InterventionStrategy,
    select_strategy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_email_a() -> DraftedEmail:
    """Sample DraftedEmail variant A."""
    return DraftedEmail(
        variant="A",
        subject="New shows picked for you",
        preview_text="Your next binge is waiting",
        greeting="Hi there,",
        body=(
            "We noticed some amazing new titles just landed in your "
            "favorite genres. From critically acclaimed dramas to "
            "binge-worthy thrillers, there's something great waiting.\n\n"
            "Based on what you've watched before, we think you'll "
            "especially enjoy our latest originals."
        ),
        cta_text="Explore New Titles",
        cta_url="https://retain.example.com/browse",
        closing="Happy streaming,\nThe Retain Team",
        tone="celebratory",
        personalization_fields_used=["favorite_genre"],
    )


@pytest.fixture
def sample_email_b() -> DraftedEmail:
    """Sample DraftedEmail variant B."""
    return DraftedEmail(
        variant="B",
        subject="Your watchlist just got better",
        preview_text="Fresh picks based on your taste",
        greeting="Hey,",
        body=(
            "Great news — we've added a bunch of new content that "
            "matches your viewing history. Whether you're in the mood "
            "for something light or a deep dive, we've got you covered.\n\n"
            "Check out our editor's picks and find your next favorite."
        ),
        cta_text="See What's New",
        cta_url="https://retain.example.com/new",
        closing="Enjoy,\nTeam Retain",
        tone="helpful",
        personalization_fields_used=["last_watched"],
    )


# =============================================================================
# TestInterventionStrategies
# =============================================================================


class TestInterventionStrategies:
    """Tests for strategy definitions and selection."""

    def test_strategy_registry_has_all_strategies(self):
        """Registry contains all 5 defined strategies."""
        expected = {
            "win_back_discount",
            "content_discovery",
            "vip_support_rescue",
            "engagement_reignite",
            "payment_recovery",
        }
        assert set(STRATEGY_REGISTRY.keys()) == expected

    def test_all_strategies_are_valid(self):
        """Every strategy has all required fields populated."""
        for sid, strategy in STRATEGY_REGISTRY.items():
            assert strategy.id == sid
            assert strategy.name
            assert strategy.churn_driver
            assert strategy.description
            assert strategy.email_tone in {
                "empathetic", "urgent", "value-focused", "helpful", "celebratory"
            }
            assert len(strategy.subject_line_templates) >= 2
            assert len(strategy.cta_options) >= 2
            assert 1 <= strategy.priority <= 5

    def test_select_strategy_payment_issues(self):
        """Payment issues select payment_recovery."""
        strategy = select_strategy("payment_issues")
        assert strategy.id == "payment_recovery"

    def test_select_strategy_disengagement(self):
        """Disengagement selects engagement_reignite."""
        strategy = select_strategy("disengagement")
        assert strategy.id == "engagement_reignite"

    def test_select_strategy_support_frustration(self):
        """Support frustration selects vip_support_rescue."""
        strategy = select_strategy("support_frustration")
        assert strategy.id == "vip_support_rescue"

    def test_select_strategy_price_sensitivity(self):
        """Price sensitivity selects win_back_discount."""
        strategy = select_strategy("price_sensitivity")
        assert strategy.id == "win_back_discount"

    def test_select_strategy_content_gap(self):
        """Content gap selects content_discovery."""
        strategy = select_strategy("content_gap")
        assert strategy.id == "content_discovery"

    def test_select_strategy_technical_issues(self):
        """Technical issues selects vip_support_rescue."""
        strategy = select_strategy("technical_issues")
        assert strategy.id == "vip_support_rescue"

    def test_select_strategy_unknown_driver(self):
        """Unknown driver defaults to engagement_reignite with warning."""
        strategy = select_strategy("some_unknown_cause")
        assert strategy.id == "engagement_reignite"

    def test_no_discount_for_new_accounts(self):
        """Accounts under 30 days tenure don't get discounts."""
        strategy = select_strategy(
            "price_sensitivity",
            {"tenure_days": 15},
        )
        # Should redirect from win_back_discount to content_discovery
        assert strategy.id == "content_discovery"

    def test_discount_allowed_for_established_accounts(self):
        """Accounts over 30 days can get discounts."""
        strategy = select_strategy(
            "price_sensitivity",
            {"tenure_days": 90},
        )
        assert strategy.id == "win_back_discount"

    def test_payment_recovery_for_failed_status(self):
        """payment_failed subscription status always gets payment_recovery."""
        strategy = select_strategy(
            "disengagement",  # Even with different root cause
            {"subscription_status": "payment_failed"},
        )
        assert strategy.id == "payment_recovery"

    def test_deterministic_selection(self):
        """Same inputs always produce same strategy."""
        for _ in range(10):
            s1 = select_strategy("disengagement", {"tenure_days": 100})
            s2 = select_strategy("disengagement", {"tenure_days": 100})
            assert s1.id == s2.id


# =============================================================================
# TestDraftedEmail
# =============================================================================


class TestDraftedEmail:
    """Tests for the DraftedEmail dataclass."""

    def test_email_has_required_fields(self, sample_email_a):
        """DraftedEmail has all required fields."""
        assert sample_email_a.variant in ("A", "B")
        assert sample_email_a.subject
        assert sample_email_a.preview_text
        assert sample_email_a.greeting
        assert sample_email_a.body
        assert sample_email_a.cta_text
        assert sample_email_a.cta_url
        assert sample_email_a.closing
        assert sample_email_a.tone

    def test_subject_under_60_chars(self, sample_email_a, sample_email_b):
        """Subject lines must be under 60 characters."""
        assert len(sample_email_a.subject) < 60
        assert len(sample_email_b.subject) < 60

    def test_preview_under_100_chars(self, sample_email_a, sample_email_b):
        """Preview text must be under 100 characters."""
        assert len(sample_email_a.preview_text) < 100
        assert len(sample_email_b.preview_text) < 100

    def test_body_under_500_words(self, sample_email_a, sample_email_b):
        """Body must be under 500 words."""
        assert len(sample_email_a.body.split()) < 500
        assert len(sample_email_b.body.split()) < 500

    def test_personalization_fields_tracked(self, sample_email_a):
        """Personalization fields used are recorded."""
        assert isinstance(sample_email_a.personalization_fields_used, list)
        assert len(sample_email_a.personalization_fields_used) > 0


# =============================================================================
# TestEmailRenderer
# =============================================================================


class TestEmailRenderer:
    """Tests for email rendering utilities."""

    def test_markdown_render(self, sample_email_a):
        """Markdown render includes key elements."""
        md = render_as_markdown(sample_email_a)

        assert "## Variant A" in md
        assert sample_email_a.subject in md
        assert sample_email_a.greeting in md
        assert sample_email_a.cta_text in md
        assert sample_email_a.closing in md

    def test_html_render_inline_styles(self, sample_email_a):
        """HTML render uses only inline styles, no <style> blocks."""
        html = render_as_html(sample_email_a)

        assert "<style>" not in html
        assert "style=" in html
        assert "<html" in html
        assert "</html>" in html

    def test_html_render_has_cta_button(self, sample_email_a):
        """HTML render includes a prominent CTA button."""
        html = render_as_html(sample_email_a)

        assert sample_email_a.cta_text in html
        assert sample_email_a.cta_url in html
        assert "background-color" in html  # Button styling

    def test_html_render_has_subject(self, sample_email_a):
        """HTML render includes the subject in the title."""
        html = render_as_html(sample_email_a)
        assert sample_email_a.subject in html

    def test_plaintext_render_no_html(self, sample_email_a):
        """Plaintext render contains zero HTML tags."""
        text = render_as_plaintext(sample_email_a)

        assert "<" not in text
        assert ">" not in text
        assert sample_email_a.subject in text
        assert sample_email_a.cta_url in text

    def test_plaintext_render_has_key_elements(self, sample_email_a):
        """Plaintext render includes subject, body, CTA, closing."""
        text = render_as_plaintext(sample_email_a)

        assert "Subject:" in text
        assert sample_email_a.greeting in text
        assert sample_email_a.cta_text in text
        assert sample_email_a.closing in text

    def test_comparison_render_two_variants(self, sample_email_a, sample_email_b):
        """Comparison render shows both variants."""
        comparison = render_comparison([sample_email_a, sample_email_b])

        assert "Variant A" in comparison
        assert "Variant B" in comparison
        assert sample_email_a.subject in comparison
        assert sample_email_b.subject in comparison

    def test_comparison_render_single_email(self, sample_email_a):
        """Comparison with one email falls back to single render."""
        result = render_comparison([sample_email_a])
        assert "Variant A" in result

    def test_comparison_render_empty(self):
        """Comparison with no emails returns placeholder."""
        result = render_comparison([])
        assert "No emails" in result


# =============================================================================
# TestInterventionDrafter
# =============================================================================


class TestInterventionDrafter:
    """Tests for the drafter agent."""

    def test_single_account_drafting(self):
        """Drafter handles single account input."""
        from src.agents.intervention.drafter import draft_intervention

        with patch("src.agents.intervention.drafter.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = """[
                {"variant": "A", "subject": "Test A", "preview_text": "Preview",
                 "greeting": "Hi", "body": "Body text", "cta_text": "Click",
                 "cta_url": "https://example.com", "closing": "Thanks",
                 "tone": "helpful", "personalization_fields_used": []},
                {"variant": "B", "subject": "Test B", "preview_text": "Preview B",
                 "greeting": "Hey", "body": "Body B", "cta_text": "Go",
                 "cta_url": "https://example.com", "closing": "Cheers",
                 "tone": "celebratory", "personalization_fields_used": []}
            ]"""
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = draft_intervention(
                account_id="ACC_00000001",
                churn_driver="disengagement",
            )

        assert isinstance(result, InterventionResult)
        assert result.account_id == "ACC_00000001"
        assert result.churn_driver == "disengagement"
        assert result.strategy.id == "engagement_reignite"

    def test_cohort_drafting(self):
        """Drafter handles multiple account input."""
        from src.agents.intervention.drafter import draft_intervention

        with patch("src.agents.intervention.drafter.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = """[
                {"variant": "A", "subject": "Team offer", "preview_text": "P",
                 "greeting": "Hi {name}", "body": "Body", "cta_text": "Go",
                 "cta_url": "https://example.com", "closing": "Thanks",
                 "tone": "helpful", "personalization_fields_used": ["name"]}
            ]"""
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = draft_intervention(
                account_ids=["ACC_00000001", "ACC_00000002"],
                churn_driver="payment_issues",
            )

        assert result.account_ids == ["ACC_00000001", "ACC_00000002"]
        assert result.account_id is None

    def test_pre_diagnosed_driver_skips_diagnosis(self):
        """Providing churn_driver skips the diagnosis node."""
        from src.agents.intervention.drafter import draft_intervention

        with patch("src.agents.intervention.drafter.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = '[]'
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = draft_intervention(
                account_id="ACC_00000001",
                churn_driver="support_frustration",
            )

        # Strategy should match the provided driver
        assert result.strategy.id == "vip_support_rescue"

    def test_nonexistent_account_handles_gracefully(self):
        """Drafter doesn't crash on nonexistent accounts."""
        from src.agents.intervention.drafter import draft_intervention

        with patch("src.agents.intervention.drafter.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = '[]'
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = draft_intervention(
                account_id="ACC_99999999",
                churn_driver="disengagement",
            )

        assert isinstance(result, InterventionResult)

    def test_mutual_exclusivity(self):
        """Cannot specify both account_id and account_ids."""
        from src.agents.intervention.drafter import draft_intervention

        with pytest.raises(ValueError, match="not both"):
            draft_intervention(
                account_id="ACC_00000001",
                account_ids=["ACC_00000002"],
            )

    def test_requires_target(self):
        """Must specify at least one target."""
        from src.agents.intervention.drafter import draft_intervention

        with pytest.raises(ValueError, match="Must specify"):
            draft_intervention()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests requiring database and/or LLM."""

    @pytest.mark.db
    def test_full_single_account_flow(self, db_engine):
        """Full flow for a single account with real DB."""
        from src.agents.intervention.drafter import draft_intervention

        # This will use real DB but mock LLM unless API key is set
        result = draft_intervention(
            account_id="ACC_00000001",
            churn_driver="disengagement",
        )
        assert isinstance(result, InterventionResult)

    def test_pipeline_integration(self):
        """DDP prescription results can feed into drafter."""
        # Simulate prescription_results from DDP pipeline
        prescription_results = {
            "intervention_strategy": "content_discovery",
            "target_accounts": ["ACC_00000001", "ACC_00000002"],
        }

        # The drafter should accept these as inputs
        from src.agents.intervention.drafter import draft_intervention

        with patch("src.agents.intervention.drafter.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = '[]'
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = draft_intervention(
                account_ids=prescription_results["target_accounts"],
                churn_driver="content_gap",
            )

        assert result.account_ids == ["ACC_00000001", "ACC_00000002"]
        assert result.strategy.id == "content_discovery"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "TestInterventionStrategies or TestEmailRenderer"])
