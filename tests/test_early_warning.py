"""
Tests for the Early Warning Agent (Phase 2).

Run with: pytest tests/test_early_warning.py -v -k "not db"
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.config import AgentConfig
from src.agents.early_warning.alerts import (
    VALID_ROOT_CAUSES,
    AlertGroup,
    EarlyWarningReport,
    RiskTransition,
    classify_transitions,
    compose_headline,
    compose_narrative,
    compute_alert_priority,
    format_alert_report_markdown,
)


# =============================================================================
# TestAlerts — Pure Python dataclasses and deterministic logic
# =============================================================================


class TestAlerts:
    """Tests for alert dataclasses and helper functions."""

    def test_risk_transition_direction_escalated(self):
        """Escalated when current tier is higher than previous."""
        t = RiskTransition(
            account_id="ACC_00000001",
            previous_tier="low",
            current_tier="high",
            previous_probability=0.2,
            current_probability=0.8,
            probability_delta=0.0,
            direction="",
        )
        assert t.direction == "escalated"
        assert t.probability_delta == pytest.approx(0.6, abs=0.01)

    def test_risk_transition_direction_improved(self):
        """Improved when current tier is lower than previous."""
        t = RiskTransition(
            account_id="ACC_00000002",
            previous_tier="high",
            current_tier="low",
            previous_probability=0.8,
            current_probability=0.2,
            probability_delta=0.0,
            direction="",
        )
        assert t.direction == "improved"

    def test_risk_transition_direction_stable(self):
        """Stable when tiers are the same."""
        t = RiskTransition(
            account_id="ACC_00000003",
            previous_tier="medium",
            current_tier="medium",
            previous_probability=0.5,
            current_probability=0.55,
            probability_delta=0.0,
            direction="",
        )
        assert t.direction == "stable"

    def test_classify_transitions_groups_correctly(self):
        """classify_transitions separates by direction."""
        transitions = [
            RiskTransition(
                "ACC_00000001", "low", "high", 0.2, 0.8, 0.6, "escalated"
            ),
            RiskTransition(
                "ACC_00000002", "high", "low", 0.8, 0.2, -0.6, "improved"
            ),
            RiskTransition(
                "ACC_00000003", "medium", "medium", 0.5, 0.5, 0.0, "stable"
            ),
            RiskTransition(
                "ACC_00000004", "low", "medium", 0.3, 0.5, 0.2, "escalated"
            ),
        ]
        groups = classify_transitions(transitions)

        assert len(groups["escalated"]) == 2
        assert len(groups["improved"]) == 1
        assert len(groups["stable"]) == 1

    def test_compute_alert_priority_large_high_delta(self):
        """Large group with high delta should be priority 5."""
        priority = compute_alert_priority(group_size=60, avg_delta=0.35)
        assert priority == 5

    def test_compute_alert_priority_small_low_delta(self):
        """Small group with low delta should be priority 1."""
        priority = compute_alert_priority(group_size=2, avg_delta=0.02)
        assert priority == 1

    def test_compute_alert_priority_medium(self):
        """Medium group with medium delta should be priority 3."""
        priority = compute_alert_priority(group_size=15, avg_delta=0.12)
        assert priority == 3

    def test_format_alert_report_markdown_structure(self):
        """format_alert_report_markdown produces valid markdown."""
        group = AlertGroup(
            root_cause="payment_issues",
            accounts=[
                RiskTransition(
                    "ACC_00000001", "medium", "high", 0.5, 0.8, 0.3, "escalated"
                ),
            ],
            representative_account_ids=["ACC_00000001"],
            evidence_summary="Payment failures detected",
            priority=4,
            recommended_action="Trigger payment recovery",
        )
        report = EarlyWarningReport(
            reference_date="2025-12-01",
            previous_date="2025-11-01",
            total_accounts_scored=1000,
            total_escalated=1,
            total_new_high_risk=1,
            total_improved=5,
            alert_groups=[group],
            headline="1 accounts escalated — payment issues drives 100%",
            narrative="Test narrative.",
            model_health="Healthy",
        )
        md = format_alert_report_markdown(report)

        assert "# Early Warning Report" in md
        assert "2025-12-01" in md
        assert "Payment Issues" in md
        assert "ACC_00000001" in md
        assert "Priority 4/5" in md

    def test_compose_headline_zero_escalations(self):
        """Zero escalations produce 'all clear' headline."""
        headline = compose_headline(0, 0, [])
        assert "all clear" in headline.lower()

    def test_compose_headline_with_escalations(self):
        """Headline includes specific count and top cause."""
        group = AlertGroup(
            root_cause="disengagement",
            accounts=[
                RiskTransition(
                    f"ACC_{i:08d}", "low", "high", 0.2, 0.8, 0.6, "escalated"
                )
                for i in range(47)
            ],
            representative_account_ids=["ACC_00000001"],
            evidence_summary="Test",
            priority=5,
            recommended_action="Test",
        )
        headline = compose_headline(47, 47, [group])
        assert "47" in headline
        assert "disengagement" in headline

    def test_compose_narrative_zero_escalations(self):
        """Zero escalations produce stable narrative."""
        report = EarlyWarningReport(
            reference_date="2025-12-01",
            previous_date="2025-11-01",
            total_accounts_scored=1000,
            total_escalated=0,
            total_new_high_risk=0,
            total_improved=10,
            alert_groups=[],
            headline="All clear",
            narrative="",
            model_health="Healthy",
        )
        narrative = compose_narrative(report)
        assert "no accounts moved" in narrative.lower() or "stable" in narrative.lower()
        assert "10" in narrative  # improved count cited


# =============================================================================
# TestEarlyWarningAgent — Graph structure and integration
# =============================================================================


class TestEarlyWarningAgent:
    """Tests for the Early Warning Agent graph."""

    def test_graph_compiles(self):
        """Agent graph compiles without error."""
        from src.agents.early_warning.detector import (
            create_early_warning_agent,
        )

        graph = create_early_warning_agent()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """Graph contains all expected nodes."""
        from src.agents.early_warning.detector import (
            create_early_warning_agent,
        )

        graph = create_early_warning_agent()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"score_comparison", "investigate_escalations", "group_and_report"}
        # LangGraph adds __start__ and __end__ nodes
        assert expected.issubset(node_names)

    def test_zero_escalations_short_circuits(self):
        """When no escalations, investigation returns empty groups."""
        from src.agents.early_warning.detector import investigate_escalations

        state = {
            "config": AgentConfig(),
            "transitions": {
                "total_scored": 100,
                "escalated": [],
                "improved": [{"account_id": "ACC_00000001"}],
                "stable_count": 99,
                "new_high_risk_count": 0,
            },
            "errors": [],
        }
        result = investigate_escalations(state)
        inv = result.get("investigation_results", {})
        assert inv["groups"] == []
        assert inv["ungrouped_count"] == 0

    def test_score_comparison_node_output_schema(self):
        """score_comparison returns expected schema keys."""
        mock_transitions = json.dumps({
            "total_scored": 500,
            "escalated_accounts": [
                {
                    "account_id": "ACC_00000001",
                    "previous_tier": "medium",
                    "current_tier": "high",
                    "previous_probability": 0.5,
                    "current_probability": 0.8,
                    "probability_delta": 0.3,
                    "direction": "escalated",
                }
            ],
            "improved_accounts": [],
            "stable_count": 499,
            "transition_counts": {"medium→high": 1},
        })
        mock_new_high = json.dumps({
            "total_new_high_risk": 1,
            "accounts": [
                {
                    "account_id": "ACC_00000001",
                    "current_probability": 0.8,
                    "previous_probability": 0.5,
                    "probability_delta": 0.3,
                    "previous_tier": "medium",
                    "current_tier": "high",
                }
            ],
        })

        with patch(
            "src.agents.early_warning.detector.get_risk_tier_transitions"
        ) as mock_trans, patch(
            "src.agents.early_warning.detector.get_new_high_risk_accounts"
        ) as mock_new:
            mock_trans.invoke = MagicMock(return_value=mock_transitions)
            mock_new.invoke = MagicMock(return_value=mock_new_high)

            from src.agents.early_warning.detector import score_comparison

            state = {
                "config": AgentConfig(),
                "reference_date": "2025-12-01",
                "previous_date": "2025-11-01",
                "errors": [],
            }
            result = score_comparison(state)

        transitions = result["transitions"]
        assert "total_scored" in transitions
        assert "escalated" in transitions
        assert "improved" in transitions
        assert "stable_count" in transitions
        assert "new_high_risk_count" in transitions
        assert transitions["total_scored"] == 500

    def test_report_schema_completeness(self):
        """EarlyWarningReport has all required fields."""
        report = EarlyWarningReport(
            reference_date="2025-12-01",
            previous_date="2025-11-01",
            total_accounts_scored=1000,
            total_escalated=10,
            total_new_high_risk=5,
            total_improved=3,
            alert_groups=[],
            headline="Test headline",
            narrative="Test narrative",
            model_health="Healthy",
            metadata={"test": True},
            success=True,
            errors=[],
        )
        # Verify all fields are accessible
        assert report.reference_date == "2025-12-01"
        assert report.previous_date == "2025-11-01"
        assert report.total_accounts_scored == 1000
        assert report.total_escalated == 10
        assert report.total_new_high_risk == 5
        assert report.total_improved == 3
        assert isinstance(report.alert_groups, list)
        assert report.headline == "Test headline"
        assert report.narrative == "Test narrative"
        assert report.model_health == "Healthy"
        assert report.success is True

    def test_headline_includes_count(self):
        """Headline mentions specific number of escalations."""
        group = AlertGroup(
            root_cause="payment_issues",
            accounts=[
                RiskTransition(
                    f"ACC_{i:08d}", "low", "high", 0.2, 0.8, 0.6, "escalated"
                )
                for i in range(25)
            ],
            representative_account_ids=["ACC_00000001"],
            evidence_summary="Test",
            priority=4,
            recommended_action="Test",
        )
        headline = compose_headline(25, 25, [group])
        assert "25" in headline

    def test_root_causes_are_valid_categories(self):
        """Only the 6 defined categories are valid."""
        assert len(VALID_ROOT_CAUSES) == 6
        expected = {
            "payment_issues", "disengagement", "support_frustration",
            "price_sensitivity", "content_gap", "technical_issues",
        }
        assert VALID_ROOT_CAUSES == expected

    def test_report_serializable(self):
        """EarlyWarningReport is JSON-serializable via safe_json_serialize."""
        from src.agents.utils import safe_json_serialize

        report = EarlyWarningReport(
            reference_date="2025-12-01",
            previous_date="2025-11-01",
            total_accounts_scored=1000,
            total_escalated=0,
            total_new_high_risk=0,
            total_improved=0,
            alert_groups=[],
            headline="All clear",
            narrative="No escalations.",
            model_health="Healthy",
        )
        serialized = safe_json_serialize(report)
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["reference_date"] == "2025-12-01"

    def test_group_and_report_node(self):
        """group_and_report produces a complete report dict."""
        from src.agents.early_warning.detector import group_and_report

        state = {
            "config": AgentConfig(),
            "reference_date": "2025-12-01",
            "previous_date": "2025-11-01",
            "transitions": {
                "total_scored": 500,
                "escalated": [
                    {
                        "account_id": "ACC_00000001",
                        "previous_tier": "medium",
                        "current_tier": "high",
                        "previous_probability": 0.5,
                        "current_probability": 0.8,
                        "probability_delta": 0.3,
                        "direction": "escalated",
                    }
                ],
                "improved": [],
                "stable_count": 499,
                "new_high_risk_count": 1,
            },
            "investigation_results": {
                "groups": [
                    {
                        "root_cause": "payment_issues",
                        "account_ids": ["ACC_00000001"],
                        "representative_ids": ["ACC_00000001"],
                        "evidence_summary": "Payment failures detected.",
                        "common_features": {"plan_type": "Basic"},
                    }
                ],
                "ungrouped_count": 0,
            },
            "errors": [],
            "metadata": {},
        }
        result = group_and_report(state)

        report = result["report"]
        assert report["total_accounts_scored"] == 500
        assert report["total_escalated"] == 1
        assert report["total_new_high_risk"] == 1
        assert len(report["alert_groups"]) == 1
        assert report["alert_groups"][0]["root_cause"] == "payment_issues"
        assert report["headline"]  # not empty
        assert report["narrative"]  # not empty
