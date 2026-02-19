"""
Tests for the DDP (Detect → Diagnose → Prescribe) pipeline.

Run with: pytest tests/test_ddp_pipeline.py -v -k "test_graph or test_route"
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.config import AgentConfig
from src.agents.pipelines.ddp_nodes import (
    _extract_json_from_response,
    route_by_phase,
)
from src.agents.state import RetainAgentState, create_initial_state


# =============================================================================
# Graph Structure Tests
# =============================================================================


class TestDDPPipelineGraph:
    """Tests for graph compilation and structure."""

    @patch("src.agents.pipelines.ddp_nodes.ChatAnthropic")
    def test_graph_compiles_without_error(self, mock_llm):
        """Graph compiles successfully."""
        from src.agents.pipelines.ddp_pipeline import create_ddp_pipeline

        graph = create_ddp_pipeline()
        assert graph is not None

    @patch("src.agents.pipelines.ddp_nodes.ChatAnthropic")
    def test_graph_has_all_nodes(self, mock_llm):
        """Graph contains all required node names."""
        from src.agents.pipelines.ddp_pipeline import create_ddp_pipeline

        graph = create_ddp_pipeline()

        # Check the graph's nodes via its underlying graph
        node_names = set(graph.get_graph().nodes.keys())

        expected_nodes = {"supervisor", "detect", "diagnose", "prescribe", "review"}
        # LangGraph adds __start__ and __end__ nodes
        assert expected_nodes.issubset(node_names)

    @patch("src.agents.pipelines.ddp_nodes.ChatAnthropic")
    def test_graph_has_correct_edges(self, mock_llm):
        """Graph has edges from phase nodes back to supervisor."""
        from src.agents.pipelines.ddp_pipeline import create_ddp_pipeline

        graph = create_ddp_pipeline()
        graph_repr = graph.get_graph()

        # Check that detect, diagnose, prescribe, review all have edges to supervisor
        edges = graph_repr.edges
        edge_pairs = [(e.source, e.target) for e in edges]

        assert ("detect", "supervisor") in edge_pairs
        assert ("diagnose", "supervisor") in edge_pairs
        assert ("prescribe", "supervisor") in edge_pairs
        assert ("review", "supervisor") in edge_pairs


# =============================================================================
# Route Logic Tests (pure function, no mocks needed)
# =============================================================================


class TestRouteByPhase:
    """Tests for the routing function."""

    def test_route_idle_goes_to_detect(self):
        """Idle phase routes to detect."""
        state = {"current_phase": "idle"}
        assert route_by_phase(state) == "detect"

    def test_route_detection_goes_to_diagnose(self):
        """Detection phase routes to diagnose."""
        state = {"current_phase": "detection"}
        assert route_by_phase(state) == "diagnose"

    def test_route_diagnosis_goes_to_prescribe(self):
        """Diagnosis phase routes to prescribe."""
        state = {"current_phase": "diagnosis"}
        assert route_by_phase(state) == "prescribe"

    def test_route_prescription_goes_to_review(self):
        """Prescription phase routes to review."""
        state = {"current_phase": "prescription"}
        assert route_by_phase(state) == "review"

    def test_route_review_goes_to_supervisor(self):
        """Review phase routes back to supervisor."""
        state = {"current_phase": "review"}
        assert route_by_phase(state) == "supervisor"

    def test_route_complete_goes_to_end(self):
        """Complete phase routes to __end__."""
        state = {"current_phase": "complete"}
        assert route_by_phase(state) == "__end__"

    def test_route_unknown_goes_to_end(self):
        """Unknown phase defaults to __end__."""
        state = {"current_phase": "unknown_phase"}
        assert route_by_phase(state) == "__end__"

    def test_route_missing_phase_defaults(self):
        """Missing phase key defaults to idle -> detect."""
        state = {}
        assert route_by_phase(state) == "detect"


# =============================================================================
# State Transition Tests
# =============================================================================


class TestStateTransitions:
    """Tests for state transitions through nodes."""

    def test_initial_state_starts_idle(self):
        """Initial state has phase=idle."""
        state = create_initial_state()
        assert state["current_phase"] == "idle"

    def test_detection_sets_phase(self):
        """Detection node sets phase to 'detection'."""
        from src.agents.pipelines.ddp_nodes import detection_node

        state = create_initial_state()
        state["current_phase"] = "detection"

        with patch("src.agents.pipelines.ddp_nodes.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = '{"total_accounts_scored": 100}'
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = detection_node(state)

        assert result["current_phase"] == "detection"
        assert result["detection_results"] is not None

    def test_diagnosis_requires_detection_results(self):
        """Diagnosis node reads detection_results from state."""
        from src.agents.pipelines.ddp_nodes import diagnosis_node

        state = create_initial_state()
        state["current_phase"] = "diagnosis"
        state["detection_results"] = {
            "highest_risk_cohort_id": 0,
            "cohorts": [{"cohort_id": 0, "sample_account_ids": ["ACC_00000001"]}],
        }

        with patch("src.agents.pipelines.ddp_nodes.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = '{"primary_root_cause": "disengagement"}'
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = diagnosis_node(state)

        assert result["diagnosis_results"] is not None

    def test_prescription_requires_diagnosis_results(self):
        """Prescription node reads diagnosis_results from state."""
        from src.agents.pipelines.ddp_nodes import prescription_node

        state = create_initial_state()
        state["current_phase"] = "prescription"
        state["detection_results"] = {"cohorts": []}
        state["diagnosis_results"] = {
            "primary_root_cause": "disengagement",
            "accounts_investigated": ["ACC_00000001"],
        }

        with patch("src.agents.pipelines.ddp_nodes.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = '{"intervention_strategy": "content_discovery"}'
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = prescription_node(state)

        assert result["prescription_results"] is not None


# =============================================================================
# Output Schema Tests
# =============================================================================


class TestOutputSchemas:
    """Tests for structured output parsing."""

    def test_detection_results_schema(self):
        """Detection results JSON has required fields."""
        sample = {
            "total_accounts_scored": 1000,
            "risk_distribution": {"high": 100, "medium": 300, "low": 600},
            "mean_churn_probability": 0.35,
            "cohorts": [
                {
                    "cohort_id": 0,
                    "size": 50,
                    "mean_churn_prob": 0.82,
                    "distinguishing_features": {},
                    "sample_account_ids": ["ACC_00000001"],
                }
            ],
            "highest_risk_cohort_id": 0,
        }

        # Verify required keys
        required_keys = {
            "total_accounts_scored", "risk_distribution",
            "mean_churn_probability", "cohorts", "highest_risk_cohort_id",
        }
        assert required_keys.issubset(sample.keys())

        # Verify cohort structure
        cohort = sample["cohorts"][0]
        cohort_keys = {"cohort_id", "size", "mean_churn_prob",
                       "distinguishing_features", "sample_account_ids"}
        assert cohort_keys.issubset(cohort.keys())

    def test_diagnosis_results_schema(self):
        """Diagnosis results JSON has required fields."""
        sample = {
            "cohort_analyzed": 0,
            "accounts_investigated": ["ACC_00000001", "ACC_00000002"],
            "primary_root_cause": "disengagement",
            "secondary_factors": ["content_gap"],
            "evidence": {
                "shap_top_features": [],
                "ticket_patterns": {},
                "payment_failure_rate": 0.05,
                "avg_watch_hours_trend": "declining",
                "common_characteristics": {},
            },
            "narrative": "The cohort shows declining engagement.",
        }

        required_keys = {
            "cohort_analyzed", "accounts_investigated", "primary_root_cause",
            "secondary_factors", "evidence", "narrative",
        }
        assert required_keys.issubset(sample.keys())

        valid_root_causes = {
            "payment_issues", "disengagement", "support_frustration",
            "price_sensitivity", "content_gap", "technical_issues",
        }
        assert sample["primary_root_cause"] in valid_root_causes

    def test_prescription_results_schema(self):
        """Prescription results JSON has required fields."""
        sample = {
            "intervention_strategy": "content_discovery",
            "target_cohort_size": 50,
            "target_accounts": ["ACC_00000001"],
            "email_templates": [
                {
                    "template_id": "tmpl_001",
                    "subject": "New shows you'll love",
                    "body": "Check out these picks...",
                    "cta_text": "Browse Now",
                    "cta_url": "https://retain.example.com/browse",
                    "tone": "celebratory",
                }
            ],
            "estimated_impact": {
                "accounts_targeted": 50,
                "projected_save_rate": 0.15,
                "projected_accounts_saved": 8,
                "projected_monthly_revenue_saved": 120.0,
            },
            "implementation_notes": "Deploy via email campaign tool.",
        }

        required_keys = {
            "intervention_strategy", "target_cohort_size", "target_accounts",
            "email_templates", "estimated_impact", "implementation_notes",
        }
        assert required_keys.issubset(sample.keys())

        impact_keys = {
            "accounts_targeted", "projected_save_rate",
            "projected_accounts_saved", "projected_monthly_revenue_saved",
        }
        assert impact_keys.issubset(sample["estimated_impact"].keys())


# =============================================================================
# Result Parsing Tests
# =============================================================================


class TestResultParsing:
    """Tests for JSON extraction from LLM responses."""

    def test_parse_detection_results_valid_json(self):
        """Valid JSON content is parsed correctly."""
        content = '{"total_accounts_scored": 1000, "risk_distribution": {"high": 100}}'
        result = _extract_json_from_response(content)

        assert result is not None
        assert result["total_accounts_scored"] == 1000

    def test_parse_detection_results_json_in_code_fence(self):
        """JSON inside markdown code fences is extracted."""
        content = """Here are the results:

```json
{"total_accounts_scored": 500}
```

That completes the detection phase."""

        result = _extract_json_from_response(content)

        assert result is not None
        assert result["total_accounts_scored"] == 500

    def test_parse_detection_results_malformed_response(self):
        """Non-JSON content returns None."""
        content = "I analyzed the accounts and found some issues."
        result = _extract_json_from_response(content)

        assert result is None

    def test_parse_diagnosis_results_valid_root_cause(self):
        """Parsed diagnosis has a valid root cause category."""
        valid_causes = {
            "payment_issues", "disengagement", "support_frustration",
            "price_sensitivity", "content_gap", "technical_issues",
        }

        content = '{"primary_root_cause": "disengagement", "narrative": "Users stopped watching."}'
        result = _extract_json_from_response(content)

        assert result is not None
        assert result["primary_root_cause"] in valid_causes

    def test_parse_json_embedded_in_text(self):
        """JSON embedded in natural language text is extracted."""
        content = 'The analysis shows: {"key": "value", "count": 42} as the result.'
        result = _extract_json_from_response(content)

        assert result is not None
        assert result["key"] == "value"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for pipeline nodes."""

    def test_empty_high_risk_cohort(self):
        """Detection handles zero high-risk accounts."""
        from src.agents.pipelines.ddp_nodes import detection_node

        state = create_initial_state()

        with patch("src.agents.pipelines.ddp_nodes.ChatAnthropic") as MockLLM:
            mock_response = MagicMock()
            mock_response.content = json.dumps({
                "total_accounts_scored": 1000,
                "risk_distribution": {"high": 0, "medium": 100, "low": 900},
                "mean_churn_probability": 0.15,
                "cohorts": [],
                "highest_risk_cohort_id": None,
            })
            mock_response.tool_calls = []

            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm

            result = detection_node(state)

        assert result["detection_results"] is not None
        # Should still complete without error
        assert "error" not in result.get("errors", [])

    def test_human_feedback_approve(self):
        """Supervisor advances past review when feedback is 'approve'."""
        from src.agents.pipelines.ddp_nodes import supervisor_node

        state = create_initial_state()
        state["current_phase"] = "review"
        state["human_feedback"] = "approve"
        state["prescription_results"] = {"strategy": "test"}

        result = supervisor_node(state)
        assert result["current_phase"] == "complete"

    def test_human_feedback_reject(self):
        """Supervisor marks rejected when feedback is 'reject'."""
        from src.agents.pipelines.ddp_nodes import supervisor_node

        state = create_initial_state()
        state["current_phase"] = "review"
        state["human_feedback"] = "reject"

        result = supervisor_node(state)
        assert result["current_phase"] == "complete"
        assert result["metadata"]["status"] == "rejected"

    def test_human_feedback_modify(self):
        """Supervisor sends back to prescribe with modification instructions."""
        from src.agents.pipelines.ddp_nodes import supervisor_node

        state = create_initial_state()
        state["current_phase"] = "review"
        state["human_feedback"] = "Make the tone more urgent"

        result = supervisor_node(state)
        assert result["current_phase"] == "prescription"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "test_graph or test_route"])
