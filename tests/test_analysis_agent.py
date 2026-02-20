"""
Tests for the Analysis Agent (Phase 3).

Run with: pytest tests/test_analysis_agent.py -v -k "not db"
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.config import AgentConfig
from src.agents.analysis.narratives import (
    AnalysisNarrative,
    KPIDefinition,
    KPI_REGISTRY,
    NarrativeSection,
    PageContext,
    assess_kpi_health,
    compute_overall_sentiment,
    format_narrative_markdown,
)


# =============================================================================
# TestNarratives — Pure Python dataclasses and deterministic logic
# =============================================================================


class TestNarratives:
    """Tests for narrative dataclasses and helper functions."""

    def test_page_context_enum_values(self):
        """PageContext has all expected values."""
        assert PageContext.EXECUTIVE_SUMMARY.value == "executive_summary"
        assert PageContext.ANALYTICS_DEEP_DIVE.value == "analytics_deep_dive"
        assert PageContext.AT_RISK_DETAIL.value == "at_risk_detail"
        assert PageContext.PRESCRIPTION_SUMMARY.value == "prescription_summary"
        assert len(PageContext) == 4

    def test_kpi_registry_has_all_definitions(self):
        """KPI_REGISTRY contains all expected KPIs."""
        expected_kpis = {
            "churn_rate_30d",
            "retention_rate_30d",
            "high_risk_count",
            "monthly_recurring_revenue",
            "avg_customer_lifetime_days",
            "payment_failure_rate",
            "avg_resolution_hours",
            "engagement_watch_hours_weekly",
        }
        assert set(KPI_REGISTRY.keys()) == expected_kpis

        for name, defn in KPI_REGISTRY.items():
            assert isinstance(defn, KPIDefinition)
            assert defn.name == name
            assert defn.description
            assert defn.healthy_range
            assert defn.unit

    def test_assess_kpi_health_below_warning(self):
        """KPI within healthy range returns 'healthy'."""
        assert assess_kpi_health("churn_rate_30d", 3.0) == "healthy"
        assert assess_kpi_health("retention_rate_30d", 95.0) == "healthy"
        assert assess_kpi_health("payment_failure_rate", 2.0) == "healthy"
        assert assess_kpi_health("avg_resolution_hours", 20.0) == "healthy"

    def test_assess_kpi_health_above_warning(self):
        """KPI above warning threshold returns 'warning' or 'critical'."""
        assert assess_kpi_health("churn_rate_30d", 6.0) == "warning"
        assert assess_kpi_health("churn_rate_30d", 10.0) == "critical"
        assert assess_kpi_health("retention_rate_30d", 89.0) == "warning"
        assert assess_kpi_health("retention_rate_30d", 85.0) == "critical"
        assert assess_kpi_health("payment_failure_rate", 6.0) == "warning"
        assert assess_kpi_health("avg_resolution_hours", 50.0) == "warning"

    def test_assess_kpi_health_unknown_kpi(self):
        """Unknown KPI returns 'healthy' by default."""
        assert assess_kpi_health("unknown_metric", 999.0) == "healthy"

    def test_compute_overall_sentiment_with_critical(self):
        """Any 'critical' section → 'action_needed'."""
        sentiments = ["positive", "neutral", "critical"]
        assert compute_overall_sentiment(sentiments) == "action_needed"

    def test_compute_overall_sentiment_with_concerning(self):
        """Any 'concerning' section → 'watch_closely'."""
        sentiments = ["positive", "concerning", "neutral"]
        assert compute_overall_sentiment(sentiments) == "watch_closely"

    def test_compute_overall_sentiment_all_healthy(self):
        """All positive/neutral → 'healthy'."""
        sentiments = ["positive", "neutral", "positive"]
        assert compute_overall_sentiment(sentiments) == "healthy"

    def test_compute_overall_sentiment_empty(self):
        """Empty list → 'healthy'."""
        assert compute_overall_sentiment([]) == "healthy"

    def test_format_narrative_markdown_structure(self):
        """format_narrative_markdown produces valid markdown."""
        narrative = AnalysisNarrative(
            page_context="executive_summary",
            generated_at="2025-12-01T00:00:00",
            reference_date="2025-12-01",
            headline="Platform health is stable",
            sections=[
                NarrativeSection(
                    heading="Churn Overview",
                    body="The churn rate stands at 4.9%.",
                    kpis_referenced=["churn_rate_30d"],
                    sentiment="neutral",
                ),
            ],
            overall_sentiment="healthy",
            key_callouts=["Churn rate is within healthy range"],
            raw_kpis={"churn_rate_30d": 4.9},
        )
        md = format_narrative_markdown(narrative)

        assert "# Analysis: Executive Summary" in md
        assert "2025-12-01" in md
        assert "Platform health is stable" in md
        assert "Churn Overview" in md
        assert "4.9%" in md
        assert "Key Highlights" in md

    def test_narrative_section_data_points_default(self):
        """NarrativeSection data_points defaults to empty dict."""
        section = NarrativeSection(
            heading="Test",
            body="Test body",
            kpis_referenced=[],
            sentiment="neutral",
        )
        assert section.data_points == {}

    def test_analysis_narrative_defaults(self):
        """AnalysisNarrative defaults are correct."""
        narrative = AnalysisNarrative(
            page_context="executive_summary",
            generated_at="2025-12-01T00:00:00",
            reference_date="2025-12-01",
            headline="Test",
            sections=[],
            overall_sentiment="healthy",
            key_callouts=[],
            raw_kpis={},
        )
        assert narrative.success is True
        assert narrative.errors == []
        assert narrative.metadata == {}


# =============================================================================
# TestAnalysisAgent — Graph structure and integration
# =============================================================================


class TestAnalysisAgent:
    """Tests for the Analysis Agent graph."""

    def test_graph_compiles(self):
        """Agent graph compiles without error."""
        from src.agents.analysis.analyzer import create_analysis_agent

        graph = create_analysis_agent()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """Graph contains all expected nodes."""
        from src.agents.analysis.analyzer import create_analysis_agent

        graph = create_analysis_agent()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"gather_data", "generate_narrative"}
        assert expected.issubset(node_names)

    def test_gather_data_executive_summary_tools(self):
        """gather_data for executive_summary calls correct tools."""
        from src.agents.analysis.analyzer import TOOL_SETS

        tools = TOOL_SETS["executive_summary"]
        tool_names = [t[0].name for t in tools]
        assert "get_executive_kpis" in tool_names
        assert "get_account_risk_scores" in tool_names
        assert "get_model_health_status" in tool_names

    def test_gather_data_analytics_deep_dive_tools(self):
        """gather_data for analytics_deep_dive calls ALL analytics tools."""
        from src.agents.analysis.analyzer import TOOL_SETS

        tools = TOOL_SETS["analytics_deep_dive"]
        tool_names = [t[0].name for t in tools]
        assert "get_executive_kpis" in tool_names
        assert "get_subscription_distribution" in tool_names
        assert "get_engagement_trends" in tool_names
        assert "get_support_health" in tool_names
        assert "get_payment_health" in tool_names
        assert "get_churn_cohort_analysis" in tool_names
        assert "get_model_health_status" in tool_names
        assert len(tools) == 7  # all 7 analytics tools

    def test_gather_data_at_risk_tools(self):
        """gather_data for at_risk_detail calls correct tools."""
        from src.agents.analysis.analyzer import TOOL_SETS

        tools = TOOL_SETS["at_risk_detail"]
        tool_names = [t[0].name for t in tools]
        assert "get_account_risk_scores" in tool_names
        assert "get_subscription_distribution" in tool_names
        assert "get_executive_kpis" in tool_names
        assert "segment_high_risk_accounts" in tool_names

    def test_gather_data_prescription_tools(self):
        """gather_data for prescription_summary calls correct tools."""
        from src.agents.analysis.analyzer import TOOL_SETS

        tools = TOOL_SETS["prescription_summary"]
        tool_names = [t[0].name for t in tools]
        assert "get_account_risk_scores" in tool_names
        assert "get_executive_kpis" in tool_names
        assert "get_feature_importance_global" in tool_names

    def test_narrative_schema_completeness(self):
        """AnalysisNarrative has all required fields."""
        narrative = AnalysisNarrative(
            page_context="executive_summary",
            generated_at="2025-12-01T00:00:00",
            reference_date="2025-12-01",
            headline="Test headline",
            sections=[
                NarrativeSection(
                    heading="Test",
                    body="Test body",
                    kpis_referenced=["churn_rate_30d"],
                    sentiment="neutral",
                ),
            ],
            overall_sentiment="healthy",
            key_callouts=["Callout 1"],
            raw_kpis={"churn_rate_30d": 4.9},
            metadata={"test": True},
        )
        assert narrative.page_context == "executive_summary"
        assert narrative.headline == "Test headline"
        assert len(narrative.sections) == 1
        assert narrative.overall_sentiment == "healthy"
        assert len(narrative.key_callouts) == 1
        assert "churn_rate_30d" in narrative.raw_kpis

    def test_invalid_page_context_raises(self):
        """Invalid page_context raises ValueError."""
        from src.agents.analysis.analyzer import run_analysis

        with pytest.raises(ValueError, match="Invalid page_context"):
            run_analysis(page_context="invalid_page")

    def test_raw_kpis_included_in_output(self):
        """raw_kpis is populated in the narrative."""
        narrative = AnalysisNarrative(
            page_context="executive_summary",
            generated_at="2025-12-01T00:00:00",
            reference_date="2025-12-01",
            headline="Test",
            sections=[],
            overall_sentiment="healthy",
            key_callouts=[],
            raw_kpis={"churn_rate_30d": 4.9, "high_risk_count": 2420},
        )
        assert narrative.raw_kpis["churn_rate_30d"] == 4.9
        assert narrative.raw_kpis["high_risk_count"] == 2420

    def test_key_callouts_max_four(self):
        """key_callouts should have at most 4 items."""
        narrative = AnalysisNarrative(
            page_context="executive_summary",
            generated_at="2025-12-01T00:00:00",
            reference_date="2025-12-01",
            headline="Test",
            sections=[],
            overall_sentiment="healthy",
            key_callouts=["1", "2", "3", "4"],
            raw_kpis={},
        )
        assert len(narrative.key_callouts) <= 4

    def test_narrative_serializable(self):
        """AnalysisNarrative is JSON-serializable via safe_json_serialize."""
        from src.agents.utils import safe_json_serialize

        narrative = AnalysisNarrative(
            page_context="executive_summary",
            generated_at="2025-12-01T00:00:00",
            reference_date="2025-12-01",
            headline="Test",
            sections=[
                NarrativeSection(
                    heading="Test",
                    body="Body",
                    kpis_referenced=["churn_rate_30d"],
                    sentiment="neutral",
                ),
            ],
            overall_sentiment="healthy",
            key_callouts=["One"],
            raw_kpis={"churn_rate_30d": 4.9},
        )
        serialized = safe_json_serialize(narrative)
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["page_context"] == "executive_summary"

    def test_gather_data_node_output_schema(self):
        """gather_data returns expected schema with kpis and supplementary_data."""
        mock_kpis = json.dumps({
            "total_active_accounts": 49684,
            "total_churned_accounts": 685,
            "churn_rate_30d": 4.9,
            "retention_rate_30d": 95.1,
            "high_risk_count": 2420,
            "monthly_recurring_revenue": 33880,
            "avg_customer_lifetime_days": 400,
            "avg_churn_probability": 0.35,
            "reference_date": "2025-12-01",
        })
        mock_risk = "| account_id | churn_probability |\n| --- | --- |\n| ACC_00000001 | 0.85 |"
        mock_model = "Model health: HEALTHY"

        with patch(
            "src.agents.analysis.analyzer.get_executive_kpis"
        ) as mock_kpi_tool, patch(
            "src.agents.analysis.analyzer.get_account_risk_scores"
        ) as mock_risk_tool, patch(
            "src.agents.analysis.analyzer.get_model_health_status"
        ) as mock_model_tool:
            mock_kpi_tool.invoke = MagicMock(return_value=mock_kpis)
            mock_kpi_tool.name = "get_executive_kpis"
            mock_kpi_tool.args_schema = MagicMock()
            mock_kpi_tool.args_schema.model_fields = {"reference_date": True}

            mock_risk_tool.invoke = MagicMock(return_value=mock_risk)
            mock_risk_tool.name = "get_account_risk_scores"
            mock_risk_tool.args_schema = MagicMock()
            mock_risk_tool.args_schema.model_fields = {}

            mock_model_tool.invoke = MagicMock(return_value=mock_model)
            mock_model_tool.name = "get_model_health_status"
            mock_model_tool.args_schema = MagicMock()
            mock_model_tool.args_schema.model_fields = {}

            # Replace TOOL_SETS for this test
            with patch(
                "src.agents.analysis.analyzer.TOOL_SETS",
                {
                    "executive_summary": [
                        (mock_kpi_tool, {}),
                        (mock_risk_tool, {"risk_tier": "high"}),
                        (mock_model_tool, {}),
                    ],
                },
            ):
                from src.agents.analysis.analyzer import gather_data

                state = {
                    "config": AgentConfig(),
                    "page_context": "executive_summary",
                    "errors": [],
                }
                result = gather_data(state)

        raw_data = result["raw_data"]
        assert raw_data["page_context"] == "executive_summary"
        assert "kpis" in raw_data
        assert "supplementary_data" in raw_data
        assert "churn_rate_30d" in raw_data["kpis"]
        assert raw_data["kpis"]["churn_rate_30d"]["value"] == 4.9
        assert raw_data["kpis"]["churn_rate_30d"]["health"] == "healthy"
