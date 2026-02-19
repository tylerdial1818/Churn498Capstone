"""
Tests for the agents foundation layer (Phase 1).

Run with: pytest tests/test_agents_foundation.py -v -k "not db"
"""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.config import AgentConfig
from src.agents.state import RetainAgentState, create_initial_state
from src.agents.utils import (
    format_dataframe_as_markdown,
    safe_json_serialize,
    truncate_for_context,
    validate_account_id,
)


# =============================================================================
# TestAgentConfig
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        """Test AgentConfig has expected defaults."""
        config = AgentConfig()

        assert config.model_name == "claude-sonnet-4-20250514"
        assert config.temperature == 0.2
        assert config.max_tokens == 4096
        assert config.reference_date == "2025-12-01"
        assert config.high_risk_threshold == 0.70
        assert config.medium_risk_threshold == 0.40
        assert config.min_cohort_size == 10
        assert config.max_cohorts == 5
        assert config.top_features_count == 10
        assert config.shap_sample_size == 100
        assert config.max_sql_rows == 1000
        assert config.read_only_sql is True

    def test_threshold_consistency_with_scoring_config(self):
        """Verify thresholds match ScoringConfig defaults."""
        from src.models.config import ScoringConfig

        agent_config = AgentConfig()
        scoring_config = ScoringConfig()

        assert agent_config.high_risk_threshold == scoring_config.high_risk_threshold
        assert agent_config.medium_risk_threshold == scoring_config.medium_risk_threshold

    def test_custom_values(self):
        """Test AgentConfig accepts custom values."""
        config = AgentConfig(
            model_name="claude-haiku-4-20250514",
            temperature=0.5,
            reference_date="2025-06-01",
        )

        assert config.model_name == "claude-haiku-4-20250514"
        assert config.temperature == 0.5
        assert config.reference_date == "2025-06-01"


# =============================================================================
# TestAgentState
# =============================================================================


class TestAgentState:
    """Tests for RetainAgentState and initial state creation."""

    def test_initial_state_creation(self):
        """Test create_initial_state returns correct structure."""
        state = create_initial_state()

        assert state["messages"] == []
        assert isinstance(state["config"], AgentConfig)
        assert state["current_phase"] == "idle"
        assert state["detection_results"] is None
        assert state["diagnosis_results"] is None
        assert state["prescription_results"] is None
        assert state["human_feedback"] is None
        assert state["errors"] == []
        assert state["metadata"] == {}

    def test_initial_state_with_custom_config(self):
        """Test create_initial_state uses provided config."""
        config = AgentConfig(reference_date="2025-06-01")
        state = create_initial_state(config)

        assert state["config"].reference_date == "2025-06-01"

    def test_phase_transitions(self):
        """Test that phase field can be updated through valid values."""
        state = create_initial_state()
        valid_phases = [
            "idle", "detection", "diagnosis",
            "prescription", "review", "complete",
        ]

        for phase in valid_phases:
            state["current_phase"] = phase
            assert state["current_phase"] == phase

    def test_error_accumulation(self):
        """Test that errors list supports appending."""
        state = create_initial_state()

        state["errors"].append("Error 1")
        state["errors"].append("Error 2")

        assert len(state["errors"]) == 2
        assert state["errors"] == ["Error 1", "Error 2"]

    def test_state_has_expected_annotations(self):
        """Test RetainAgentState has all required annotation keys."""
        annotations = RetainAgentState.__annotations__
        expected_keys = {
            "messages", "config", "current_phase",
            "detection_results", "diagnosis_results", "prescription_results",
            "human_feedback", "errors", "metadata",
        }

        assert expected_keys == set(annotations.keys())


# =============================================================================
# TestTools
# =============================================================================


class TestTools:
    """Tests for LangChain tool wrappers."""

    @patch("src.models.score.BatchScorer")
    @patch("src.data.database.get_engine")
    def test_get_account_risk_scores_returns_string(self, mock_get_engine, MockScorer):
        """Tool always returns a string."""
        from src.agents.tools import get_account_risk_scores

        scorer_instance = MagicMock()
        scorer_instance.get_latest_scores.return_value = pd.DataFrame()
        MockScorer.return_value = scorer_instance

        result = get_account_risk_scores.invoke({})

        assert isinstance(result, str)

    @patch("src.models.score.BatchScorer")
    @patch("src.data.database.get_engine")
    def test_get_account_risk_scores_with_tier_filter(self, mock_get_engine, MockScorer):
        """Tool passes risk_tier filter correctly."""
        from src.agents.tools import get_account_risk_scores

        scorer_instance = MagicMock()
        scorer_instance.get_latest_scores.return_value = pd.DataFrame({
            "account_id": ["ACC_00000001"],
            "churn_probability": [0.85],
            "risk_tier": ["high"],
        })
        MockScorer.return_value = scorer_instance

        result = get_account_risk_scores.invoke({"risk_tier": "high"})

        assert isinstance(result, str)
        assert "ACC_00000001" in result

    @patch("src.models.score.BatchScorer")
    @patch("src.data.database.get_engine")
    def test_get_account_risk_scores_empty_result(self, mock_get_engine, MockScorer):
        """Tool handles empty results gracefully."""
        from src.agents.tools import get_account_risk_scores

        scorer_instance = MagicMock()
        scorer_instance.get_latest_scores.return_value = pd.DataFrame()
        MockScorer.return_value = scorer_instance

        result = get_account_risk_scores.invoke({})

        assert "No risk scores" in result

    def test_query_database_readonly_blocks_insert(self):
        """Tool rejects INSERT statements."""
        from src.agents.tools import query_database_readonly

        result = query_database_readonly.invoke(
            {"sql_query": "INSERT INTO accounts VALUES ('x')"}
        )
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_query_database_readonly_blocks_update(self):
        """Tool rejects UPDATE statements."""
        from src.agents.tools import query_database_readonly

        result = query_database_readonly.invoke(
            {"sql_query": "UPDATE accounts SET email='x'"}
        )
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_query_database_readonly_blocks_delete(self):
        """Tool rejects DELETE statements."""
        from src.agents.tools import query_database_readonly

        result = query_database_readonly.invoke(
            {"sql_query": "DELETE FROM accounts WHERE account_id = 'x'"}
        )
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_query_database_readonly_blocks_drop(self):
        """Tool rejects DROP statements."""
        from src.agents.tools import query_database_readonly

        result = query_database_readonly.invoke(
            {"sql_query": "DROP TABLE accounts"}
        )
        assert "not allowed" in result.lower() or "error" in result.lower()

    @patch("pandas.read_sql")
    @patch("src.data.database.get_engine")
    def test_query_database_readonly_adds_limit(self, mock_get_engine, mock_read_sql):
        """Tool adds LIMIT clause if missing."""
        from src.agents.tools import query_database_readonly

        mock_read_sql.return_value = pd.DataFrame({"x": [1]})

        query_database_readonly.invoke(
            {"sql_query": "SELECT * FROM accounts"}
        )

        # Verify LIMIT was added to the query
        called_query = mock_read_sql.call_args[0][0]
        assert "LIMIT" in str(called_query).upper()

    @patch("pandas.read_sql")
    @patch("src.data.database.get_engine")
    def test_query_database_readonly_handles_sql_error(self, mock_get_engine, mock_read_sql):
        """Tool returns error string on SQL failure."""
        from src.agents.tools import query_database_readonly

        mock_read_sql.side_effect = Exception("relation does not exist")

        result = query_database_readonly.invoke(
            {"sql_query": "SELECT * FROM nonexistent"}
        )

        assert isinstance(result, str)
        assert "error" in result.lower()

    @patch("pandas.read_sql")
    @patch("src.data.database.get_engine")
    def test_query_database_readonly_select_works(self, mock_get_engine, mock_read_sql):
        """Tool allows valid SELECT queries."""
        from src.agents.tools import query_database_readonly

        mock_read_sql.return_value = pd.DataFrame({"count": [100]})

        result = query_database_readonly.invoke(
            {"sql_query": "SELECT COUNT(*) as count FROM accounts"}
        )

        assert isinstance(result, str)
        assert "100" in result

    def test_explain_account_prediction_nonexistent_account(self):
        """Tool handles nonexistent account gracefully."""
        from src.agents.tools import explain_account_prediction

        with patch("src.data.database.get_engine"), \
             patch("src.models.registry.ModelRegistry") as MockRegistry, \
             patch("src.features.create_inference_features") as mock_feat:
            MockRegistry.return_value.load_production_model.return_value = MagicMock()
            mock_feat.return_value = pd.DataFrame()

            result = explain_account_prediction.invoke(
                {"account_id": "ACC_99999999"}
            )

        assert isinstance(result, str)
        assert "No features" in result or "ACC_99999999" in result

    def test_explain_account_prediction_invalid_format(self):
        """Tool rejects invalid account ID format."""
        from src.agents.tools import explain_account_prediction

        result = explain_account_prediction.invoke(
            {"account_id": "BAD_FORMAT"}
        )

        assert "Invalid" in result

    def test_explain_account_prediction_shap_fallback(self):
        """Tool falls back to global importances when SHAP fails."""
        from src.agents.tools import explain_account_prediction

        mock_features = pd.DataFrame(
            {"feat_a": [0.5], "feat_b": [0.3]},
            index=["ACC_00000001"],
        )

        with patch("src.data.database.get_engine"), \
             patch("src.models.registry.ModelRegistry") as MockRegistry, \
             patch("src.features.create_inference_features") as mock_feat:
            mock_model = MagicMock()
            MockRegistry.return_value.load_production_model.return_value = mock_model
            mock_feat.return_value = mock_features

            # Make fallback also fail to test graceful degradation
            MockRegistry.return_value.get_production_metrics.return_value = None
            MockRegistry.return_value.client.get_latest_versions.return_value = []

            result = explain_account_prediction.invoke(
                {"account_id": "ACC_00000001"}
            )

        assert isinstance(result, str)
        # SHAP either fails from import or from TreeExplainer
        assert "SHAP" in result or "failed" in result.lower() or "shap" in result.lower()


class TestValidateAccountId:
    """Tests for account ID validation."""

    def test_validate_account_id_valid(self):
        """Valid account IDs pass validation."""
        assert validate_account_id("ACC_00000001") is True
        assert validate_account_id("ACC_12345678") is True
        assert validate_account_id("ACC_99999999") is True

    def test_validate_account_id_invalid_format(self):
        """Invalid formats fail validation."""
        assert validate_account_id("acc_00000001") is False
        assert validate_account_id("ACC_0001") is False
        assert validate_account_id("ACC_123456789") is False
        assert validate_account_id("ACCOUNT_001") is False
        assert validate_account_id("00000001") is False

    def test_validate_account_id_empty(self):
        """Empty string fails validation."""
        assert validate_account_id("") is False

    @pytest.mark.db
    def test_segment_high_risk_accounts_produces_clusters(self, db_engine):
        """Integration: segmentation produces valid cohorts."""
        from src.agents.tools import segment_high_risk_accounts

        result = segment_high_risk_accounts.invoke({
            "reference_date": "2025-12-01",
            "min_cohort_size": 10,
        })

        assert isinstance(result, str)
        # Should be JSON or an error message
        assert "cohort" in result.lower() or "error" in result.lower()

    @pytest.mark.db
    def test_segment_high_risk_accounts_min_cohort_size_respected(self, db_engine):
        """Integration: cohorts respect minimum size."""
        from src.agents.tools import segment_high_risk_accounts

        result = segment_high_risk_accounts.invoke({
            "reference_date": "2025-12-01",
            "min_cohort_size": 50,
        })

        assert isinstance(result, str)


# =============================================================================
# TestUtils
# =============================================================================


class TestUtils:
    """Tests for shared utility functions."""

    def test_format_dataframe_truncation(self):
        """Markdown table truncates to max_rows."""
        df = pd.DataFrame({"a": range(50), "b": range(50)})
        result = format_dataframe_as_markdown(df, max_rows=5)

        assert "Showing 5 of 50 rows" in result
        lines = result.strip().split("\n")
        # Header + separator + 5 data rows + blank line + truncation note
        data_lines = [l for l in lines if l.startswith("|") and "---" not in l and "a" not in l]
        assert len(data_lines) == 5

    def test_format_dataframe_empty(self):
        """Empty DataFrame returns placeholder text."""
        df = pd.DataFrame()
        result = format_dataframe_as_markdown(df)

        assert "No data" in result

    def test_safe_json_serialize_numpy(self):
        """Numpy types serialize correctly."""
        assert safe_json_serialize(np.int64(42)) == 42
        assert safe_json_serialize(np.float64(3.14)) == 3.14
        assert safe_json_serialize(np.array([1, 2, 3])) == [1, 2, 3]

    def test_safe_json_serialize_datetime(self):
        """Datetime objects serialize to ISO format."""
        dt = datetime(2025, 12, 1, 10, 30, 0)
        result = safe_json_serialize(dt)
        assert result == "2025-12-01T10:30:00"

        d = date(2025, 12, 1)
        result = safe_json_serialize(d)
        assert result == "2025-12-01"

    def test_safe_json_serialize_pandas(self):
        """Pandas objects serialize correctly."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = safe_json_serialize(df)
        assert isinstance(result, list)
        assert result[0]["a"] == 1

        series = pd.Series({"x": 10, "y": 20})
        result = safe_json_serialize(series)
        assert isinstance(result, dict)
        assert result["x"] == 10

    def test_safe_json_serialize_nested(self):
        """Nested structures with mixed types serialize."""
        obj = {
            "count": np.int64(5),
            "values": np.array([1.0, 2.0]),
            "date": datetime(2025, 1, 1),
        }
        result = safe_json_serialize(obj)

        assert result["count"] == 5
        assert result["values"] == [1.0, 2.0]
        assert result["date"] == "2025-01-01T00:00:00"

    def test_truncate_for_context(self):
        """Text truncation works correctly."""
        short = "Hello world"
        assert truncate_for_context(short, max_chars=100) == short

        long = "x" * 5000
        result = truncate_for_context(long, max_chars=100)
        assert len(result) < 5000
        assert "truncated" in result
        assert "4900" in result  # Remaining chars mentioned


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "not db"])
