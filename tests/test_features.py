"""
Tests for feature engineering pipeline.

These tests validate the feature transformers and pipeline logic
using mock data to ensure correctness without database dependencies.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from src.features.config import (
    FeatureConfig,
    FeatureSpec,
    FeatureType,
    FEATURE_SPECS,
    TIME_WINDOWS,
    get_features_by_type,
    get_required_tables,
)
from src.features.validation import (
    FeatureValidator,
    ValidationResult,
    validate_feature_schema,
)
from src.features.transformers import TransformerContext
from src.features.pipeline import FeaturePipeline, PipelineResult
from src.features.export import build_analytics_dataframe, export_analytics_csv


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create sample feature DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        "account_id": [f"ACC_{i:08d}" for i in range(n_samples)],
        "avg_watch_hours_per_week_30d": np.random.exponential(5, n_samples),
        "days_since_last_stream": np.random.randint(0, 60, n_samples),
        "total_watch_sessions_30d": np.random.poisson(10, n_samples),
        "genre_diversity_score": np.random.uniform(0, 1, n_samples),
        "failed_payment_count_30d": np.random.poisson(0.3, n_samples),
        "ticket_count_30d": np.random.poisson(0.5, n_samples),
        "account_tenure_days": np.random.randint(30, 730, n_samples),
        "age": np.random.randint(18, 70, n_samples),
        "plan_tier_encoded": np.random.choice([1, 2, 3], n_samples),
    }).set_index("account_id")


@pytest.fixture
def sample_target() -> pd.DataFrame:
    """Create sample target DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        "account_id": [f"ACC_{i:08d}" for i in range(n_samples)],
        "churned": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }).set_index("account_id")


@pytest.fixture
def feature_config() -> FeatureConfig:
    """Create default feature configuration."""
    return FeatureConfig(
        reference_date="2025-12-01",
        rolling_windows=["30d", "90d"],
        min_history_days=14,
        validate_output=True,
    )


# =============================================================================
# Config Tests
# =============================================================================

class TestFeatureConfig:
    """Tests for feature configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()
        
        assert config.reference_date is None
        assert "30d" in config.rolling_windows
        assert "90d" in config.rolling_windows
        assert config.min_history_days == 14
        assert config.output_format == "parquet"
        assert config.validate_output is True
    
    def test_time_windows(self):
        """Test time window definitions."""
        assert "7d" in TIME_WINDOWS
        assert "30d" in TIME_WINDOWS
        assert "90d" in TIME_WINDOWS
        
        assert TIME_WINDOWS["30d"].days == 30
        assert TIME_WINDOWS["90d"].timedelta == timedelta(days=90)
    
    def test_feature_specs_complete(self):
        """Test that all feature specs have required fields."""
        for name, spec in FEATURE_SPECS.items():
            assert spec.name == name
            assert isinstance(spec.feature_type, FeatureType)
            assert spec.description
            assert len(spec.source_tables) > 0
    
    def test_get_features_by_type(self):
        """Test filtering features by type."""
        engagement_features = get_features_by_type(FeatureType.ENGAGEMENT)
        
        assert len(engagement_features) > 0
        assert all(
            f.feature_type == FeatureType.ENGAGEMENT 
            for f in engagement_features
        )
    
    def test_get_required_tables(self):
        """Test determining required tables for features."""
        # All features need streaming_events
        tables = get_required_tables()
        assert "streaming_events" in tables
        assert "accounts" in tables
        
        # Specific features need specific tables
        tables = get_required_tables(["avg_monthly_spend"])
        assert "payments" in tables


# =============================================================================
# Validation Tests
# =============================================================================

class TestFeatureValidator:
    """Tests for feature validation."""
    
    def test_valid_features_pass(self, sample_features, sample_target):
        """Test that valid features pass validation."""
        validator = FeatureValidator()
        result = validator.validate(sample_features, sample_target)
        
        assert result.passed
        assert len(result.issues) == 0
    
    def test_missing_values_detected(self, sample_features, sample_target):
        """Test detection of excessive missing values."""
        # Add column with many missing values
        sample_features["bad_column"] = np.nan
        
        validator = FeatureValidator(max_missing_ratio=0.1)
        result = validator.validate(sample_features, sample_target)
        
        assert any("missing" in issue.lower() for issue in result.issues)
    
    def test_infinite_values_detected(self, sample_features, sample_target):
        """Test detection of infinite values."""
        sample_features.loc[sample_features.index[0], "age"] = np.inf
        
        validator = FeatureValidator()
        result = validator.validate(sample_features, sample_target)
        
        assert any("infinite" in issue.lower() for issue in result.issues)
    
    def test_constant_features_warned(self, sample_features, sample_target):
        """Test warning for constant features."""
        sample_features["constant_col"] = 1
        
        validator = FeatureValidator()
        result = validator.validate(sample_features, sample_target)
        
        assert any("constant" in w.lower() for w in result.warnings)
    
    def test_range_violations_warned(self, sample_features, sample_target):
        """Test warning for values outside expected ranges."""
        sample_features.loc[sample_features.index[0], "age"] = 150
        
        validator = FeatureValidator()
        result = validator.validate(sample_features, sample_target)
        
        assert any("age" in w.lower() for w in result.warnings)
    
    def test_target_leakage_detected(self, sample_features, sample_target):
        """Test detection of target leakage."""
        # Create feature perfectly correlated with target
        sample_features["leaky_feature"] = sample_target["churned"].values
        
        validator = FeatureValidator()
        result = validator.validate(sample_features, sample_target)
        
        assert any("leakage" in issue.lower() for issue in result.issues)
    
    def test_minimum_samples_check(self, sample_target):
        """Test minimum sample size validation."""
        small_features = pd.DataFrame({
            "feature1": [1, 2, 3],
        }, index=[f"ACC_{i}" for i in range(3)])
        small_target = sample_target.iloc[:3]
        
        validator = FeatureValidator(min_samples=10)
        result = validator.validate(small_features, small_target)
        
        assert not result.passed
        assert any("insufficient" in issue.lower() for issue in result.issues)


class TestSchemaValidation:
    """Tests for schema validation."""
    
    def test_valid_schema_passes(self, sample_features):
        """Test that matching schema passes."""
        expected = list(sample_features.columns)
        result = validate_feature_schema(sample_features, expected)
        
        assert result.passed
    
    def test_missing_columns_fail(self, sample_features):
        """Test that missing columns are detected."""
        expected = list(sample_features.columns) + ["extra_required"]
        result = validate_feature_schema(sample_features, expected)
        
        assert not result.passed
        assert any("missing" in issue.lower() for issue in result.issues)
    
    def test_extra_columns_warned(self, sample_features):
        """Test that extra columns generate warnings."""
        expected = list(sample_features.columns)[:-2]  # Remove 2 columns
        result = validate_feature_schema(sample_features, expected, strict=False)
        
        assert result.passed  # Non-strict mode
        assert len(result.warnings) > 0
    
    def test_strict_mode_fails_on_extra(self, sample_features):
        """Test that strict mode fails on extra columns."""
        expected = list(sample_features.columns)[:-2]
        result = validate_feature_schema(sample_features, expected, strict=True)
        
        assert not result.passed


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipelineResult:
    """Tests for pipeline result structure."""
    
    def test_result_structure(self, sample_features, sample_target):
        """Test PipelineResult has expected structure."""
        result = PipelineResult(
            features=sample_features,
            target=sample_target,
            validation=None,
            metadata={"n_accounts": 100},
            success=True,
        )
        
        assert result.success
        assert len(result.features) == 100
        assert "churned" in result.target.columns
        assert result.metadata["n_accounts"] == 100
    
    def test_result_with_errors(self, sample_features, sample_target):
        """Test PipelineResult with errors."""
        result = PipelineResult(
            features=sample_features,
            target=sample_target,
            validation=None,
            metadata={},
            success=False,
            errors=["Transformer failed", "Connection lost"],
        )
        
        assert not result.success
        assert len(result.errors) == 2


class TestTransformerContext:
    """Tests for transformer context."""
    
    def test_context_creation(self):
        """Test TransformerContext initialization."""
        mock_engine = MagicMock()
        ref_date = date(2025, 12, 1)
        windows = [TIME_WINDOWS["30d"], TIME_WINDOWS["90d"]]
        
        context = TransformerContext(
            engine=mock_engine,
            reference_date=ref_date,
            time_windows=windows,
            account_ids=["ACC_001", "ACC_002"],
        )
        
        assert context.reference_date == ref_date
        assert len(context.time_windows) == 2
        assert len(context.account_ids) == 2
    
    def test_context_without_account_filter(self):
        """Test context for all accounts."""
        mock_engine = MagicMock()
        context = TransformerContext(
            engine=mock_engine,
            reference_date=date.today(),
            time_windows=[TIME_WINDOWS["30d"]],
        )
        
        assert context.account_ids is None


# =============================================================================
# Integration-style Tests (mocked database)
# =============================================================================

class TestFeaturePipelineIntegration:
    """Integration tests for the full pipeline with mocked database."""
    
    @patch("src.features.pipeline.FeaturePipeline._get_base_accounts")
    def test_pipeline_fills_defaults(self, mock_base_accounts, feature_config):
        """Test that pipeline fills missing values with defaults."""
        # Create mock engine
        mock_engine = MagicMock()
        
        # Create sample base accounts
        base_accounts = pd.DataFrame(
            index=[f"ACC_{i:08d}" for i in range(10)]
        )
        mock_base_accounts.return_value = base_accounts
        
        # Test the default filling logic directly
        pipeline = FeaturePipeline(feature_config, mock_engine, transformers=[])
        
        # Create a dataframe with missing values
        df = pd.DataFrame({
            "ticket_count_30d": [1, 2, None, 4, None],
            "has_billing_complaint": [1, None, 0, None, 1],
            "days_since_last_stream": [5, None, 10, None, 15],
        })
        
        filled = pipeline._fill_defaults(df)
        
        # Count columns should be 0
        assert filled["ticket_count_30d"].isna().sum() == 0
        assert filled.loc[2, "ticket_count_30d"] == 0
        
        # Boolean columns should be 0
        assert filled["has_billing_complaint"].isna().sum() == 0
        
        # Days since columns should be 9999
        assert filled.loc[1, "days_since_last_stream"] == 9999


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        validator = FeatureValidator(min_samples=0)
        result = validator.validate(empty_df, None)
        
        # Should handle gracefully
        assert result.statistics["n_samples"] == 0
    
    def test_single_row(self, sample_target):
        """Test validation with single row."""
        single_row = pd.DataFrame({
            "feature1": [1.0],
        }, index=["ACC_001"])
        
        validator = FeatureValidator(min_samples=1)
        result = validator.validate(single_row, sample_target.iloc[:1])
        
        # Should pass with min_samples=1
        assert result.passed or "constant" in str(result.warnings)
    
    def test_all_missing_column(self, sample_features, sample_target):
        """Test handling of column with all missing values."""
        sample_features["all_missing"] = np.nan
        
        validator = FeatureValidator()
        result = validator.validate(sample_features, sample_target)
        
        assert any("all_missing" in issue for issue in result.issues)
    
    def test_mixed_dtypes(self, sample_target):
        """Test handling of mixed data types."""
        mixed_df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "string": ["a", "b", "c"],
            "boolean": [True, False, True],
        }, index=["ACC_001", "ACC_002", "ACC_003"])
        
        validator = FeatureValidator(min_samples=1)
        result = validator.validate(mixed_df, sample_target.iloc[:3])
        
        # Should handle all types
        assert "numeric" in result.statistics["feature_summary"]
        assert "string" in result.statistics["feature_summary"]
        assert "boolean" in result.statistics["feature_summary"]
        # Boolean columns get special treatment
        assert "true_count" in result.statistics["feature_summary"]["boolean"]


# =============================================================================
# Export Tests
# =============================================================================

class TestBuildAnalyticsDataframe:
    """Tests for the analytics export builder."""
    
    def test_combines_metadata_target_features(self, sample_features, sample_target):
        """Test that all three sources are merged correctly."""
        # Mock the metadata fetch
        metadata = pd.DataFrame({
            "email": [f"user{i}@test.com" for i in range(len(sample_features))],
            "signup_date": pd.Timestamp("2024-06-15"),
            "country": "US",
            "plan_type": "Premium",
            "subscription_status": "active",
        }, index=sample_features.index)
        metadata.index.name = "account_id"
        
        mock_engine = MagicMock()
        result = PipelineResult(
            features=sample_features,
            target=sample_target,
            validation=None,
            metadata={"n_accounts": len(sample_features)},
            success=True,
        )
        
        # Patch the metadata query
        with patch("src.features.export._fetch_account_metadata", return_value=metadata):
            analytics = build_analytics_dataframe(mock_engine, result)
        
        # Should contain columns from all three sources
        assert "email" in analytics.columns           # metadata
        assert "churned" in analytics.columns          # target
        assert "age" in analytics.columns              # features
        
        # Row count should match metadata (left join base)
        assert len(analytics) == len(metadata)
    
    def test_handles_empty_target(self, sample_features):
        """Test export works when target is not included."""
        metadata = pd.DataFrame({
            "email": [f"user{i}@test.com" for i in range(len(sample_features))],
        }, index=sample_features.index)
        metadata.index.name = "account_id"
        
        mock_engine = MagicMock()
        result = PipelineResult(
            features=sample_features,
            target=pd.DataFrame(),
            validation=None,
            metadata={},
            success=True,
        )
        
        with patch("src.features.export._fetch_account_metadata", return_value=metadata):
            analytics = build_analytics_dataframe(mock_engine, result)
        
        assert "email" in analytics.columns
        assert "churned" not in analytics.columns
        assert len(analytics) == len(sample_features)


class TestExportAnalyticsCsv:
    """Tests for the CSV export function."""
    
    def test_creates_stable_file(self, tmp_path, sample_features, sample_target):
        """Test that a stable retain_analytics.csv is created."""
        metadata = pd.DataFrame({
            "email": [f"user{i}@test.com" for i in range(len(sample_features))],
        }, index=sample_features.index)
        metadata.index.name = "account_id"
        
        mock_engine = MagicMock()
        result = PipelineResult(
            features=sample_features,
            target=sample_target,
            validation=None,
            metadata={},
            success=True,
        )
        
        with patch("src.features.export._fetch_account_metadata", return_value=metadata):
            csv_path = export_analytics_csv(
                mock_engine, result, output_dir=tmp_path
            )
        
        assert csv_path.exists()
        assert csv_path.name == "retain_analytics.csv"
        
        # Verify it's a valid CSV we can read back
        df = pd.read_csv(csv_path, index_col="account_id")
        assert len(df) == len(sample_features)
    
    def test_creates_snapshot(self, tmp_path, sample_features, sample_target):
        """Test that a timestamped snapshot is also created."""
        metadata = pd.DataFrame({
            "email": [f"user{i}@test.com" for i in range(len(sample_features))],
        }, index=sample_features.index)
        metadata.index.name = "account_id"
        
        mock_engine = MagicMock()
        result = PipelineResult(
            features=sample_features,
            target=sample_target,
            validation=None,
            metadata={},
            success=True,
        )
        
        with patch("src.features.export._fetch_account_metadata", return_value=metadata):
            export_analytics_csv(mock_engine, result, output_dir=tmp_path)
        
        snapshot_dir = tmp_path / "snapshots"
        assert snapshot_dir.exists()
        
        snapshots = list(snapshot_dir.glob("retain_analytics_*.csv"))
        assert len(snapshots) == 1
    
    def test_overwrites_stable_file(self, tmp_path, sample_features, sample_target):
        """Test that re-running overwrites the stable file."""
        import time
        
        metadata = pd.DataFrame({
            "email": [f"user{i}@test.com" for i in range(len(sample_features))],
        }, index=sample_features.index)
        metadata.index.name = "account_id"
        
        mock_engine = MagicMock()
        result = PipelineResult(
            features=sample_features,
            target=sample_target,
            validation=None,
            metadata={},
            success=True,
        )
        
        with patch("src.features.export._fetch_account_metadata", return_value=metadata):
            path1 = export_analytics_csv(mock_engine, result, output_dir=tmp_path)
            time.sleep(1.1)  # Ensure different timestamp for snapshot filename
            path2 = export_analytics_csv(mock_engine, result, output_dir=tmp_path)
        
        # Same stable path both times
        assert path1 == path2
        
        # But two snapshots
        snapshots = list((tmp_path / "snapshots").glob("retain_analytics_*.csv"))
        assert len(snapshots) == 2
    
    def test_skip_snapshot(self, tmp_path, sample_features, sample_target):
        """Test that snapshots can be disabled."""
        metadata = pd.DataFrame({
            "email": [f"user{i}@test.com" for i in range(len(sample_features))],
        }, index=sample_features.index)
        metadata.index.name = "account_id"
        
        mock_engine = MagicMock()
        result = PipelineResult(
            features=sample_features,
            target=sample_target,
            validation=None,
            metadata={},
            success=True,
        )
        
        with patch("src.features.export._fetch_account_metadata", return_value=metadata):
            export_analytics_csv(
                mock_engine, result, output_dir=tmp_path, save_snapshot=False
            )
        
        assert not (tmp_path / "snapshots").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
