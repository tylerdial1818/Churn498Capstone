"""
Tests for the models module.

Run with: pytest tests/test_models.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models import (
    BatchScorer,
    CheckStatus,
    EvaluationConfig,
    EvaluationGate,
    HealthStatus,
    ModelConfig,
    ModelMonitor,
    ModelTrainer,
    ModelType,
    MonitoringConfig,
    ScoringConfig,
    TrainingConfig,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic binary classification data for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced like churn data
        random_state=42,
    )

    # Convert to DataFrame with feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="churned")

    return X_df, y_series


@pytest.fixture
def training_config():
    """Test training configuration."""
    return TrainingConfig(
        experiment_name="test_experiment",
        tracking_uri="./test_mlruns",
        compute_shap=False,  # Skip SHAP for faster tests
        train_baseline=True,
    )


@pytest.fixture
def model_config():
    """Test model configuration."""
    return ModelConfig(
        model_type=ModelType.LOGISTIC_REGRESSION,  # Fastest for testing
        random_seed=42,
        cv_folds=3,  # Fewer folds for speed
    )


# =============================================================================
# Configuration Tests
# =============================================================================


def test_model_config_defaults():
    """Test ModelConfig has sensible defaults."""
    config = ModelConfig()

    assert config.model_type == ModelType.XGBOOST
    assert config.test_size == 0.2
    assert config.random_seed == 42
    assert config.cv_folds == 5


def test_model_config_get_params():
    """Test getting hyperparameters for different model types."""
    # XGBoost
    config = ModelConfig(model_type=ModelType.XGBOOST)
    params = config.get_params()
    assert "max_depth" in params
    assert params["objective"] == "binary:logistic"

    # Logistic Regression
    config = ModelConfig(model_type=ModelType.LOGISTIC_REGRESSION)
    params = config.get_params()
    assert "penalty" in params
    assert params["class_weight"] == "balanced"


def test_scoring_config_classify_risk():
    """Test risk tier classification."""
    config = ScoringConfig()

    assert config.classify_risk(0.8) == "high"
    assert config.classify_risk(0.5) == "medium"
    assert config.classify_risk(0.2) == "low"


# =============================================================================
# Training Tests
# =============================================================================


def test_model_trainer_initialization(model_config, training_config):
    """Test ModelTrainer initializes correctly."""
    trainer = ModelTrainer(model_config, training_config)

    assert trainer.model_config == model_config
    assert trainer.training_config == training_config


def test_model_training_end_to_end(synthetic_data, model_config, training_config):
    """Test complete training pipeline with synthetic data."""
    X, y = synthetic_data

    trainer = ModelTrainer(model_config, training_config)
    result = trainer.train(X, y)

    # Check result structure
    assert result.success is True
    assert len(result.errors) == 0
    assert result.model is not None
    assert result.baseline_model is not None  # Because train_baseline=True

    # Check metrics
    assert "auc" in result.metrics
    assert 0 <= result.metrics["auc"] <= 1
    assert "precision" in result.metrics
    assert "recall" in result.metrics

    # Check baseline comparison
    assert "auc" in result.baseline_metrics
    assert result.improvement_over_baseline is not None

    # Check feature importances
    assert len(result.feature_importances) == X.shape[1]
    assert "feature" in result.feature_importances.columns
    assert "importance" in result.feature_importances.columns

    # Check CV metrics
    assert "auc" in result.cv_metrics
    assert len(result.cv_metrics["auc"]) == model_config.cv_folds

    # Check metadata
    assert result.n_train_samples > 0
    assert result.n_val_samples > 0
    assert result.n_test_samples > 0


def test_training_with_xgboost(synthetic_data, training_config):
    """Test training with XGBoost model."""
    X, y = synthetic_data

    config = ModelConfig(model_type=ModelType.XGBOOST, cv_folds=2)
    trainer = ModelTrainer(config, training_config)
    result = trainer.train(X, y)

    assert result.success is True
    assert result.model is not None
    assert result.metrics["auc"] > 0.5  # Better than random


# =============================================================================
# Evaluation Tests
# =============================================================================


def test_evaluation_gate_initialization():
    """Test EvaluationGate initializes correctly."""
    config = EvaluationConfig()
    gate = EvaluationGate(config)

    assert gate.config == config


def test_evaluation_gate_performance_threshold(synthetic_data, model_config, training_config):
    """Test evaluation gate passes/fails based on thresholds."""
    X, y = synthetic_data

    # Train a model
    trainer = ModelTrainer(model_config, training_config)
    result = trainer.train(X, y)

    # Split for test set
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate with lenient thresholds (should pass)
    config = EvaluationConfig(min_auc=0.5, min_precision=0.2, min_recall=0.2)
    gate = EvaluationGate(config)
    eval_result = gate.evaluate(result, X_test, y_test)

    assert eval_result.passed is True
    assert eval_result.recommendation in ["promote", "review"]
    assert len(eval_result.checks) > 0

    # Check that we have the right checks
    check_names = [c.name for c in eval_result.checks]
    assert "performance_threshold" in check_names
    assert "improvement_over_baseline" in check_names


def test_evaluation_gate_fails_low_performance(synthetic_data, model_config, training_config):
    """Test evaluation gate rejects models with poor performance."""
    X, y = synthetic_data

    # Train a model
    trainer = ModelTrainer(model_config, training_config)
    result = trainer.train(X, y)

    # Split for test set
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate with impossibly high thresholds (should fail)
    config = EvaluationConfig(min_auc=0.99, min_precision=0.99, min_recall=0.99)
    gate = EvaluationGate(config)
    eval_result = gate.evaluate(result, X_test, y_test)

    assert eval_result.passed is False
    assert eval_result.recommendation == "reject"

    # Check for failures
    failed_checks = eval_result.get_failed_checks()
    assert len(failed_checks) > 0


# =============================================================================
# Scoring Tests
# =============================================================================


def test_batch_scorer_initialization(model_config, training_config, synthetic_data):
    """Test BatchScorer initializes correctly."""
    # Need to create a mock engine - skip for now, test logic instead
    pass


def test_scoring_result_structure(synthetic_data):
    """Test ScoringResult has correct structure."""
    from src.models.score import ScoringResult

    X, _ = synthetic_data

    # Create mock predictions
    predictions_df = pd.DataFrame({
        "account_id": [f"acc_{i}" for i in range(100)],
        "churn_probability": np.random.rand(100),
        "risk_tier": np.random.choice(["low", "medium", "high"], 100),
        "model_version": [1] * 100,
        "scored_at": [pd.Timestamp.now()] * 100,
    })

    result = ScoringResult(
        n_accounts_scored=100,
        predictions=predictions_df,
        risk_distribution={"low": 40, "medium": 40, "high": 20},
        model_version=1,
        model_name="test_model",
    )

    assert result.n_accounts_scored == 100
    assert len(result.predictions) == 100

    # Test high risk filtering
    high_risk = result.get_high_risk_accounts()
    assert all(high_risk["risk_tier"] == "high")

    # Test summary
    summary = result.get_summary()
    assert "n_accounts" in summary
    assert "mean_score" in summary


# =============================================================================
# Monitoring Tests
# =============================================================================


def test_model_monitor_initialization():
    """Test ModelMonitor initializes correctly."""
    from src.models.monitoring import ModelMonitor
    from unittest.mock import Mock

    mock_engine = Mock()
    config = MonitoringConfig()

    monitor = ModelMonitor(mock_engine, config)
    assert monitor.config == config


def test_drift_detection_no_drift(synthetic_data):
    """Test drift detection with identical distributions."""
    X, _ = synthetic_data

    # Create reference and current data from same distribution
    reference = X.iloc[:500]
    current = X.iloc[500:]

    from src.models.monitoring import ModelMonitor
    from unittest.mock import Mock

    mock_engine = Mock()
    config = MonitoringConfig()
    monitor = ModelMonitor(mock_engine, config)

    report = monitor.check_drift(current, reference)

    # Should be healthy since data is from same distribution
    assert report.health_status == HealthStatus.HEALTHY
    assert len(report.feature_drift) > 0


def test_drift_detection_with_drift(synthetic_data):
    """Test drift detection with shifted distributions."""
    X, _ = synthetic_data

    # Reference data
    reference = X.iloc[:500].copy()

    # Current data with artificial drift (shift all features)
    current = X.iloc[500:].copy()
    current = current + 2.0  # Add constant to create drift

    from src.models.monitoring import ModelMonitor
    from unittest.mock import Mock

    mock_engine = Mock()
    config = MonitoringConfig(
        psi_warning_threshold=0.05,  # Lower threshold to catch drift
        psi_critical_threshold=0.15,
    )
    monitor = ModelMonitor(mock_engine, config)

    report = monitor.check_drift(current, reference)

    # Should detect drift
    drifted = report.get_drifted_features()
    assert len(drifted) > 0

    # Health should not be healthy
    assert report.health_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]


def test_psi_calculation():
    """Test PSI calculation for feature drift."""
    from src.models.monitoring import ModelMonitor
    from unittest.mock import Mock

    mock_engine = Mock()
    config = MonitoringConfig()
    monitor = ModelMonitor(mock_engine, config)

    # Create two different distributions
    reference = pd.Series(np.random.normal(0, 1, 1000))
    current = pd.Series(np.random.normal(0.5, 1, 1000))  # Shifted mean

    psi = monitor._calculate_psi(reference, current)

    # PSI should be positive (there is a shift)
    assert psi > 0

    # PSI should be reasonable (not infinite)
    assert psi < 10


def test_monitoring_report_summary():
    """Test MonitoringReport summary generation."""
    from src.models.monitoring import DriftResult, MonitoringReport

    drift_results = [
        DriftResult(
            feature_name="feature_1",
            drift_score=0.3,
            status=HealthStatus.CRITICAL,
            reference_mean=0.0,
            current_mean=1.0,
            reference_std=1.0,
            current_std=1.0,
        ),
        DriftResult(
            feature_name="feature_2",
            drift_score=0.15,
            status=HealthStatus.WARNING,
            reference_mean=0.0,
            current_mean=0.5,
            reference_std=1.0,
            current_std=1.0,
        ),
    ]

    report = MonitoringReport(
        health_status=HealthStatus.CRITICAL,
        feature_drift=drift_results,
        recommended_actions=["Retrain model"],
    )

    # Test filtering
    critical = report.get_drifted_features(HealthStatus.CRITICAL)
    assert len(critical) == 1
    assert critical[0].feature_name == "feature_1"

    warnings = report.get_drifted_features(HealthStatus.WARNING)
    assert len(warnings) == 1

    # Test summary
    summary = report.summary()
    assert "CRITICAL" in summary
    assert "Retrain model" in summary


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_training_evaluation_flow(synthetic_data, model_config, training_config):
    """Test full flow: train -> evaluate -> promote decision."""
    X, y = synthetic_data

    # 1. Train model
    trainer = ModelTrainer(model_config, training_config)
    training_result = trainer.train(X, y)

    assert training_result.success

    # 2. Evaluate model
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    eval_config = EvaluationConfig(min_auc=0.6, min_precision=0.3, min_recall=0.3)
    gate = EvaluationGate(eval_config)
    eval_result = gate.evaluate(training_result, X_test, y_test)

    # 3. Make decision
    assert eval_result.recommendation in ["promote", "review", "reject"]

    # If it passes, it should have passed checks
    if eval_result.passed:
        assert eval_result.recommendation in ["promote", "review"]
        failed = eval_result.get_failed_checks()
        assert len(failed) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
