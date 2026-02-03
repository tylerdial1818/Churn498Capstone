"""
Model training and deployment configuration.

This module defines configuration dataclasses for training, evaluation,
scoring, and monitoring. Configuration-driven approach enables easy
experimentation and reproducibility.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelType(Enum):
    """Supported model types for churn prediction."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"


class DriftMethod(Enum):
    """Drift detection methods for monitoring."""
    PSI = "psi"  # Population Stability Index
    KS = "ks"  # Kolmogorov-Smirnov test
    CHI_SQUARE = "chi_square"


@dataclass
class ModelConfig:
    """
    Configuration for model training.

    Hyperparameters are set with sensible defaults for churn prediction.
    Override these for experimentation, but defaults should work well.
    """
    # Model selection
    model_type: ModelType = ModelType.XGBOOST

    # Train/test split configuration
    test_size: float = 0.2  # 20% holdout for final evaluation
    validation_size: float = 0.2  # 20% of remaining for validation (60/20/20 split)
    random_seed: int = 42
    stratify: bool = True  # Stratified split on target

    # Cross-validation
    cv_folds: int = 5

    # XGBoost hyperparameters
    # These defaults are tuned for churn prediction with class imbalance
    xgboost_params: dict[str, Any] = field(default_factory=lambda: {
        "max_depth": 6,  # Moderate depth to prevent overfitting
        "learning_rate": 0.1,  # Standard learning rate
        "n_estimators": 100,  # Boosting rounds
        "min_child_weight": 3,  # Higher for imbalanced data
        "gamma": 0.1,  # Regularization to prevent overfitting
        "subsample": 0.8,  # Row sampling for generalization
        "colsample_bytree": 0.8,  # Feature sampling
        "scale_pos_weight": 3,  # Upweight minority class (churned users)
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",  # Fast histogram-based algorithm
        "random_state": 42,
    })

    # LightGBM hyperparameters
    lightgbm_params: dict[str, Any] = field(default_factory=lambda: {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3,
        "objective": "binary",
        "metric": "auc",
        "random_state": 42,
        "verbose": -1,
    })

    # Logistic regression hyperparameters
    logistic_params: dict[str, Any] = field(default_factory=lambda: {
        "penalty": "l2",
        "C": 1.0,  # Regularization strength (inverse)
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",  # Handle class imbalance
        "random_state": 42,
    })

    # Feature selection
    feature_selection: bool = False  # Set to True to enable feature selection
    max_features: int | None = None  # Limit features (None = use all)

    # Early stopping
    early_stopping_rounds: int = 10

    def get_params(self) -> dict[str, Any]:
        """Get hyperparameters for the configured model type."""
        match self.model_type:
            case ModelType.XGBOOST:
                return self.xgboost_params.copy()
            case ModelType.LIGHTGBM:
                return self.lightgbm_params.copy()
            case ModelType.LOGISTIC_REGRESSION:
                return self.logistic_params.copy()


@dataclass
class TrainingConfig:
    """
    Configuration for the training pipeline.

    Controls experiment tracking, evaluation metrics, and output behavior.
    """
    # MLflow experiment tracking
    experiment_name: str = "churn_prediction"
    run_name: str | None = None  # Auto-generated if None
    tracking_uri: str = "./mlruns"  # Local MLflow tracking directory

    # Evaluation metrics to compute and log
    # Primary metric is AUC-ROC for ranking performance
    metrics: list[str] = field(default_factory=lambda: [
        "auc",  # Area under ROC curve (primary)
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",  # Area under PR curve
        "log_loss",
    ])

    # SHAP explainability
    compute_shap: bool = True
    shap_sample_size: int = 100  # Sample size for SHAP computation (for speed)

    # Training behavior
    log_model: bool = True  # Log trained model to MLflow
    log_dataset: bool = False  # Log training data (can be large)

    # Baseline model (always trained for comparison)
    train_baseline: bool = True
    baseline_type: ModelType = ModelType.LOGISTIC_REGRESSION


@dataclass
class EvaluationConfig:
    """
    Configuration for the evaluation gate.

    Defines thresholds and checks that determine whether a candidate
    model should be promoted to production.
    """
    # Performance thresholds
    min_auc: float = 0.70  # Minimum acceptable AUC-ROC
    min_precision: float = 0.40  # Minimum precision
    min_recall: float = 0.50  # Minimum recall

    # Improvement thresholds
    min_improvement_vs_baseline: float = 0.02  # 2% AUC improvement over baseline
    min_improvement_vs_production: float = 0.00  # Must match or exceed production

    # Calibration check
    check_calibration: bool = True
    calibration_n_bins: int = 10
    max_calibration_error: float = 0.15  # Max mean absolute calibration error

    # Distribution checks
    check_prediction_distribution: bool = True
    max_distribution_shift: float = 0.20  # Max change in mean prediction

    # Feature importance stability
    check_feature_stability: bool = True
    top_n_features: int = 10  # Number of top features to check
    max_feature_rank_change: int = 5  # Max allowable rank change


@dataclass
class ScoringConfig:
    """
    Configuration for batch scoring pipeline.

    Controls how predictions are generated and stored.
    """
    # Scoring behavior
    batch_size: int = 1000  # Process in batches to manage memory

    # Output
    output_table: str = "predictions"
    create_table_if_missing: bool = True

    # Risk tier thresholds (probability → tier mapping)
    # High risk: likely to churn, needs immediate intervention
    # Medium risk: moderate churn probability, needs monitoring
    # Low risk: unlikely to churn
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40

    # Model selection
    use_production_model: bool = True  # If False, specify model_version
    model_version: int | None = None

    def classify_risk(self, probability: float) -> str:
        """Classify a churn probability into a risk tier."""
        if probability >= self.high_risk_threshold:
            return "high"
        elif probability >= self.medium_risk_threshold:
            return "medium"
        else:
            return "low"


@dataclass
class MonitoringConfig:
    """
    Configuration for model monitoring.

    Defines drift detection methods and alert thresholds.
    """
    # Drift detection
    drift_method: DriftMethod = DriftMethod.PSI

    # PSI thresholds (Population Stability Index)
    # PSI < 0.1: No significant shift
    # 0.1 <= PSI < 0.25: Moderate shift, investigate
    # PSI >= 0.25: Significant shift, action needed
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.25

    # Reference window for comparison
    reference_window_size: int = 30  # Days of reference data

    # Prediction drift thresholds
    prediction_mean_shift_threshold: float = 2.0  # Standard deviations

    # Performance monitoring (when labels available)
    check_performance_drift: bool = True
    min_samples_for_performance: int = 100
    performance_degradation_threshold: float = 0.05  # 5% AUC drop

    # Monitoring output
    log_table: str = "monitoring_log"
    create_table_if_missing: bool = True


# Default configurations for common use cases
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
DEFAULT_SCORING_CONFIG = ScoringConfig()
DEFAULT_MONITORING_CONFIG = MonitoringConfig()
