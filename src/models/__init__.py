"""
Model training, evaluation, and deployment for churn prediction.

This module provides production ML engineering capabilities including:
- Model training with MLflow tracking
- Automated evaluation gates for model promotion
- Model registry and versioning
- Batch scoring pipelines
- Model monitoring and drift detection

Example:
    >>> from src.models import ModelTrainer, ModelConfig, TrainingConfig
    >>> from src.features import create_training_dataset
    >>> from sqlalchemy import create_engine
    >>>
    >>> # Train a model
    >>> engine = create_engine("postgresql://...")
    >>> X, y = create_training_dataset(engine)
    >>> trainer = ModelTrainer(ModelConfig(), TrainingConfig())
    >>> result = trainer.train(X, y)
    >>>
    >>> # Register and promote
    >>> from src.models import ModelRegistry
    >>> registry = ModelRegistry()
    >>> registry.register_model(result)
    >>> registry.promote_model(version=1, stage="Production")
    >>>
    >>> # Score accounts
    >>> from src.models import score_active_accounts
    >>> scoring_result = score_active_accounts(engine)

For CLI usage:
    python -m src.models.build_models --help
"""

# Configuration
from .config import (
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_MONITORING_CONFIG,
    DEFAULT_SCORING_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DriftMethod,
    EvaluationConfig,
    ModelConfig,
    ModelType,
    MonitoringConfig,
    ScoringConfig,
    TrainingConfig,
)

# Training
from .train import ModelTrainer, TrainingResult

# Evaluation
from .evaluate import (
    CheckResult,
    CheckStatus,
    EvaluationGate,
    EvaluationResult,
    print_evaluation_report,
)

# Registry
from .registry import ModelMetadata, ModelRegistry

# Scoring
from .score import BatchScorer, ScoringResult, score_active_accounts

# Monitoring
from .monitoring import (
    DriftResult,
    HealthStatus,
    ModelMonitor,
    MonitoringReport,
    print_monitoring_report,
)


__all__ = [
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ScoringConfig",
    "MonitoringConfig",
    "ModelType",
    "DriftMethod",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_EVALUATION_CONFIG",
    "DEFAULT_SCORING_CONFIG",
    "DEFAULT_MONITORING_CONFIG",
    # Training
    "ModelTrainer",
    "TrainingResult",
    # Evaluation
    "EvaluationGate",
    "EvaluationResult",
    "CheckResult",
    "CheckStatus",
    "print_evaluation_report",
    # Registry
    "ModelRegistry",
    "ModelMetadata",
    # Scoring
    "BatchScorer",
    "ScoringResult",
    "score_active_accounts",
    # Monitoring
    "ModelMonitor",
    "MonitoringReport",
    "DriftResult",
    "HealthStatus",
    "print_monitoring_report",
]
