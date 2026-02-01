"""
Feature engineering module for Retain churn prediction.

This module provides a production-grade feature engineering pipeline
for transforming raw customer data into ML-ready features.

Example:
    >>> from src.features import FeaturePipeline, FeatureConfig
    >>> from sqlalchemy import create_engine
    >>>
    >>> engine = create_engine("postgresql://...")
    >>> config = FeatureConfig(reference_date="2025-12-01")
    >>> pipeline = FeaturePipeline(config, engine)
    >>> result = pipeline.run()
    >>> X = result.features
    >>> y = result.target["churned"]

For CLI usage:
    python -m src.features.build_features --help
"""

from .config import (
    FeatureConfig,
    FeatureSpec,
    FeatureType,
    FEATURE_SPECS,
    TIME_WINDOWS,
    get_features_by_type,
    get_required_tables,
)

from .pipeline import (
    FeaturePipeline,
    PipelineResult,
    create_training_dataset,
    create_inference_features,
)

from .validation import (
    FeatureValidator,
    ValidationResult,
    validate_feature_schema,
)

from .export import (
    export_analytics_csv,
    build_analytics_dataframe,
)

from .transformers import (
    BaseTransformer,
    TransformerContext,
    TRANSFORMER_REGISTRY,
)


__all__ = [
    # Config
    "FeatureConfig",
    "FeatureSpec",
    "FeatureType",
    "FEATURE_SPECS",
    "TIME_WINDOWS",
    "get_features_by_type",
    "get_required_tables",
    # Pipeline
    "FeaturePipeline",
    "PipelineResult",
    "create_training_dataset",
    "create_inference_features",
    # Validation
    "FeatureValidator",
    "ValidationResult",
    "validate_feature_schema",
    # Export
    "export_analytics_csv",
    "build_analytics_dataframe",
    # Transformers
    "BaseTransformer",
    "TransformerContext",
    "TRANSFORMER_REGISTRY",
]
