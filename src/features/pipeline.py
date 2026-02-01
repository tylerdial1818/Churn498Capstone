"""
Feature engineering pipeline orchestrator.

This module coordinates feature generation across all transformers,
handles data validation, and manages output persistence.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import TIME_WINDOWS, FeatureConfig, TimeWindow
from .transformers import (
    TRANSFORMER_REGISTRY,
    BaseTransformer,
    TransformerContext,
)
from .validation import FeatureValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    features: pd.DataFrame
    target: pd.DataFrame
    validation: ValidationResult | None
    metadata: dict[str, Any]
    success: bool
    errors: list[str] = field(default_factory=list)


class FeaturePipeline:
    """
    Orchestrates the feature engineering pipeline.
    
    This pipeline:
    1. Loads raw data from PostgreSQL
    2. Executes feature transformers
    3. Validates output features
    4. Persists results to configured destination
    
    Example:
        >>> config = FeatureConfig(reference_date="2025-12-01")
        >>> pipeline = FeaturePipeline(config, engine)
        >>> result = pipeline.run()
        >>> result.features.to_parquet("features.parquet")
    """
    
    def __init__(
        self,
        config: FeatureConfig,
        engine: Engine,
        transformers: list[str] | None = None,
    ):
        """
        Initialize the feature pipeline.
        
        Args:
            config: Pipeline configuration
            engine: SQLAlchemy database engine
            transformers: List of transformer names to run (None = all)
        """
        self.config = config
        self.engine = engine
        self.transformer_names = transformers or list(TRANSFORMER_REGISTRY.keys())
        self.validator = FeatureValidator()
        
        # Parse reference date
        if config.reference_date:
            self.reference_date = datetime.strptime(
                config.reference_date, "%Y-%m-%d"
            ).date()
        else:
            self.reference_date = date.today()
        
        # Parse time windows
        self.time_windows = [
            TIME_WINDOWS[w] for w in config.rolling_windows
            if w in TIME_WINDOWS
        ]
        
        logger.info(
            f"Initialized pipeline with reference_date={self.reference_date}, "
            f"windows={[w.name for w in self.time_windows]}"
        )
    
    def run(
        self,
        account_ids: list[str] | None = None,
        include_target: bool = True,
    ) -> PipelineResult:
        """
        Execute the full feature engineering pipeline.
        
        Args:
            account_ids: Optional list of account IDs to process (None = all)
            include_target: Whether to include target variable
        
        Returns:
            PipelineResult containing features, target, and metadata
        """
        start_time = datetime.now()
        errors: list[str] = []
        
        logger.info("Starting feature pipeline execution")
        
        # Create transformer context
        context = TransformerContext(
            engine=self.engine,
            reference_date=self.reference_date,
            time_windows=self.time_windows,
            account_ids=account_ids,
        )
        
        # Execute feature transformers
        feature_dfs: list[pd.DataFrame] = []
        target_df: pd.DataFrame | None = None
        
        for name in self.transformer_names:
            if name == "target" and not include_target:
                continue
                
            if name not in TRANSFORMER_REGISTRY:
                logger.warning(f"Unknown transformer: {name}, skipping")
                continue
            
            try:
                logger.info(f"Running transformer: {name}")
                transformer_class = TRANSFORMER_REGISTRY[name]
                transformer = transformer_class(context)
                result_df = transformer.transform()
                
                if name == "target":
                    target_df = result_df
                else:
                    feature_dfs.append(result_df)
                    
                logger.info(
                    f"Transformer {name} completed: "
                    f"{len(result_df)} rows, {len(result_df.columns)} columns"
                )
                
            except Exception as e:
                error_msg = f"Transformer {name} failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                if self.config.fail_on_validation_error:
                    raise
        
        # Merge all feature dataframes
        if not feature_dfs:
            return PipelineResult(
                features=pd.DataFrame(),
                target=pd.DataFrame(),
                validation=None,
                metadata={"error": "No features generated"},
                success=False,
                errors=errors,
            )
        
        logger.info("Merging feature sets")
        features = self._merge_features(feature_dfs)
        
        # Get base account list and align
        base_accounts = self._get_base_accounts(context)
        features = base_accounts.join(features, how="left")
        
        if target_df is not None:
            target_df = base_accounts.join(target_df, how="left")
        else:
            target_df = pd.DataFrame(index=base_accounts.index)
        
        # Fill missing values with defaults
        features = self._fill_defaults(features)
        
        # Validate features
        validation_result = None
        if self.config.validate_output:
            logger.info("Validating features")
            validation_result = self.validator.validate(features, target_df)
            
            if not validation_result.passed:
                for issue in validation_result.issues:
                    logger.warning(f"Validation issue: {issue}")
                if self.config.fail_on_validation_error:
                    errors.extend(validation_result.issues)
        
        # Build metadata
        end_time = datetime.now()
        metadata = {
            "reference_date": str(self.reference_date),
            "time_windows": [w.name for w in self.time_windows],
            "n_accounts": len(features),
            "n_features": len(features.columns),
            "execution_time_seconds": (end_time - start_time).total_seconds(),
            "transformers_run": self.transformer_names,
            "generated_at": end_time.isoformat(),
        }
        
        logger.info(
            f"Pipeline completed: {metadata['n_accounts']} accounts, "
            f"{metadata['n_features']} features, "
            f"{metadata['execution_time_seconds']:.1f}s"
        )
        
        return PipelineResult(
            features=features,
            target=target_df,
            validation=validation_result,
            metadata=metadata,
            success=len(errors) == 0,
            errors=errors,
        )
    
    def _merge_features(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple feature DataFrames on index (account_id)."""
        if len(dfs) == 0:
            return pd.DataFrame()
        
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how="outer")
        
        return result
    
    def _get_base_accounts(self, context: TransformerContext) -> pd.DataFrame:
        """Get base account list meeting minimum history requirement."""
        query = """
        SELECT DISTINCT a.account_id
        FROM accounts a
        JOIN subscriptions s ON a.account_id = s.account_id
        WHERE a.signup_date <= :cutoff_date
        """
        
        if context.account_ids:
            ids = ", ".join(f"'{aid}'" for aid in context.account_ids)
            query += f" AND a.account_id IN ({ids})"
        
        cutoff_date = self.reference_date - pd.Timedelta(
            days=self.config.min_history_days
        )
        
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query), 
                conn, 
                params={"cutoff_date": cutoff_date}
            )
        
        return df.set_index("account_id")
    
    def _fill_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults."""
        # Numeric columns: fill with 0 for counts, NaN for rates/averages
        count_cols = [c for c in df.columns if "count" in c.lower()]
        for col in count_cols:
            df[col] = df[col].fillna(0)
        
        # Boolean columns: fill with 0
        bool_cols = [c for c in df.columns if c.startswith("has_") or c.startswith("is_")]
        for col in bool_cols:
            df[col] = df[col].fillna(0).astype(int)
        
        # Days since columns: fill with large value (no activity)
        days_cols = [c for c in df.columns if "days_since" in c.lower()]
        for col in days_cols:
            df[col] = df[col].fillna(9999)
        
        return df
    
    def save(
        self, 
        result: PipelineResult, 
        output_path: str | Path | None = None,
        export_analytics: bool = False,
    ) -> Path:
        """
        Save pipeline results to disk.
        
        Args:
            result: Pipeline execution result
            output_path: Override output path from config
            export_analytics: Also write an analyst-friendly CSV
                to data/processed/retain_analytics.csv
        
        Returns:
            Path to saved features file
        """
        output_path = Path(output_path or self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.output_format == "parquet":
            features_file = output_path / f"features_{timestamp}.parquet"
            result.features.to_parquet(features_file)
            
            if not result.target.empty:
                target_file = output_path / f"target_{timestamp}.parquet"
                result.target.to_parquet(target_file)
                
        elif self.config.output_format == "csv":
            features_file = output_path / f"features_{timestamp}.csv"
            result.features.to_csv(features_file)
            
            if not result.target.empty:
                target_file = output_path / f"target_{timestamp}.csv"
                result.target.to_csv(target_file)
        else:
            raise ValueError(f"Unknown output format: {self.config.output_format}")
        
        # Save metadata
        import json
        metadata_file = output_path / f"metadata_{timestamp}.json"
        with open(metadata_file, "w") as f:
            json.dump(result.metadata, f, indent=2)
        
        logger.info(f"Saved features to {features_file}")
        
        # Export analytics CSV for ad-hoc reporting
        if export_analytics:
            from .export import export_analytics_csv
            
            analytics_dir = output_path.parent  # data/processed/
            export_analytics_csv(self.engine, result, output_dir=analytics_dir)
        
        return features_file


def create_training_dataset(
    engine: Engine,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to create a training dataset.
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    config = config or FeatureConfig()
    pipeline = FeaturePipeline(config, engine)
    result = pipeline.run(include_target=True)
    
    if not result.success:
        raise RuntimeError(f"Pipeline failed: {result.errors}")
    
    target = result.target["churned"] if "churned" in result.target.columns else None
    return result.features, target


def create_inference_features(
    engine: Engine,
    account_ids: list[str],
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Create features for specific accounts (inference/scoring).
    
    Args:
        engine: Database engine
        account_ids: List of accounts to score
        config: Optional configuration override
    
    Returns:
        Features DataFrame for the specified accounts
    """
    config = config or FeatureConfig()
    pipeline = FeaturePipeline(config, engine)
    result = pipeline.run(account_ids=account_ids, include_target=False)
    
    return result.features
