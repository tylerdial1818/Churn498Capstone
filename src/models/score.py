"""
Batch scoring pipeline.

This module implements batch prediction for churn risk, generating
predictions for all active accounts and writing them to the database.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import ScoringConfig
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of a batch scoring run."""
    n_accounts_scored: int
    predictions: pd.DataFrame
    risk_distribution: dict[str, int]
    model_version: int
    model_name: str
    scoring_date: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    errors: list[str] = field(default_factory=list)

    def get_high_risk_accounts(self) -> pd.DataFrame:
        """Get accounts classified as high risk."""
        return self.predictions[self.predictions["risk_tier"] == "high"]

    def get_summary(self) -> dict[str, Any]:
        """Get scoring summary statistics."""
        return {
            "n_accounts": self.n_accounts_scored,
            "model_version": self.model_version,
            "risk_distribution": self.risk_distribution,
            "mean_score": float(self.predictions["churn_probability"].mean()),
            "median_score": float(self.predictions["churn_probability"].median()),
            "scoring_date": self.scoring_date,
        }


class BatchScorer:
    """
    Batch scoring pipeline for churn prediction.

    Loads the production model and generates predictions for all active
    accounts, writing results to the database for downstream use.

    Example:
        >>> scorer = BatchScorer(engine, ScoringConfig())
        >>> result = scorer.score(features_df)
        >>> print(f"Scored {result.n_accounts_scored} accounts")
        >>> print(f"High risk: {result.risk_distribution['high']}")
    """

    def __init__(
        self,
        engine: Engine,
        config: ScoringConfig,
        registry: ModelRegistry | None = None,
    ):
        """
        Initialize the batch scorer.

        Args:
            engine: SQLAlchemy database engine
            config: Scoring configuration
            registry: Optional ModelRegistry (creates one if None)
        """
        self.engine = engine
        self.config = config
        self.registry = registry or ModelRegistry()

        logger.info("Initialized batch scorer")

    def score(self, features: pd.DataFrame) -> ScoringResult:
        """
        Generate predictions for a batch of accounts.

        Args:
            features: DataFrame with account features (index = account_id)

        Returns:
            ScoringResult with predictions and metadata
        """
        logger.info(f"Starting batch scoring for {len(features)} accounts")
        errors: list[str] = []

        try:
            # Load model
            if self.config.use_production_model:
                logger.info("Loading production model")
                model = self.registry.load_production_model()
                # Get production model version
                prod_versions = self.registry.client.get_latest_versions(
                    self.registry.model_name, stages=["Production"]
                )
                model_version = int(prod_versions[0].version) if prod_versions else 0
            elif self.config.model_version:
                logger.info(f"Loading model v{self.config.model_version}")
                model = self.registry.load_model(version=self.config.model_version)
                model_version = self.config.model_version
            else:
                raise ValueError("Must specify either use_production_model=True or model_version")

            # Generate predictions in batches
            predictions = []
            batch_size = self.config.batch_size

            for i in range(0, len(features), batch_size):
                batch = features.iloc[i : i + batch_size]
                batch_preds = model.predict(batch)

                # MLflow pyfunc returns numpy array, get probabilities
                if len(batch_preds.shape) == 2:
                    # Probability for positive class
                    batch_probs = batch_preds[:, 1]
                else:
                    # Already probabilities
                    batch_probs = batch_preds

                predictions.extend(batch_probs)

                if (i // batch_size + 1) % 10 == 0:
                    logger.debug(f"Processed {i + len(batch)}/{len(features)} accounts")

            # Build predictions DataFrame
            predictions_df = pd.DataFrame(
                {
                    "account_id": features.index,
                    "churn_probability": predictions,
                }
            )

            # Classify into risk tiers
            predictions_df["risk_tier"] = predictions_df["churn_probability"].apply(
                self.config.classify_risk
            )

            # Add metadata
            predictions_df["model_version"] = model_version
            predictions_df["scored_at"] = datetime.now()

            # Compute risk distribution
            risk_distribution = (
                predictions_df["risk_tier"].value_counts().to_dict()
            )

            logger.info(
                f"Scoring complete: {len(predictions_df)} accounts, "
                f"high_risk={risk_distribution.get('high', 0)}, "
                f"medium_risk={risk_distribution.get('medium', 0)}, "
                f"low_risk={risk_distribution.get('low', 0)}"
            )

            result = ScoringResult(
                n_accounts_scored=len(predictions_df),
                predictions=predictions_df,
                risk_distribution=risk_distribution,
                model_version=model_version,
                model_name=self.registry.model_name,
                success=True,
                errors=errors,
            )

            return result

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            errors.append(str(e))
            raise

    def score_and_save(
        self,
        features: pd.DataFrame,
        write_to_db: bool = True,
    ) -> ScoringResult:
        """
        Score accounts and optionally write to database.

        Args:
            features: Account features
            write_to_db: If True, write predictions to database

        Returns:
            ScoringResult
        """
        result = self.score(features)

        if write_to_db:
            self.save_predictions(result.predictions)

        return result

    def save_predictions(self, predictions: pd.DataFrame) -> None:
        """
        Write predictions to the database.

        Args:
            predictions: DataFrame with predictions
        """
        logger.info(f"Writing {len(predictions)} predictions to {self.config.output_table}")

        # Ensure table exists
        if self.config.create_table_if_missing:
            self._create_predictions_table()

        # Write to database
        with self.engine.begin() as conn:
            # Use to_sql with if_exists='append'
            predictions.to_sql(
                self.config.output_table,
                conn,
                if_exists="append",
                index=False,
                method="multi",
            )

        logger.info("Predictions saved successfully")

    def _create_predictions_table(self) -> None:
        """Create the predictions table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.output_table} (
            id SERIAL PRIMARY KEY,
            account_id VARCHAR(50) NOT NULL,
            churn_probability FLOAT NOT NULL,
            risk_tier VARCHAR(20) NOT NULL,
            model_version INTEGER NOT NULL,
            scored_at TIMESTAMP NOT NULL,
            CONSTRAINT valid_probability CHECK (churn_probability >= 0 AND churn_probability <= 1),
            CONSTRAINT valid_risk_tier CHECK (risk_tier IN ('low', 'medium', 'high'))
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_account_id
            ON {self.config.output_table} (account_id);

        CREATE INDEX IF NOT EXISTS idx_predictions_scored_at
            ON {self.config.output_table} (scored_at);

        CREATE INDEX IF NOT EXISTS idx_predictions_risk_tier
            ON {self.config.output_table} (risk_tier);
        """

        with self.engine.begin() as conn:
            conn.execute(text(create_table_sql))

        logger.info(f"Ensured {self.config.output_table} table exists")

    def get_latest_scores(
        self,
        account_ids: list[str] | None = None,
        risk_tier: str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve latest predictions from the database.

        Args:
            account_ids: Optional list of account IDs to filter
            risk_tier: Optional risk tier filter ('low', 'medium', 'high')

        Returns:
            DataFrame with latest predictions
        """
        # Build query to get latest prediction per account
        query = f"""
        WITH latest_scores AS (
            SELECT
                account_id,
                churn_probability,
                risk_tier,
                model_version,
                scored_at,
                ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY scored_at DESC) as rn
            FROM {self.config.output_table}
        )
        SELECT
            account_id,
            churn_probability,
            risk_tier,
            model_version,
            scored_at
        FROM latest_scores
        WHERE rn = 1
        """

        # Add filters
        conditions = []
        if account_ids:
            ids_str = ", ".join(f"'{aid}'" for aid in account_ids)
            conditions.append(f"account_id IN ({ids_str})")

        if risk_tier:
            conditions.append(f"risk_tier = '{risk_tier}'")

        if conditions:
            query += " AND " + " AND ".join(conditions)

        query += " ORDER BY churn_probability DESC"

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        logger.info(f"Retrieved {len(df)} predictions from database")

        return df

    def get_scoring_history(
        self, account_id: str, limit: int = 10
    ) -> pd.DataFrame:
        """
        Get scoring history for a specific account.

        Args:
            account_id: Account to get history for
            limit: Maximum number of records to return

        Returns:
            DataFrame with scoring history
        """
        query = f"""
        SELECT
            scored_at,
            churn_probability,
            risk_tier,
            model_version
        FROM {self.config.output_table}
        WHERE account_id = :account_id
        ORDER BY scored_at DESC
        LIMIT :limit
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params={"account_id": account_id, "limit": limit},
            )

        return df


def score_active_accounts(
    engine: Engine,
    config: ScoringConfig | None = None,
    reference_date: str | None = None,
) -> ScoringResult:
    """
    Convenience function to score all active accounts.

    This runs the feature pipeline for current data and generates predictions.

    Args:
        engine: Database engine
        config: Optional scoring configuration
        reference_date: Optional reference date for feature generation

    Returns:
        ScoringResult with predictions
    """
    from src.features import FeatureConfig, FeaturePipeline

    config = config or ScoringConfig()

    # Generate features for active accounts
    logger.info("Generating features for active accounts")
    feature_config = FeatureConfig(reference_date=reference_date)
    pipeline = FeaturePipeline(feature_config, engine)

    # Get only active accounts (not churned)
    # We'll filter in the feature pipeline or score all and filter later
    result = pipeline.run(include_target=False)

    if not result.success:
        raise RuntimeError(f"Feature pipeline failed: {result.errors}")

    # Score the features
    scorer = BatchScorer(engine, config)
    scoring_result = scorer.score_and_save(result.features, write_to_db=True)

    return scoring_result
