"""
Model registry interface.

This module provides a clean interface to MLflow's model registry,
handling model versioning, promotion, and retrieval.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from .train import TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model version."""
    name: str
    version: int
    stage: str  # "None", "Staging", "Production", "Archived"
    run_id: str
    metrics: dict[str, float]
    feature_list: list[str]
    training_date: str
    data_hash: str | None = None
    description: str | None = None


class ModelRegistry:
    """
    Wrapper around MLflow Model Registry.

    Provides a clean interface for registering models, promoting them
    through stages, and loading production models for scoring.

    Example:
        >>> registry = ModelRegistry("churn_model")
        >>> registry.register_model(training_result)
        >>> registry.promote_model(version=1, stage="Production")
        >>> model = registry.load_production_model()
    """

    def __init__(
        self,
        model_name: str = "churn_prediction_model",
        tracking_uri: str = "./mlruns",
    ):
        """
        Initialize the model registry.

        Args:
            model_name: Registered model name in MLflow
            tracking_uri: MLflow tracking server URI
        """
        self.model_name = model_name
        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

        logger.info(
            f"Initialized model registry: {model_name} @ {tracking_uri}"
        )

    def register_model(
        self,
        training_result: TrainingResult,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelMetadata:
        """
        Register a trained model to the registry.

        Args:
            training_result: Result from ModelTrainer
            description: Optional model description
            tags: Optional tags for the model version

        Returns:
            ModelMetadata for the registered version
        """
        if not training_result.success:
            raise ValueError("Cannot register a failed training run")

        if not training_result.mlflow_run_id:
            raise ValueError("Training result has no MLflow run ID")

        logger.info(f"Registering model from run {training_result.mlflow_run_id}")

        # Get the model artifact URI from the run
        run = self.client.get_run(training_result.mlflow_run_id)
        model_uri = f"runs:/{training_result.mlflow_run_id}/model"

        # Register the model
        try:
            model_version = mlflow.register_model(model_uri, self.model_name)
            version_number = int(model_version.version)

            logger.info(f"Registered as {self.model_name} v{version_number}")

        except Exception as e:
            # If model doesn't exist, create it first
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.info(f"Creating new registered model: {self.model_name}")
                self.client.create_registered_model(
                    self.model_name,
                    description=f"Churn prediction model - created {datetime.now().isoformat()}",
                )
                model_version = mlflow.register_model(model_uri, self.model_name)
                version_number = int(model_version.version)
            else:
                raise

        # Update version description and tags
        if description:
            self.client.update_model_version(
                name=self.model_name,
                version=version_number,
                description=description,
            )

        # Add tags
        default_tags = {
            "training_date": training_result.training_date,
            "model_type": training_result.model_config.model_type.value
            if training_result.model_config
            else "unknown",
            "auc": str(training_result.metrics.get("auc", 0)),
            "data_hash": training_result.data_hash or "unknown",
        }

        if tags:
            default_tags.update(tags)

        for key, value in default_tags.items():
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version_number,
                key=key,
                value=value,
            )

        # Build metadata
        feature_list = training_result.feature_importances["feature"].tolist()

        metadata = ModelMetadata(
            name=self.model_name,
            version=version_number,
            stage="None",
            run_id=training_result.mlflow_run_id,
            metrics=training_result.metrics,
            feature_list=feature_list,
            training_date=training_result.training_date,
            data_hash=training_result.data_hash,
            description=description,
        )

        logger.info(
            f"Model registered: v{version_number}, AUC={training_result.metrics['auc']:.4f}"
        )

        return metadata

    def promote_model(
        self,
        version: int,
        stage: str = "Production",
        archive_existing: bool = True,
    ) -> None:
        """
        Promote a model version to a stage.

        Args:
            version: Model version number to promote
            stage: Target stage ("Staging", "Production", or "Archived")
            archive_existing: If True, archive existing models in the target stage
        """
        valid_stages = ["Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")

        logger.info(f"Promoting {self.model_name} v{version} to {stage}")

        # Archive existing models in the target stage if requested
        if archive_existing and stage != "Archived":
            existing = self.client.get_latest_versions(self.model_name, stages=[stage])
            for model_version in existing:
                logger.info(
                    f"Archiving existing {stage} model v{model_version.version}"
                )
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=model_version.version,
                    stage="Archived",
                )

        # Promote the new version
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
        )

        logger.info(f"Model v{version} promoted to {stage}")

    def load_model(self, version: int | None = None, stage: str | None = None) -> Any:
        """
        Load a model from the registry.

        Args:
            version: Specific version number (if None, uses stage)
            stage: Stage to load from (if version is None)

        Returns:
            Loaded model object
        """
        if version is not None:
            model_uri = f"models:/{self.model_name}/{version}"
            logger.info(f"Loading {self.model_name} v{version}")
        elif stage is not None:
            model_uri = f"models:/{self.model_name}/{stage}"
            logger.info(f"Loading {self.model_name} from {stage} stage")
        else:
            raise ValueError("Must specify either version or stage")

        model = mlflow.pyfunc.load_model(model_uri)
        return model

    def load_production_model(self) -> Any:
        """Load the current production model."""
        return self.load_model(stage="Production")

    def get_model_metadata(self, version: int) -> ModelMetadata:
        """
        Get metadata for a specific model version.

        Args:
            version: Model version number

        Returns:
            ModelMetadata object
        """
        model_version = self.client.get_model_version(self.model_name, version)

        # Get run to retrieve metrics
        run = self.client.get_run(model_version.run_id)
        metrics = run.data.metrics

        # Get tags
        tags = model_version.tags

        return ModelMetadata(
            name=self.model_name,
            version=int(model_version.version),
            stage=model_version.current_stage,
            run_id=model_version.run_id,
            metrics=metrics,
            feature_list=tags.get("features", "").split(",") if "features" in tags else [],
            training_date=tags.get("training_date", "unknown"),
            data_hash=tags.get("data_hash"),
            description=model_version.description,
        )

    def get_model_history(self, max_results: int = 10) -> list[ModelMetadata]:
        """
        Get history of all model versions.

        Args:
            max_results: Maximum number of versions to return

        Returns:
            List of ModelMetadata objects, sorted by version (newest first)
        """
        try:
            versions = self.client.search_model_versions(
                f"name='{self.model_name}'",
                max_results=max_results,
            )
        except Exception as e:
            logger.warning(f"Model {self.model_name} not found in registry: {e}")
            return []

        # Sort by version number (descending)
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

        metadata_list = []
        for version in versions:
            try:
                # Get run for metrics
                run = self.client.get_run(version.run_id)
                metrics = run.data.metrics
                tags = version.tags

                metadata_list.append(
                    ModelMetadata(
                        name=self.model_name,
                        version=int(version.version),
                        stage=version.current_stage,
                        run_id=version.run_id,
                        metrics=metrics,
                        feature_list=tags.get("features", "").split(",") if "features" in tags else [],
                        training_date=tags.get("training_date", "unknown"),
                        data_hash=tags.get("data_hash"),
                        description=version.description,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to load metadata for v{version.version}: {e}")

        return metadata_list

    def compare_models(
        self, version1: int, version2: int
    ) -> pd.DataFrame:
        """
        Side-by-side comparison of two model versions.

        Args:
            version1: First model version
            version2: Second model version

        Returns:
            DataFrame with comparison
        """
        meta1 = self.get_model_metadata(version1)
        meta2 = self.get_model_metadata(version2)

        # Build comparison data
        comparison = {
            "Metric": [],
            f"v{version1} ({meta1.stage})": [],
            f"v{version2} ({meta2.stage})": [],
            "Difference": [],
        }

        # Compare common metrics
        all_metrics = set(meta1.metrics.keys()) | set(meta2.metrics.keys())
        for metric in sorted(all_metrics):
            val1 = meta1.metrics.get(metric, float("nan"))
            val2 = meta2.metrics.get(metric, float("nan"))
            diff = val2 - val1

            comparison["Metric"].append(metric)
            comparison[f"v{version1} ({meta1.stage})"].append(f"{val1:.4f}")
            comparison[f"v{version2} ({meta2.stage})"].append(f"{val2:.4f}")
            comparison["Difference"].append(f"{diff:+.4f}")

        df = pd.DataFrame(comparison)

        logger.info(f"Compared v{version1} vs v{version2}")

        return df

    def delete_model_version(self, version: int) -> None:
        """
        Delete a specific model version.

        Args:
            version: Version number to delete
        """
        logger.warning(f"Deleting {self.model_name} v{version}")
        self.client.delete_model_version(self.model_name, version)

    def get_production_metrics(self) -> dict[str, float] | None:
        """
        Get metrics for the current production model.

        Returns:
            Metrics dict or None if no production model exists
        """
        try:
            prod_versions = self.client.get_latest_versions(
                self.model_name, stages=["Production"]
            )
            if not prod_versions:
                return None

            version = int(prod_versions[0].version)
            metadata = self.get_model_metadata(version)
            return metadata.metrics

        except Exception as e:
            logger.warning(f"Failed to get production metrics: {e}")
            return None
