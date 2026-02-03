"""
Model training pipeline.

This module implements the training pipeline for churn prediction models,
including data splitting, model training, cross-validation, and MLflow tracking.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import ModelConfig, ModelType, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of a training pipeline execution."""

    # Trained models
    model: Any  # Primary model (XGBoost, LightGBM, etc.)
    baseline_model: Any | None  # Logistic regression baseline

    # Performance metrics
    metrics: dict[str, float]  # Test set metrics
    baseline_metrics: dict[str, float]  # Baseline model metrics
    cv_metrics: dict[str, list[float]]  # Cross-validation metrics per fold

    # Model interpretation
    feature_importances: pd.DataFrame  # Feature importance scores
    shap_values: np.ndarray | None = None  # SHAP values for test set
    shap_summary: dict[str, Any] = field(default_factory=dict)

    # Metadata
    mlflow_run_id: str | None = None
    training_date: str = field(default_factory=lambda: datetime.now().isoformat())
    model_config: ModelConfig | None = None
    training_config: TrainingConfig | None = None

    # Data fingerprint for tracking
    data_hash: str | None = None
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_test_samples: int = 0

    # Success tracking
    success: bool = True
    errors: list[str] = field(default_factory=list)

    @property
    def beats_baseline(self) -> bool:
        """Check if primary model beats baseline."""
        if not self.baseline_metrics:
            return True
        primary_auc = self.metrics.get("auc", 0)
        baseline_auc = self.baseline_metrics.get("auc", 0)
        return primary_auc > baseline_auc

    @property
    def improvement_over_baseline(self) -> float:
        """Compute AUC improvement over baseline."""
        if not self.baseline_metrics:
            return 0.0
        primary_auc = self.metrics.get("auc", 0)
        baseline_auc = self.baseline_metrics.get("auc", 0)
        return primary_auc - baseline_auc


class ModelTrainer:
    """
    Trains churn prediction models with MLflow tracking.

    Implements the "two model" pattern: always trains both a complex model
    (XGBoost/LightGBM) and a simple baseline (logistic regression) for comparison.

    Example:
        >>> from src.features import create_training_dataset
        >>> X, y = create_training_dataset(engine)
        >>> trainer = ModelTrainer(ModelConfig(), TrainingConfig())
        >>> result = trainer.train(X, y)
        >>> print(f"Test AUC: {result.metrics['auc']:.3f}")
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ):
        """
        Initialize the model trainer.

        Args:
            model_config: Model hyperparameters and settings
            training_config: Training pipeline configuration
        """
        self.model_config = model_config
        self.training_config = training_config

        # Set up MLflow tracking
        mlflow.set_tracking_uri(training_config.tracking_uri)
        mlflow.set_experiment(training_config.experiment_name)

        logger.info(
            f"Initialized trainer with model={model_config.model_type.value}, "
            f"experiment={training_config.experiment_name}"
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> TrainingResult:
        """
        Execute the full training pipeline.

        Args:
            X: Feature matrix
            y: Target variable (binary: 0=not churned, 1=churned)
            feature_names: Optional feature name list (uses X.columns if None)

        Returns:
            TrainingResult with trained models and metrics
        """
        start_time = datetime.now()
        errors: list[str] = []

        # Handle target as Series
        if isinstance(y, pd.DataFrame):
            if "churned" in y.columns:
                y = y["churned"]
            else:
                y = y.iloc[:, 0]

        if feature_names is None:
            feature_names = X.columns.tolist()

        logger.info(
            f"Starting training: {len(X)} samples, {len(feature_names)} features"
        )

        # Compute data hash for versioning
        data_hash = self._hash_data(X, y)

        try:
            # Split data: 60% train, 20% validation, 20% test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=self.model_config.test_size,
                random_state=self.model_config.random_seed,
                stratify=y if self.model_config.stratify else None,
            )

            # Split remaining into train and validation
            val_size_adjusted = self.model_config.validation_size / (
                1 - self.model_config.test_size
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_size_adjusted,
                random_state=self.model_config.random_seed,
                stratify=y_temp if self.model_config.stratify else None,
            )

            logger.info(
                f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            )

            # Start MLflow run
            run_name = self.training_config.run_name or self._generate_run_name()

            with mlflow.start_run(run_name=run_name) as run:
                mlflow_run_id = run.info.run_id

                # Log configuration
                self._log_config(mlflow_run_id)
                mlflow.log_param("n_features", len(feature_names))
                mlflow.log_param("n_train", len(X_train))
                mlflow.log_param("n_val", len(X_val))
                mlflow.log_param("n_test", len(X_test))
                mlflow.log_param("data_hash", data_hash)

                # Train primary model
                logger.info(
                    f"Training primary model: {self.model_config.model_type.value}"
                )
                model = self._train_model(X_train, y_train, X_val, y_val)

                # Compute primary model metrics
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics = self._compute_metrics(y_test, y_pred_proba)
                self._log_metrics(metrics, prefix="test")

                # Cross-validation
                logger.info("Running cross-validation")
                cv_metrics = self._cross_validate(X_temp, y_temp)
                self._log_cv_metrics(cv_metrics)

                # Train baseline model
                baseline_model = None
                baseline_metrics = {}
                if self.training_config.train_baseline:
                    logger.info("Training baseline model")
                    baseline_model = self._train_baseline(X_train, y_train)
                    y_baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
                    baseline_metrics = self._compute_metrics(y_test, y_baseline_proba)
                    self._log_metrics(baseline_metrics, prefix="baseline")

                    # Log comparison
                    improvement = metrics["auc"] - baseline_metrics["auc"]
                    mlflow.log_metric("auc_improvement_over_baseline", improvement)
                    mlflow.log_metric(
                        "beats_baseline", 1.0 if improvement > 0 else 0.0
                    )

                # Feature importances
                feature_importances = self._get_feature_importances(
                    model, feature_names
                )
                self._log_feature_importances(feature_importances)

                # SHAP values
                shap_values = None
                shap_summary = {}
                if self.training_config.compute_shap:
                    try:
                        logger.info("Computing SHAP values")
                        shap_values, shap_summary = self._compute_shap(
                            model, X_test, feature_names
                        )
                    except Exception as e:
                        logger.warning(f"SHAP computation failed: {e}")
                        errors.append(f"SHAP computation failed: {e}")

                # Log models
                if self.training_config.log_model:
                    self._log_model(model, feature_names)
                    if baseline_model is not None:
                        mlflow.sklearn.log_model(baseline_model, "baseline_model")

                # Training duration
                duration = (datetime.now() - start_time).total_seconds()
                mlflow.log_metric("training_duration_seconds", duration)

                logger.info(
                    f"Training complete: AUC={metrics['auc']:.4f}, "
                    f"duration={duration:.1f}s, run_id={mlflow_run_id}"
                )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            errors.append(str(e))
            raise

        return TrainingResult(
            model=model,
            baseline_model=baseline_model,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            cv_metrics=cv_metrics,
            feature_importances=feature_importances,
            shap_values=shap_values,
            shap_summary=shap_summary,
            mlflow_run_id=mlflow_run_id,
            model_config=self.model_config,
            training_config=self.training_config,
            data_hash=data_hash,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val),
            n_test_samples=len(X_test),
            success=len(errors) == 0,
            errors=errors,
        )

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Any:
        """Train the primary model based on configuration."""
        params = self.model_config.get_params()

        match self.model_config.model_type:
            case ModelType.XGBOOST:
                import xgboost as xgb

                model = xgb.XGBClassifier(**params)

                # Train with early stopping on validation set
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

            case ModelType.LIGHTGBM:
                import lightgbm as lgb

                model = lgb.LGBMClassifier(**params)

                # Train with early stopping
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(self.model_config.early_stopping_rounds)
                    ],
                )

            case ModelType.LOGISTIC_REGRESSION:
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

        return model

    def _train_baseline(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> LogisticRegression:
        """Train a logistic regression baseline."""
        baseline_params = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": self.model_config.random_seed,
        }
        model = LogisticRegression(**baseline_params)
        model.fit(X_train, y_train)
        return model

    def _cross_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, list[float]]:
        """Perform stratified k-fold cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.model_config.cv_folds,
            shuffle=True,
            random_state=self.model_config.random_seed,
        )

        cv_metrics: dict[str, list[float]] = {
            "auc": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            # Train model on this fold (without validation set for CV)
            params = self.model_config.get_params()

            match self.model_config.model_type:
                case ModelType.XGBOOST:
                    import xgboost as xgb

                    model = xgb.XGBClassifier(**params)
                    model.fit(X_fold_train, y_fold_train, verbose=False)

                case ModelType.LIGHTGBM:
                    import lightgbm as lgb

                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_fold_train, y_fold_train)

                case ModelType.LOGISTIC_REGRESSION:
                    model = LogisticRegression(**params)
                    model.fit(X_fold_train, y_fold_train)

            # Evaluate
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            cv_metrics["auc"].append(roc_auc_score(y_fold_val, y_pred_proba))
            cv_metrics["accuracy"].append(accuracy_score(y_fold_val, y_pred))
            cv_metrics["precision"].append(
                precision_score(y_fold_val, y_pred, zero_division=0)
            )
            cv_metrics["recall"].append(
                recall_score(y_fold_val, y_pred, zero_division=0)
            )
            cv_metrics["f1"].append(f1_score(y_fold_val, y_pred, zero_division=0))

            logger.debug(f"Fold {fold + 1}/{self.model_config.cv_folds}: AUC={cv_metrics['auc'][-1]:.4f}")

        return cv_metrics

    def _compute_metrics(
        self, y_true: pd.Series, y_pred_proba: np.ndarray
    ) -> dict[str, float]:
        """Compute evaluation metrics."""
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "auc": roc_auc_score(y_true, y_pred_proba),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "average_precision": average_precision_score(y_true, y_pred_proba),
            "log_loss": log_loss(y_true, y_pred_proba),
        }

        return metrics

    def _get_feature_importances(
        self, model: Any, feature_names: list[str]
    ) -> pd.DataFrame:
        """Extract feature importances from the model."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For logistic regression, use absolute coefficients
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model has no feature importances")
            importances = np.zeros(len(feature_names))

        df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        return df

    def _compute_shap(
        self, model: Any, X: pd.DataFrame, feature_names: list[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute SHAP values for model interpretation."""
        try:
            import shap
        except ImportError:
            logger.warning("shap not installed, skipping SHAP computation")
            return None, {}

        # Sample data if too large
        if len(X) > self.training_config.shap_sample_size:
            X_sample = X.sample(
                n=self.training_config.shap_sample_size, random_state=42
            )
        else:
            X_sample = X

        # Create explainer based on model type
        if self.model_config.model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # For binary classification, TreeExplainer returns values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # Compute summary statistics
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)

        summary = {
            "top_features": shap_importance.head(10).to_dict("records"),
            "mean_prediction": float(np.mean(shap_values)),
        }

        return shap_values, summary

    def _log_config(self, run_id: str) -> None:
        """Log configuration parameters to MLflow."""
        mlflow.log_param("model_type", self.model_config.model_type.value)
        mlflow.log_param("random_seed", self.model_config.random_seed)
        mlflow.log_param("cv_folds", self.model_config.cv_folds)

        # Log hyperparameters
        params = self.model_config.get_params()
        for key, value in params.items():
            mlflow.log_param(f"hp_{key}", value)

    def _log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        """Log metrics to MLflow."""
        for name, value in metrics.items():
            metric_name = f"{prefix}_{name}" if prefix else name
            mlflow.log_metric(metric_name, value)

    def _log_cv_metrics(self, cv_metrics: dict[str, list[float]]) -> None:
        """Log cross-validation metrics to MLflow."""
        for metric_name, values in cv_metrics.items():
            mlflow.log_metric(f"cv_{metric_name}_mean", np.mean(values))
            mlflow.log_metric(f"cv_{metric_name}_std", np.std(values))

    def _log_feature_importances(self, importances: pd.DataFrame) -> None:
        """Log feature importances to MLflow."""
        # Log top 20 features as parameters
        for idx, row in importances.head(20).iterrows():
            mlflow.log_param(
                f"top_feature_{idx + 1}", f"{row['feature']}={row['importance']:.4f}"
            )

    def _log_model(self, model: Any, feature_names: list[str]) -> None:
        """Log trained model to MLflow."""
        match self.model_config.model_type:
            case ModelType.XGBOOST:
                mlflow.xgboost.log_model(model, "model")
            case ModelType.LIGHTGBM:
                import mlflow.lightgbm

                mlflow.lightgbm.log_model(model, "model")
            case ModelType.LOGISTIC_REGRESSION:
                mlflow.sklearn.log_model(model, "model")

    def _hash_data(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Compute hash of training data for versioning."""
        # Hash based on shape and first/last rows
        data_str = f"{X.shape}_{y.shape}_{X.iloc[0].sum()}_{X.iloc[-1].sum()}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def _generate_run_name(self) -> str:
        """Generate a run name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.model_config.model_type.value}_{timestamp}"
