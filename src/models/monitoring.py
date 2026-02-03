"""
Model monitoring and drift detection.

This module implements data drift, prediction drift, and performance
monitoring to track model health over time.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import MonitoringConfig

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status of the model."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature_name: str
    drift_score: float
    status: HealthStatus
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass
class MonitoringReport:
    """Complete monitoring report for a model."""
    # Overall health
    health_status: HealthStatus
    report_date: str = field(default_factory=lambda: datetime.now().isoformat())

    # Drift detection
    feature_drift: list[DriftResult] = field(default_factory=list)
    prediction_drift_score: float | None = None
    prediction_drift_status: HealthStatus = HealthStatus.HEALTHY

    # Performance metrics (when labels available)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    performance_degradation: float | None = None

    # Recommendations
    recommended_actions: list[str] = field(default_factory=list)

    # Metadata
    reference_period: str | None = None
    monitoring_period: str | None = None

    def get_drifted_features(self, status: HealthStatus | None = None) -> list[DriftResult]:
        """Get features with drift (optionally filtered by status)."""
        if status:
            return [f for f in self.feature_drift if f.status == status]
        return [f for f in self.feature_drift if f.status != HealthStatus.HEALTHY]

    def summary(self) -> str:
        """Get a text summary of the monitoring report."""
        critical = self.get_drifted_features(HealthStatus.CRITICAL)
        warning = self.get_drifted_features(HealthStatus.WARNING)

        summary = f"Model Health: {self.health_status.value.upper()}\n"
        summary += f"Features with critical drift: {len(critical)}\n"
        summary += f"Features with warning drift: {len(warning)}\n"

        if self.prediction_drift_score is not None:
            summary += f"Prediction drift: {self.prediction_drift_status.value}\n"

        if self.performance_degradation is not None:
            summary += f"Performance degradation: {self.performance_degradation:.4f}\n"

        if self.recommended_actions:
            summary += "\nRecommended actions:\n"
            for action in self.recommended_actions:
                summary += f"  - {action}\n"

        return summary


class ModelMonitor:
    """
    Monitor model health through drift detection.

    Tracks data drift (feature distributions), prediction drift (output
    distributions), and performance drift (when ground truth labels are available).

    Example:
        >>> monitor = ModelMonitor(engine, MonitoringConfig())
        >>> report = monitor.check_drift(current_features, reference_features)
        >>> if report.health_status == HealthStatus.CRITICAL:
        >>>     print("Model needs retraining!")
    """

    def __init__(self, engine: Engine, config: MonitoringConfig):
        """
        Initialize the model monitor.

        Args:
            engine: SQLAlchemy database engine
            config: Monitoring configuration
        """
        self.engine = engine
        self.config = config
        logger.info("Initialized model monitor")

    def check_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        current_predictions: pd.Series | None = None,
        reference_predictions: pd.Series | None = None,
    ) -> MonitoringReport:
        """
        Check for data and prediction drift.

        Args:
            current_data: Recent feature data
            reference_data: Reference feature data (training distribution)
            current_predictions: Optional current predictions
            reference_predictions: Optional reference predictions

        Returns:
            MonitoringReport with drift analysis
        """
        logger.info(
            f"Checking drift: {len(current_data)} current vs {len(reference_data)} reference samples"
        )

        feature_drift = []
        recommended_actions = []

        # Check drift for each feature
        common_features = set(current_data.columns) & set(reference_data.columns)

        for feature in common_features:
            drift_result = self._compute_feature_drift(
                reference_data[feature],
                current_data[feature],
                feature,
            )
            feature_drift.append(drift_result)

        # Prediction drift
        prediction_drift_score = None
        prediction_drift_status = HealthStatus.HEALTHY

        if current_predictions is not None and reference_predictions is not None:
            prediction_drift_score, prediction_drift_status = self._check_prediction_drift(
                reference_predictions, current_predictions
            )

        # Determine overall health status
        critical_features = [f for f in feature_drift if f.status == HealthStatus.CRITICAL]
        warning_features = [f for f in feature_drift if f.status == HealthStatus.WARNING]

        if len(critical_features) > 0 or prediction_drift_status == HealthStatus.CRITICAL:
            health_status = HealthStatus.CRITICAL
            recommended_actions.append(
                f"URGENT: Retrain model - {len(critical_features)} features with critical drift"
            )
        elif len(warning_features) > 0 or prediction_drift_status == HealthStatus.WARNING:
            health_status = HealthStatus.WARNING
            recommended_actions.append(
                f"Investigate drift in {len(warning_features)} features"
            )
        else:
            health_status = HealthStatus.HEALTHY

        # Add specific recommendations
        if critical_features:
            top_drifted = sorted(critical_features, key=lambda x: x.drift_score, reverse=True)[:3]
            recommended_actions.append(
                f"Focus on features: {', '.join(f.feature_name for f in top_drifted)}"
            )

        report = MonitoringReport(
            health_status=health_status,
            feature_drift=feature_drift,
            prediction_drift_score=prediction_drift_score,
            prediction_drift_status=prediction_drift_status,
            recommended_actions=recommended_actions,
        )

        logger.info(
            f"Drift check complete: {health_status.value} "
            f"({len(critical_features)} critical, {len(warning_features)} warning)"
        )

        return report

    def _compute_feature_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> DriftResult:
        """
        Compute drift score for a single feature using PSI.

        Population Stability Index (PSI) is a standard metric for drift detection:
        - PSI < 0.1: No significant shift
        - 0.1 <= PSI < 0.25: Moderate shift, investigate
        - PSI >= 0.25: Significant shift, action needed
        """
        # Handle missing values
        reference = reference.dropna()
        current = current.dropna()

        # Compute statistics
        ref_mean = float(reference.mean())
        ref_std = float(reference.std())
        cur_mean = float(current.mean())
        cur_std = float(current.std())

        # Compute PSI
        psi = self._calculate_psi(reference, current)

        # Determine status based on PSI thresholds
        if psi >= self.config.psi_critical_threshold:
            status = HealthStatus.CRITICAL
        elif psi >= self.config.psi_warning_threshold:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return DriftResult(
            feature_name=feature_name,
            drift_score=psi,
            status=status,
            reference_mean=ref_mean,
            current_mean=cur_mean,
            reference_std=ref_std,
            current_std=cur_std,
        )

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift in a distribution by comparing the proportion
        of samples in each bin between reference and current data.
        """
        # Create bins based on reference distribution
        try:
            # Use quantile-based binning
            _, bin_edges = pd.qcut(reference, q=n_bins, retbins=True, duplicates="drop")
        except ValueError:
            # Fall back to equal-width bins if quantile binning fails
            bin_edges = np.linspace(reference.min(), reference.max(), n_bins + 1)

        # Ensure we have valid bins
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Bin both distributions
        ref_counts = pd.cut(reference, bins=bin_edges).value_counts(normalize=True, sort=False)
        cur_counts = pd.cut(current, bins=bin_edges).value_counts(normalize=True, sort=False)

        # Align the two series
        ref_counts, cur_counts = ref_counts.align(cur_counts, fill_value=0)

        # Add small constant to avoid log(0)
        epsilon = 1e-10
        ref_counts = ref_counts + epsilon
        cur_counts = cur_counts + epsilon

        # Calculate PSI
        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))

        return float(psi)

    def _check_prediction_drift(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
    ) -> tuple[float, HealthStatus]:
        """
        Check for drift in prediction distributions.

        Uses statistical comparison of mean predictions to detect shifts.
        """
        ref_mean = reference_predictions.mean()
        ref_std = reference_predictions.std()
        cur_mean = current_predictions.mean()

        # Check if current mean is outside expected range (z-score approach)
        z_score = abs(cur_mean - ref_mean) / (ref_std + 1e-10)

        if z_score >= self.config.prediction_mean_shift_threshold:
            status = HealthStatus.CRITICAL
        elif z_score >= self.config.prediction_mean_shift_threshold / 2:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        logger.debug(
            f"Prediction drift: mean {ref_mean:.3f} -> {cur_mean:.3f}, z-score={z_score:.2f}"
        )

        return float(z_score), status

    def check_performance_drift(
        self,
        y_true: pd.Series,
        y_pred_proba: pd.Series,
        reference_auc: float,
    ) -> tuple[dict[str, float], float]:
        """
        Check for performance drift when ground truth labels are available.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            reference_auc: Reference AUC from training

        Returns:
            Tuple of (current_metrics, degradation)
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        y_pred = (y_pred_proba >= 0.5).astype(int)

        current_metrics = {
            "auc": roc_auc_score(y_true, y_pred_proba),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
        }

        degradation = reference_auc - current_metrics["auc"]

        logger.info(
            f"Performance: AUC={current_metrics['auc']:.4f} "
            f"(reference={reference_auc:.4f}, degradation={degradation:.4f})"
        )

        return current_metrics, degradation

    def save_monitoring_report(self, report: MonitoringReport) -> None:
        """
        Save monitoring report to database.

        Args:
            report: MonitoringReport to save
        """
        if self.config.create_table_if_missing:
            self._create_monitoring_table()

        # Prepare report data
        report_data = {
            "report_date": datetime.fromisoformat(report.report_date),
            "health_status": report.health_status.value,
            "n_drifted_features": len(report.get_drifted_features()),
            "n_critical_features": len(report.get_drifted_features(HealthStatus.CRITICAL)),
            "prediction_drift_score": report.prediction_drift_score,
            "prediction_drift_status": report.prediction_drift_status.value,
            "performance_degradation": report.performance_degradation,
            "recommended_actions": "; ".join(report.recommended_actions),
        }

        # Insert into database
        with self.engine.begin() as conn:
            df = pd.DataFrame([report_data])
            df.to_sql(
                self.config.log_table,
                conn,
                if_exists="append",
                index=False,
            )

        logger.info(f"Monitoring report saved to {self.config.log_table}")

    def _create_monitoring_table(self) -> None:
        """Create the monitoring log table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.log_table} (
            id SERIAL PRIMARY KEY,
            report_date TIMESTAMP NOT NULL,
            health_status VARCHAR(20) NOT NULL,
            n_drifted_features INTEGER,
            n_critical_features INTEGER,
            prediction_drift_score FLOAT,
            prediction_drift_status VARCHAR(20),
            performance_degradation FLOAT,
            recommended_actions TEXT,
            CONSTRAINT valid_health_status CHECK (health_status IN ('healthy', 'warning', 'critical'))
        );

        CREATE INDEX IF NOT EXISTS idx_monitoring_report_date
            ON {self.config.log_table} (report_date);
        """

        with self.engine.begin() as conn:
            conn.execute(text(create_table_sql))

        logger.info(f"Ensured {self.config.log_table} table exists")

    def get_monitoring_history(
        self, days: int = 30
    ) -> pd.DataFrame:
        """
        Retrieve monitoring history from the database.

        Args:
            days: Number of days of history to retrieve

        Returns:
            DataFrame with monitoring history
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        query = f"""
        SELECT *
        FROM {self.config.log_table}
        WHERE report_date >= :cutoff_date
        ORDER BY report_date DESC
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params={"cutoff_date": cutoff_date},
            )

        logger.info(f"Retrieved {len(df)} monitoring reports from last {days} days")

        return df


def print_monitoring_report(report: MonitoringReport) -> None:
    """Print a formatted monitoring report."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Header
    status_styles = {
        HealthStatus.HEALTHY: "bold green",
        HealthStatus.WARNING: "bold yellow",
        HealthStatus.CRITICAL: "bold red",
    }
    style = status_styles[report.health_status]
    console.print(f"\n[{style}]Model Health: {report.health_status.value.upper()}[/{style}]\n")

    # Feature drift table
    if report.feature_drift:
        table = Table(title="Feature Drift Analysis")
        table.add_column("Feature", style="cyan")
        table.add_column("PSI Score", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Mean Change", justify="right")

        # Show top 10 most drifted features
        sorted_drift = sorted(report.feature_drift, key=lambda x: x.drift_score, reverse=True)
        for drift in sorted_drift[:10]:
            status_icon = {
                HealthStatus.HEALTHY: "[green]✓[/green]",
                HealthStatus.WARNING: "[yellow]⚠[/yellow]",
                HealthStatus.CRITICAL: "[red]✗[/red]",
            }[drift.status]

            mean_change = drift.current_mean - drift.reference_mean
            mean_change_pct = (
                (mean_change / (drift.reference_mean + 1e-10)) * 100
            )

            table.add_row(
                drift.feature_name,
                f"{drift.drift_score:.4f}",
                status_icon,
                f"{mean_change_pct:+.1f}%",
            )

        console.print(table)

    # Prediction drift
    if report.prediction_drift_score is not None:
        console.print(f"\n[bold]Prediction Drift:[/bold] {report.prediction_drift_score:.4f}")
        console.print(f"Status: {report.prediction_drift_status.value}")

    # Performance
    if report.performance_degradation is not None:
        console.print(f"\n[bold]Performance Degradation:[/bold] {report.performance_degradation:.4f}")

    # Recommendations
    if report.recommended_actions:
        console.print("\n[bold]Recommended Actions:[/bold]")
        for action in report.recommended_actions:
            console.print(f"  • {action}")
