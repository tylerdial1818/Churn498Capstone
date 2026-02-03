"""
Model evaluation gate.

This module implements quality checks that determine whether a candidate
model should be promoted to production. Think of this as "unit tests for models."
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .config import EvaluationConfig
from .train import TrainingResult

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Status of an evaluation check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single evaluation check."""
    name: str
    status: CheckStatus
    message: str
    value: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of running all evaluation checks."""
    passed: bool
    recommendation: str  # "promote", "reject", or "review"
    checks: list[CheckResult]
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float] | None = None
    production_metrics: dict[str, float] | None = None
    summary: str = ""

    def get_failed_checks(self) -> list[CheckResult]:
        """Get all failed checks."""
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    def get_warnings(self) -> list[CheckResult]:
        """Get all warnings."""
        return [c for c in self.checks if c.status == CheckStatus.WARNING]


class EvaluationGate:
    """
    Evaluation gate for model promotion decisions.

    Implements a suite of checks to determine if a candidate model should
    replace the current production model. This is the "unit test for models"
    concept - automated quality gates before deployment.

    Example:
        >>> gate = EvaluationGate(EvaluationConfig())
        >>> result = gate.evaluate(training_result, X_test, y_test)
        >>> if result.recommendation == "promote":
        >>>     print("Model passed all checks!")
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluation gate.

        Args:
            config: Evaluation configuration with thresholds
        """
        self.config = config
        logger.info("Initialized evaluation gate")

    def evaluate(
        self,
        candidate: TrainingResult,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        production_metrics: dict[str, float] | None = None,
        reference_feature_importances: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        """
        Run all evaluation checks on a candidate model.

        Args:
            candidate: Training result for the candidate model
            X_test: Test features
            y_test: Test labels
            production_metrics: Metrics from current production model (if exists)
            reference_feature_importances: Feature importances from previous model

        Returns:
            EvaluationResult with pass/fail decision and detailed check results
        """
        logger.info("Starting model evaluation")
        checks: list[CheckResult] = []

        # Get predictions for additional checks
        y_pred_proba = candidate.model.predict_proba(X_test)[:, 1]

        # Check 1: Performance threshold
        checks.append(self._check_performance_threshold(candidate.metrics))

        # Check 2: Improvement over baseline
        if candidate.baseline_metrics:
            checks.append(
                self._check_improvement_over_baseline(
                    candidate.metrics, candidate.baseline_metrics
                )
            )

        # Check 3: Improvement over production
        if production_metrics:
            checks.append(
                self._check_improvement_over_production(
                    candidate.metrics, production_metrics
                )
            )

        # Check 4: Calibration
        if self.config.check_calibration:
            checks.append(self._check_calibration(y_test, y_pred_proba))

        # Check 5: Prediction distribution
        if self.config.check_prediction_distribution:
            checks.append(self._check_prediction_distribution(y_pred_proba, y_test))

        # Check 6: Feature importance stability
        if self.config.check_feature_stability and reference_feature_importances is not None:
            checks.append(
                self._check_feature_stability(
                    candidate.feature_importances, reference_feature_importances
                )
            )

        # Determine overall pass/fail
        failed_checks = [c for c in checks if c.status == CheckStatus.FAIL]
        warnings = [c for c in checks if c.status == CheckStatus.WARNING]
        passed = len(failed_checks) == 0

        # Generate recommendation
        if passed and len(warnings) == 0:
            recommendation = "promote"
            summary = "All checks passed. Safe to promote to production."
        elif passed and len(warnings) > 0:
            recommendation = "review"
            summary = f"All checks passed but {len(warnings)} warnings. Manual review recommended."
        else:
            recommendation = "reject"
            summary = f"Failed {len(failed_checks)} checks. Do not promote."

        logger.info(
            f"Evaluation complete: {recommendation} "
            f"({len(failed_checks)} failures, {len(warnings)} warnings)"
        )

        return EvaluationResult(
            passed=passed,
            recommendation=recommendation,
            checks=checks,
            candidate_metrics=candidate.metrics,
            baseline_metrics=candidate.baseline_metrics,
            production_metrics=production_metrics,
            summary=summary,
        )

    def _check_performance_threshold(
        self, metrics: dict[str, float]
    ) -> CheckResult:
        """Check if model meets minimum performance thresholds."""
        auc = metrics.get("auc", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)

        failures = []
        if auc < self.config.min_auc:
            failures.append(
                f"AUC {auc:.3f} < {self.config.min_auc:.3f}"
            )
        if precision < self.config.min_precision:
            failures.append(
                f"Precision {precision:.3f} < {self.config.min_precision:.3f}"
            )
        if recall < self.config.min_recall:
            failures.append(
                f"Recall {recall:.3f} < {self.config.min_recall:.3f}"
            )

        if failures:
            return CheckResult(
                name="performance_threshold",
                status=CheckStatus.FAIL,
                message="; ".join(failures),
                value=auc,
                threshold=self.config.min_auc,
                details={"auc": auc, "precision": precision, "recall": recall},
            )

        return CheckResult(
            name="performance_threshold",
            status=CheckStatus.PASS,
            message=f"Exceeds minimum thresholds (AUC={auc:.3f})",
            value=auc,
            threshold=self.config.min_auc,
            details={"auc": auc, "precision": precision, "recall": recall},
        )

    def _check_improvement_over_baseline(
        self,
        candidate_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
    ) -> CheckResult:
        """Check if candidate beats baseline by required margin."""
        candidate_auc = candidate_metrics.get("auc", 0)
        baseline_auc = baseline_metrics.get("auc", 0)
        improvement = candidate_auc - baseline_auc

        if improvement < self.config.min_improvement_vs_baseline:
            return CheckResult(
                name="improvement_over_baseline",
                status=CheckStatus.FAIL,
                message=f"Improvement {improvement:.4f} < required {self.config.min_improvement_vs_baseline:.4f}",
                value=improvement,
                threshold=self.config.min_improvement_vs_baseline,
                details={
                    "candidate_auc": candidate_auc,
                    "baseline_auc": baseline_auc,
                },
            )

        status = CheckStatus.PASS if improvement >= 0.02 else CheckStatus.WARNING
        message = (
            f"Beats baseline by {improvement:.4f}"
            if status == CheckStatus.PASS
            else f"Marginal improvement over baseline ({improvement:.4f})"
        )

        return CheckResult(
            name="improvement_over_baseline",
            status=status,
            message=message,
            value=improvement,
            threshold=self.config.min_improvement_vs_baseline,
            details={
                "candidate_auc": candidate_auc,
                "baseline_auc": baseline_auc,
            },
        )

    def _check_improvement_over_production(
        self,
        candidate_metrics: dict[str, float],
        production_metrics: dict[str, float],
    ) -> CheckResult:
        """Check if candidate matches or exceeds production model."""
        candidate_auc = candidate_metrics.get("auc", 0)
        production_auc = production_metrics.get("auc", 0)
        improvement = candidate_auc - production_auc

        if improvement < self.config.min_improvement_vs_production:
            return CheckResult(
                name="improvement_over_production",
                status=CheckStatus.FAIL,
                message=f"Does not match production (delta={improvement:.4f})",
                value=improvement,
                threshold=self.config.min_improvement_vs_production,
                details={
                    "candidate_auc": candidate_auc,
                    "production_auc": production_auc,
                },
            )

        message = (
            f"Improves production by {improvement:.4f}"
            if improvement > 0.01
            else f"Matches production (delta={improvement:.4f})"
        )

        return CheckResult(
            name="improvement_over_production",
            status=CheckStatus.PASS,
            message=message,
            value=improvement,
            threshold=self.config.min_improvement_vs_production,
            details={
                "candidate_auc": candidate_auc,
                "production_auc": production_auc,
            },
        )

    def _check_calibration(
        self, y_true: pd.Series, y_pred_proba: np.ndarray
    ) -> CheckResult:
        """Check if predicted probabilities are well-calibrated."""
        # Compute calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=self.config.calibration_n_bins
            )

            # Compute mean absolute calibration error
            calibration_error = np.mean(
                np.abs(fraction_of_positives - mean_predicted_value)
            )

            if calibration_error > self.config.max_calibration_error:
                return CheckResult(
                    name="calibration",
                    status=CheckStatus.WARNING,
                    message=f"Calibration error {calibration_error:.3f} exceeds {self.config.max_calibration_error:.3f}",
                    value=calibration_error,
                    threshold=self.config.max_calibration_error,
                    details={
                        "mean_error": float(calibration_error),
                        "n_bins": self.config.calibration_n_bins,
                    },
                )

            return CheckResult(
                name="calibration",
                status=CheckStatus.PASS,
                message=f"Well-calibrated (error={calibration_error:.3f})",
                value=calibration_error,
                threshold=self.config.max_calibration_error,
            )

        except Exception as e:
            logger.warning(f"Calibration check failed: {e}")
            return CheckResult(
                name="calibration",
                status=CheckStatus.SKIPPED,
                message=f"Check failed: {e}",
            )

    def _check_prediction_distribution(
        self, y_pred_proba: np.ndarray, y_true: pd.Series
    ) -> CheckResult:
        """Check if prediction distribution is reasonable."""
        mean_pred = np.mean(y_pred_proba)
        actual_churn_rate = y_true.mean()

        # Check if predicted churn rate is within reasonable range of actual
        shift = abs(mean_pred - actual_churn_rate)

        if shift > self.config.max_distribution_shift:
            return CheckResult(
                name="prediction_distribution",
                status=CheckStatus.WARNING,
                message=f"Prediction mean {mean_pred:.3f} differs from actual {actual_churn_rate:.3f} by {shift:.3f}",
                value=shift,
                threshold=self.config.max_distribution_shift,
                details={
                    "mean_prediction": float(mean_pred),
                    "actual_rate": float(actual_churn_rate),
                },
            )

        return CheckResult(
            name="prediction_distribution",
            status=CheckStatus.PASS,
            message=f"Prediction distribution reasonable (mean={mean_pred:.3f}, actual={actual_churn_rate:.3f})",
            value=shift,
            threshold=self.config.max_distribution_shift,
            details={
                "mean_prediction": float(mean_pred),
                "actual_rate": float(actual_churn_rate),
            },
        )

    def _check_feature_stability(
        self,
        current_importances: pd.DataFrame,
        reference_importances: pd.DataFrame,
    ) -> CheckResult:
        """Check if top features are stable across training runs."""
        # Get top N features from both
        current_top = set(
            current_importances.head(self.config.top_n_features)["feature"].tolist()
        )
        reference_top = set(
            reference_importances.head(self.config.top_n_features)["feature"].tolist()
        )

        # Check overlap
        overlap = len(current_top & reference_top)
        overlap_ratio = overlap / self.config.top_n_features

        # Check rank changes for common features
        max_rank_change = 0
        for feature in current_top & reference_top:
            current_rank = (
                current_importances[current_importances["feature"] == feature]
                .index[0]
                + 1
            )
            reference_rank = (
                reference_importances[reference_importances["feature"] == feature]
                .index[0]
                + 1
            )
            rank_change = abs(current_rank - reference_rank)
            max_rank_change = max(max_rank_change, rank_change)

        if max_rank_change > self.config.max_feature_rank_change:
            return CheckResult(
                name="feature_stability",
                status=CheckStatus.WARNING,
                message=f"Feature importance unstable (max rank change={max_rank_change})",
                value=float(max_rank_change),
                threshold=float(self.config.max_feature_rank_change),
                details={
                    "overlap": overlap,
                    "overlap_ratio": overlap_ratio,
                    "max_rank_change": max_rank_change,
                },
            )

        return CheckResult(
            name="feature_stability",
            status=CheckStatus.PASS,
            message=f"Feature importances stable ({overlap}/{self.config.top_n_features} overlap)",
            value=float(overlap),
            threshold=float(self.config.top_n_features),
            details={
                "overlap": overlap,
                "overlap_ratio": overlap_ratio,
                "max_rank_change": max_rank_change,
            },
        )


def print_evaluation_report(result: EvaluationResult) -> None:
    """Print a formatted evaluation report."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Header
    if result.recommendation == "promote":
        header_style = "bold green"
        header_text = "✓ PASSED - Recommend Promotion"
    elif result.recommendation == "review":
        header_style = "bold yellow"
        header_text = "⚠ REVIEW REQUIRED"
    else:
        header_style = "bold red"
        header_text = "✗ FAILED - Do Not Promote"

    console.print(f"\n[{header_style}]{header_text}[/{header_style}]")
    console.print(f"\n{result.summary}\n")

    # Checks table
    table = Table(title="Evaluation Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Message", style="white")

    for check in result.checks:
        status_icon = {
            CheckStatus.PASS: "[green]✓[/green]",
            CheckStatus.FAIL: "[red]✗[/red]",
            CheckStatus.WARNING: "[yellow]⚠[/yellow]",
            CheckStatus.SKIPPED: "[dim]○[/dim]",
        }[check.status]

        table.add_row(check.name, status_icon, check.message)

    console.print(table)

    # Metrics comparison
    if result.baseline_metrics or result.production_metrics:
        console.print("\n[bold]Metrics Comparison:[/bold]")
        metrics_table = Table()
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Candidate", style="green")
        if result.baseline_metrics:
            metrics_table.add_column("Baseline", style="yellow")
        if result.production_metrics:
            metrics_table.add_column("Production", style="blue")

        for metric in ["auc", "precision", "recall", "f1"]:
            row = [metric, f"{result.candidate_metrics.get(metric, 0):.4f}"]
            if result.baseline_metrics:
                row.append(f"{result.baseline_metrics.get(metric, 0):.4f}")
            if result.production_metrics:
                row.append(f"{result.production_metrics.get(metric, 0):.4f}")
            metrics_table.add_row(*row)

        console.print(metrics_table)
