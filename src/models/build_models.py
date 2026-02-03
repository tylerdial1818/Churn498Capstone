"""
Command-line interface for model training, evaluation, and deployment.

Usage:
    python -m src.models.build_models train        # Train a new model
    python -m src.models.build_models evaluate     # Evaluate candidate vs production
    python -m src.models.build_models promote      # Promote model to production
    python -m src.models.build_models score        # Run batch scoring
    python -m src.models.build_models monitor      # Run monitoring checks
    python -m src.models.build_models compare      # Compare two models
    python -m src.models.build_models history      # Show model history

Examples:
    # Train XGBoost model with custom experiment name
    python -m src.models.build_models train --model-type xgboost --experiment churn_v2

    # Evaluate candidate model version 3
    python -m src.models.build_models evaluate --version 3

    # Promote model version 3 to production
    python -m src.models.build_models promote --version 3

    # Run batch scoring with production model
    python -m src.models.build_models score

    # Check for drift
    python -m src.models.build_models monitor --reference-version 1
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sqlalchemy import create_engine

from .config import (
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_MONITORING_CONFIG,
    DEFAULT_SCORING_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    EvaluationConfig,
    ModelConfig,
    ModelType,
    MonitoringConfig,
    ScoringConfig,
    TrainingConfig,
)
from .evaluate import EvaluationGate, print_evaluation_report
from .monitoring import ModelMonitor, print_monitoring_report
from .registry import ModelRegistry
from .score import BatchScorer, score_active_accounts
from .train import ModelTrainer

# Load environment variables
load_dotenv()

# Set up rich console
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def get_database_url() -> str:
    """Get database URL from environment."""
    url = os.getenv("DATABASE_URL")
    if not url:
        # Try to construct from individual components
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "retain_dev")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    return url


def load_training_data(engine, reference_date: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and target for training."""
    from src.features import create_training_dataset, FeatureConfig

    console.print("Loading training data from feature pipeline...")
    config = FeatureConfig(reference_date=reference_date)
    X, y = create_training_dataset(engine, config)
    console.print(f"[green]✓ Loaded {len(X)} samples with {len(X.columns)} features[/green]")
    return X, y


def cmd_train(args) -> int:
    """Train a new model."""
    console.print("[bold blue]Training new model...[/bold blue]\n")

    # Build configurations
    model_config = ModelConfig(
        model_type=ModelType(args.model_type),
        random_seed=args.seed,
        cv_folds=args.cv_folds,
    )

    training_config = TrainingConfig(
        experiment_name=args.experiment,
        tracking_uri=args.mlflow_uri,
        compute_shap=not args.no_shap,
    )

    # Connect to database
    database_url = get_database_url()
    engine = create_engine(database_url)

    # Load data
    X, y = load_training_data(engine, args.reference_date)

    # Train model
    trainer = ModelTrainer(model_config, training_config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Training model...", total=None)
        result = trainer.train(X, y)
        progress.update(task, completed=True)

    if not result.success:
        console.print("[red]Training failed:[/red]")
        for error in result.errors:
            console.print(f"  • {error}")
        return 1

    # Print results
    console.print("\n[bold green]✓ Training complete![/bold green]\n")

    metrics_table = Table(title="Model Performance")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Primary Model", style="green")
    if result.baseline_metrics:
        metrics_table.add_column("Baseline", style="yellow")
        metrics_table.add_column("Improvement", style="white")

    for metric in ["auc", "accuracy", "precision", "recall", "f1"]:
        row = [metric, f"{result.metrics.get(metric, 0):.4f}"]
        if result.baseline_metrics:
            baseline_val = result.baseline_metrics.get(metric, 0)
            improvement = result.metrics.get(metric, 0) - baseline_val
            row.append(f"{baseline_val:.4f}")
            row.append(f"{improvement:+.4f}")
        metrics_table.add_row(*row)

    console.print(metrics_table)

    # Cross-validation results
    if result.cv_metrics:
        cv_auc_mean = sum(result.cv_metrics["auc"]) / len(result.cv_metrics["auc"])
        cv_auc_std = pd.Series(result.cv_metrics["auc"]).std()
        console.print(f"\n[bold]Cross-validation AUC:[/bold] {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

    # Top features
    console.print("\n[bold]Top 10 Features:[/bold]")
    for idx, row in result.feature_importances.head(10).iterrows():
        console.print(f"  {idx + 1}. {row['feature']}: {row['importance']:.4f}")

    console.print(f"\n[dim]MLflow run ID: {result.mlflow_run_id}[/dim]")

    # Auto-register if requested
    if args.register:
        console.print("\n[bold]Registering model...[/bold]")
        registry = ModelRegistry()
        metadata = registry.register_model(
            result,
            description=f"{args.model_type} model - AUC {result.metrics['auc']:.4f}",
        )
        console.print(f"[green]✓ Registered as v{metadata.version}[/green]")

    return 0


def cmd_evaluate(args) -> int:
    """Evaluate a candidate model."""
    console.print("[bold blue]Evaluating candidate model...[/bold blue]\n")

    if not args.run_id and not args.version:
        console.print("[red]Error: Must specify either --run-id or --version[/red]")
        return 1

    # Connect to database
    database_url = get_database_url()
    engine = create_engine(database_url)

    # Load test data
    X, y = load_training_data(engine, args.reference_date)

    # Split to get test set (matching training split)
    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load candidate model and get its training result
    # For simplicity, we'll create a minimal TrainingResult from MLflow
    registry = ModelRegistry()

    if args.version:
        console.print(f"Loading model v{args.version}...")
        metadata = registry.get_model_metadata(args.version)
        model = registry.load_model(version=args.version)
        run_id = metadata.run_id
    else:
        console.print(f"Loading model from run {args.run_id}...")
        # This requires the model was registered
        console.print("[yellow]Warning: Evaluation works best with registered models[/yellow]")
        return 1

    # Get metrics from MLflow run
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    # Create a minimal TrainingResult for evaluation
    from .train import TrainingResult

    candidate = TrainingResult(
        model=model,
        baseline_model=None,
        metrics={
            "auc": metrics.get("test_auc", 0),
            "precision": metrics.get("test_precision", 0),
            "recall": metrics.get("test_recall", 0),
            "f1": metrics.get("test_f1", 0),
        },
        baseline_metrics={},
        cv_metrics={},
        feature_importances=pd.DataFrame(),
        mlflow_run_id=run_id,
    )

    # Get production metrics
    production_metrics = registry.get_production_metrics()

    # Run evaluation
    config = EvaluationConfig()
    gate = EvaluationGate(config)
    result = gate.evaluate(candidate, X_test, y_test, production_metrics)

    # Print report
    print_evaluation_report(result)

    return 0 if result.recommendation == "promote" else 1


def cmd_promote(args) -> int:
    """Promote a model to production."""
    console.print(f"[bold blue]Promoting model v{args.version} to {args.stage}...[/bold blue]\n")

    registry = ModelRegistry()

    # Show model info
    metadata = registry.get_model_metadata(args.version)
    console.print(f"Model: v{metadata.version}")
    console.print(f"Current stage: {metadata.stage}")
    console.print(f"AUC: {metadata.metrics.get('test_auc', 'N/A')}")
    console.print(f"Training date: {metadata.training_date}")

    # Confirm if not forced
    if not args.force:
        response = console.input("\n[yellow]Proceed with promotion? (y/n):[/yellow] ")
        if response.lower() != "y":
            console.print("[dim]Promotion cancelled[/dim]")
            return 0

    # Promote
    registry.promote_model(args.version, args.stage, archive_existing=not args.no_archive)

    console.print(f"\n[green]✓ Model v{args.version} promoted to {args.stage}[/green]")

    return 0


def cmd_score(args) -> int:
    """Run batch scoring."""
    console.print("[bold blue]Running batch scoring...[/bold blue]\n")

    # Connect to database
    database_url = get_database_url()
    engine = create_engine(database_url)

    # Configure scoring
    config = ScoringConfig()
    if args.model_version:
        config.use_production_model = False
        config.model_version = args.model_version

    # Run scoring
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating predictions...", total=None)
        result = score_active_accounts(engine, config, args.reference_date)
        progress.update(task, completed=True)

    # Print results
    console.print("\n[bold green]✓ Scoring complete![/bold green]\n")

    summary_table = Table(title="Scoring Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Accounts Scored", f"{result.n_accounts_scored:,}")
    summary_table.add_row("Model Version", str(result.model_version))
    summary_table.add_row("High Risk", f"{result.risk_distribution.get('high', 0):,}")
    summary_table.add_row("Medium Risk", f"{result.risk_distribution.get('medium', 0):,}")
    summary_table.add_row("Low Risk", f"{result.risk_distribution.get('low', 0):,}")

    console.print(summary_table)

    # Show sample of high-risk accounts
    if not result.predictions.empty:
        high_risk = result.get_high_risk_accounts()
        if len(high_risk) > 0:
            console.print("\n[bold]Sample High-Risk Accounts:[/bold]")
            sample = high_risk.head(5)
            for _, row in sample.iterrows():
                console.print(
                    f"  Account {row['account_id']}: "
                    f"churn probability = {row['churn_probability']:.3f}"
                )

    return 0


def cmd_monitor(args) -> int:
    """Run monitoring checks."""
    console.print("[bold blue]Running model monitoring...[/bold blue]\n")

    # Connect to database
    database_url = get_database_url()
    engine = create_engine(database_url)

    # Load current data
    console.print("Loading current data...")
    from src.features import create_training_dataset, FeatureConfig

    current_config = FeatureConfig(reference_date=args.current_date)
    X_current, _ = create_training_dataset(engine, current_config)

    # Load reference data
    console.print("Loading reference data...")
    ref_config = FeatureConfig(reference_date=args.reference_date)
    X_reference, _ = create_training_dataset(engine, ref_config)

    # Run monitoring
    config = MonitoringConfig()
    monitor = ModelMonitor(engine, config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking for drift...", total=None)
        report = monitor.check_drift(X_current, X_reference)
        progress.update(task, completed=True)

    # Print report
    print_monitoring_report(report)

    # Save report
    if args.save:
        monitor.save_monitoring_report(report)
        console.print(f"\n[dim]Report saved to {config.log_table}[/dim]")

    return 0 if report.health_status.value == "healthy" else 1


def cmd_compare(args) -> int:
    """Compare two model versions."""
    console.print(f"[bold blue]Comparing models v{args.version1} vs v{args.version2}...[/bold blue]\n")

    registry = ModelRegistry()
    comparison = registry.compare_models(args.version1, args.version2)

    console.print(comparison.to_string(index=False))

    return 0


def cmd_history(args) -> int:
    """Show model version history."""
    console.print("[bold blue]Model Version History[/bold blue]\n")

    registry = ModelRegistry()
    history = registry.get_model_history(max_results=args.limit)

    if not history:
        console.print("[yellow]No models found in registry[/yellow]")
        return 0

    table = Table()
    table.add_column("Version", style="cyan")
    table.add_column("Stage", style="yellow")
    table.add_column("AUC", justify="right", style="green")
    table.add_column("Training Date", style="white")
    table.add_column("Run ID", style="dim")

    for meta in history:
        table.add_row(
            str(meta.version),
            meta.stage,
            f"{meta.metrics.get('test_auc', 0):.4f}",
            meta.training_date[:10] if meta.training_date else "unknown",
            meta.run_id[:8],
        )

    console.print(table)

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Model training, evaluation, and deployment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--model-type",
        type=str,
        choices=["xgboost", "lightgbm", "logistic_regression"],
        default="xgboost",
        help="Model type to train (default: xgboost)",
    )
    train_parser.add_argument(
        "--experiment",
        type=str,
        default="churn_prediction",
        help="MLflow experiment name",
    )
    train_parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI",
    )
    train_parser.add_argument(
        "--reference-date",
        type=str,
        help="Reference date for feature generation (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    train_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    train_parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP computation",
    )
    train_parser.add_argument(
        "--register",
        action="store_true",
        help="Automatically register model after training",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a candidate model")
    eval_parser.add_argument(
        "--version",
        type=int,
        help="Model version to evaluate",
    )
    eval_parser.add_argument(
        "--run-id",
        type=str,
        help="MLflow run ID to evaluate",
    )
    eval_parser.add_argument(
        "--reference-date",
        type=str,
        help="Reference date for test data",
    )

    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to production")
    promote_parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version to promote",
    )
    promote_parser.add_argument(
        "--stage",
        type=str,
        choices=["Staging", "Production", "Archived"],
        default="Production",
        help="Target stage (default: Production)",
    )
    promote_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    promote_parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Don't archive existing models in target stage",
    )

    # Score command
    score_parser = subparsers.add_parser("score", help="Run batch scoring")
    score_parser.add_argument(
        "--model-version",
        type=int,
        help="Specific model version (default: use production)",
    )
    score_parser.add_argument(
        "--reference-date",
        type=str,
        help="Reference date for feature generation",
    )

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Run monitoring checks")
    monitor_parser.add_argument(
        "--reference-date",
        type=str,
        help="Reference date for reference data",
    )
    monitor_parser.add_argument(
        "--current-date",
        type=str,
        help="Current date for monitoring",
    )
    monitor_parser.add_argument(
        "--save",
        action="store_true",
        help="Save monitoring report to database",
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two model versions")
    compare_parser.add_argument(
        "--version1",
        type=int,
        required=True,
        help="First model version",
    )
    compare_parser.add_argument(
        "--version2",
        type=int,
        required=True,
        help="Second model version",
    )

    # History command
    history_parser = subparsers.add_parser("history", help="Show model version history")
    history_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of versions to show",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 1

    # Route to command handler
    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "promote": cmd_promote,
        "score": cmd_score,
        "monitor": cmd_monitor,
        "compare": cmd_compare,
        "history": cmd_history,
    }

    try:
        return commands[args.command](args)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logging.exception("Command failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
