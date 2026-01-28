"""
Command-line interface for feature engineering pipeline.

Usage:
    python -m src.features.build_features [OPTIONS]

Examples:
    # Generate all features with default settings
    python -m src.features.build_features

    # Generate features for a specific date
    python -m src.features.build_features --reference-date 2025-12-01

    # Generate features and save to specific path
    python -m src.features.build_features --output-path ./data/processed/v2

    # Run specific transformers only
    python -m src.features.build_features --transformers engagement financial
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sqlalchemy import create_engine

from .config import FeatureConfig, FEATURE_SPECS, FeatureType
from .pipeline import FeaturePipeline, PipelineResult
from .transformers import TRANSFORMER_REGISTRY

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


def print_feature_summary(result: PipelineResult) -> None:
    """Print a summary table of generated features."""
    console.print("\n[bold green]✓ Feature generation complete![/bold green]\n")
    
    # Summary stats
    table = Table(title="Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Accounts", f"{result.metadata['n_accounts']:,}")
    table.add_row("Total Features", f"{result.metadata['n_features']:,}")
    table.add_row("Execution Time", f"{result.metadata['execution_time_seconds']:.1f}s")
    table.add_row("Reference Date", result.metadata['reference_date'])
    
    console.print(table)
    
    # Target distribution if available
    if result.validation and "target_distribution" in result.validation.statistics:
        dist = result.validation.statistics["target_distribution"]
        console.print("\n[bold]Target Distribution:[/bold]")
        for label, count in dist.items():
            label_name = "Churned" if label == 1 else "Active"
            console.print(f"  {label_name}: {count:,}")
    
    # Validation results
    if result.validation:
        if result.validation.passed:
            console.print("\n[green]✓ All validation checks passed[/green]")
        else:
            console.print("\n[yellow]⚠ Validation issues found:[/yellow]")
            for issue in result.validation.issues:
                console.print(f"  • {issue}")
        
        if result.validation.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in result.validation.warnings[:5]:  # Limit to 5
                console.print(f"  • {warning}")
            if len(result.validation.warnings) > 5:
                console.print(f"  ... and {len(result.validation.warnings) - 5} more")


def print_feature_catalog() -> None:
    """Print catalog of available features."""
    console.print("\n[bold]Available Features:[/bold]\n")
    
    for feature_type in FeatureType:
        features = [
            spec for spec in FEATURE_SPECS.values()
            if spec.feature_type == feature_type
        ]
        
        if features:
            table = Table(title=f"{feature_type.value.title()} Features")
            table.add_column("Feature", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Source Tables", style="dim")
            
            for spec in features:
                table.add_row(
                    spec.name,
                    spec.description,
                    ", ".join(spec.source_tables),
                )
            
            console.print(table)
            console.print()


def main() -> int:
    """Main entry point for feature generation."""
    parser = argparse.ArgumentParser(
        description="Generate features for churn prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="Reference date for feature calculation (YYYY-MM-DD). Default: today",
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/features",
        help="Output directory for feature files",
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (default: parquet)",
    )
    
    parser.add_argument(
        "--transformers",
        nargs="+",
        choices=list(TRANSFORMER_REGISTRY.keys()),
        default=None,
        help="Specific transformers to run (default: all)",
    )
    
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["30d", "90d"],
        help="Time windows for rolling features (default: 30d 90d)",
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation checks",
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on validation errors",
    )
    
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List all available features and exit",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle list features
    if args.list_features:
        print_feature_catalog()
        return 0
    
    # Build configuration
    config = FeatureConfig(
        reference_date=args.reference_date,
        rolling_windows=args.windows,
        output_format=args.output_format,
        output_path=args.output_path,
        validate_output=not args.no_validate,
        fail_on_validation_error=args.strict,
    )
    
    # Dry run - show configuration and exit
    if args.dry_run:
        console.print("[bold]Dry run - configuration:[/bold]")
        console.print(f"  Reference date: {config.reference_date or 'today'}")
        console.print(f"  Time windows: {config.rolling_windows}")
        console.print(f"  Output path: {config.output_path}")
        console.print(f"  Output format: {config.output_format}")
        console.print(f"  Transformers: {args.transformers or 'all'}")
        console.print(f"  Validation: {not args.no_validate}")
        return 0
    
    try:
        # Connect to database
        database_url = get_database_url()
        console.print(f"Connecting to database...")
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        console.print("[green]✓ Database connected[/green]")
        
        # Create and run pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running feature pipeline...", total=None)
            
            pipeline = FeaturePipeline(
                config=config,
                engine=engine,
                transformers=args.transformers,
            )
            
            result = pipeline.run(include_target=True)
            
            progress.update(task, completed=True)
        
        if not result.success:
            console.print("[red]Pipeline completed with errors:[/red]")
            for error in result.errors:
                console.print(f"  • {error}")
            return 1
        
        # Save results
        output_file = pipeline.save(result)
        console.print(f"\n[bold]Output saved to:[/bold] {output_file}")
        
        # Print summary
        print_feature_summary(result)
        
        return 0
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
