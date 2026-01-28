#!/usr/bin/env python
"""
Integration test script for the feature engineering pipeline.

This script tests the full pipeline against your PostgreSQL database.
Run this after you have:
1. Set up your PostgreSQL database with the Retain schema
2. Loaded the synthetic data
3. Created a .env file with DATABASE_URL

Usage:
    python scripts/test_integration.py

    # Test specific transformers
    python scripts/test_integration.py --transformers engagement financial

    # Test with a sample of accounts
    python scripts/test_integration.py --sample 1000
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sqlalchemy import create_engine, text

console = Console()


def check_database_connection(engine) -> bool:
    """Verify database connection and check for required tables."""
    console.print("\n[bold]1. Checking database connection...[/bold]")
    
    required_tables = [
        "accounts", "subscriptions", "content_catalog",
        "payments", "support_tickets", "streaming_events"
    ]
    
    try:
        with engine.connect() as conn:
            # Test connection
            conn.execute(text("SELECT 1"))
            console.print("   [green]✓ Connection successful[/green]")
            
            # Check tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            existing_tables = {row[0] for row in result}
            
            missing = set(required_tables) - existing_tables
            if missing:
                console.print(f"   [red]✗ Missing tables: {missing}[/red]")
                return False
            
            console.print("   [green]✓ All required tables exist[/green]")
            
            # Show row counts
            table = Table(title="Table Row Counts")
            table.add_column("Table", style="cyan")
            table.add_column("Rows", style="green", justify="right")
            
            for tbl in required_tables:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                table.add_row(tbl, f"{count:,}")
            
            console.print(table)
            return True
            
    except Exception as e:
        console.print(f"   [red]✗ Connection failed: {e}[/red]")
        return False


def test_individual_transformers(engine, transformers: list[str] | None = None) -> dict:
    """Test each transformer individually."""
    from src.features.config import TIME_WINDOWS
    from src.features.transformers import TRANSFORMER_REGISTRY, TransformerContext
    
    console.print("\n[bold]2. Testing individual transformers...[/bold]")
    
    transformers = transformers or list(TRANSFORMER_REGISTRY.keys())
    time_windows = [TIME_WINDOWS["30d"], TIME_WINDOWS["90d"]]
    reference_date = date(2025, 12, 1)  # Use a date within your data range
    
    context = TransformerContext(
        engine=engine,
        reference_date=reference_date,
        time_windows=time_windows,
        account_ids=None,
    )
    
    results = {}
    
    for name in transformers:
        if name not in TRANSFORMER_REGISTRY:
            console.print(f"   [yellow]⚠ Unknown transformer: {name}[/yellow]")
            continue
        
        try:
            transformer_class = TRANSFORMER_REGISTRY[name]
            transformer = transformer_class(context)
            df = transformer.transform()
            
            results[name] = {
                "success": True,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
            }
            
            console.print(
                f"   [green]✓ {name}:[/green] "
                f"{len(df):,} rows, {len(df.columns)} columns"
            )
            
        except Exception as e:
            results[name] = {
                "success": False,
                "error": str(e),
            }
            console.print(f"   [red]✗ {name}: {e}[/red]")
    
    return results


def test_full_pipeline(engine, sample_size: int | None = None) -> dict:
    """Test the full pipeline end-to-end."""
    from src.features import FeaturePipeline, FeatureConfig
    
    console.print("\n[bold]3. Testing full pipeline...[/bold]")
    
    config = FeatureConfig(
        reference_date="2025-12-01",
        rolling_windows=["30d", "90d"],
        validate_output=True,
    )
    
    pipeline = FeaturePipeline(config, engine)
    
    # Optionally sample accounts for faster testing
    account_ids = None
    if sample_size:
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT account_id FROM accounts 
                ORDER BY RANDOM() 
                LIMIT {sample_size}
            """))
            account_ids = [row[0] for row in result]
        console.print(f"   Sampling {sample_size} accounts for testing...")
    
    try:
        result = pipeline.run(account_ids=account_ids, include_target=True)
        
        console.print(f"   [green]✓ Pipeline completed successfully[/green]")
        console.print(f"   • Accounts: {len(result.features):,}")
        console.print(f"   • Features: {len(result.features.columns)}")
        console.print(f"   • Execution time: {result.metadata['execution_time_seconds']:.1f}s")
        
        if result.validation:
            if result.validation.passed:
                console.print(f"   [green]✓ Validation passed[/green]")
            else:
                console.print(f"   [yellow]⚠ Validation issues:[/yellow]")
                for issue in result.validation.issues[:3]:
                    console.print(f"     • {issue}")
        
        return {
            "success": True,
            "n_accounts": len(result.features),
            "n_features": len(result.features.columns),
            "execution_time": result.metadata["execution_time_seconds"],
            "validation_passed": result.validation.passed if result.validation else None,
        }
        
    except Exception as e:
        console.print(f"   [red]✗ Pipeline failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def test_convenience_functions(engine) -> dict:
    """Test the convenience API functions."""
    console.print("\n[bold]4. Testing convenience functions...[/bold]")
    
    results = {}
    
    # Test create_training_dataset
    try:
        from src.features import create_training_dataset, FeatureConfig
        
        # Use small sample for speed
        config = FeatureConfig(reference_date="2025-12-01")
        
        # Get a small sample of account IDs
        with engine.connect() as conn:
            sample_ids = [
                row[0] for row in 
                conn.execute(text("SELECT account_id FROM accounts LIMIT 100"))
            ]
        
        # Note: create_training_dataset doesn't support account_ids filter,
        # so we test the pipeline directly
        from src.features import FeaturePipeline
        pipeline = FeaturePipeline(config, engine)
        result = pipeline.run(account_ids=sample_ids, include_target=True)
        X = result.features
        y = result.target["churned"] if "churned" in result.target.columns else None
        
        console.print(f"   [green]✓ create_training_dataset:[/green] X={X.shape}, y={y.shape if y is not None else 'None'}")
        results["create_training_dataset"] = {"success": True, "shape": X.shape}
        
    except Exception as e:
        console.print(f"   [red]✗ create_training_dataset: {e}[/red]")
        results["create_training_dataset"] = {"success": False, "error": str(e)}
    
    # Test create_inference_features
    try:
        from src.features import create_inference_features
        
        # Get some account IDs
        with engine.connect() as conn:
            account_ids = [
                row[0] for row in 
                conn.execute(text("SELECT account_id FROM accounts LIMIT 10"))
            ]
        
        features = create_inference_features(engine, account_ids)
        
        console.print(f"   [green]✓ create_inference_features:[/green] {features.shape}")
        results["create_inference_features"] = {"success": True, "shape": features.shape}
        
    except Exception as e:
        console.print(f"   [red]✗ create_inference_features: {e}[/red]")
        results["create_inference_features"] = {"success": False, "error": str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Integration tests for feature pipeline")
    parser.add_argument(
        "--transformers", 
        nargs="+", 
        help="Test specific transformers only"
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=None,
        help="Sample size for full pipeline test (default: all accounts)"
    )
    parser.add_argument(
        "--skip-full", 
        action="store_true",
        help="Skip full pipeline test (faster)"
    )
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    console.print("[bold blue]═══ Feature Pipeline Integration Tests ═══[/bold blue]")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "retain_dev")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
    
    # Create engine
    try:
        engine = create_engine(database_url)
    except Exception as e:
        console.print(f"[red]Failed to create database engine: {e}[/red]")
        console.print("\nMake sure you have a .env file with DATABASE_URL or individual DB_* variables.")
        return 1
    
    # Run tests
    all_passed = True
    
    # 1. Check database
    if not check_database_connection(engine):
        console.print("\n[red]Database check failed. Please ensure your database is set up correctly.[/red]")
        return 1
    
    # 2. Test individual transformers
    transformer_results = test_individual_transformers(engine, args.transformers)
    if any(not r["success"] for r in transformer_results.values()):
        all_passed = False
    
    # 3. Test full pipeline
    if not args.skip_full:
        pipeline_result = test_full_pipeline(engine, args.sample)
        if not pipeline_result["success"]:
            all_passed = False
    
    # 4. Test convenience functions
    convenience_results = test_convenience_functions(engine)
    if any(not r["success"] for r in convenience_results.values()):
        all_passed = False
    
    # Summary
    console.print("\n[bold]═══ Summary ═══[/bold]")
    if all_passed:
        console.print("[bold green]All tests passed! ✓[/bold green]")
        return 0
    else:
        console.print("[bold red]Some tests failed. See details above.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
