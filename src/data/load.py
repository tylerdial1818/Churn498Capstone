"""
Data loading utilities for importing CSV files into PostgreSQL.

Usage:
    python -m src.data.load                    # Load all tables
    python -m src.data.load --table accounts   # Load specific table
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sqlalchemy import text

from src.data.database import get_engine

console = Console()

# Table loading order (respects foreign key constraints)
LOAD_ORDER = [
    "accounts",
    "content_catalog",
    "subscriptions",
    "payments",
    "support_tickets",
    "streaming_events",
]

# CSV filename mapping
CSV_FILES = {
    "accounts": "retain_accounts.csv",
    "content_catalog": "retain_content_catalog.csv",
    "subscriptions": "retain_subscriptions.csv",
    "payments": "retain_payments.csv",
    "support_tickets": "retain_support_tickets.csv",
    "streaming_events": "retain_streaming_events.csv",
}


def find_data_directory() -> Path:
    """Find the directory containing CSV files."""
    possible_paths = [
        Path("data/raw"),
        Path("data"),
        Path("."),
        Path(__file__).parent.parent.parent / "data" / "raw",
    ]
    
    for path in possible_paths:
        if path.exists():
            # Check if any CSV files exist here
            csv_files = list(path.glob("retain_*.csv"))
            if csv_files:
                return path
    
    raise FileNotFoundError(
        "Could not find CSV data files. Expected locations:\n"
        + "\n".join(f"  - {p}" for p in possible_paths)
        + "\n\nPlease ensure CSV files are in one of these directories."
    )


def load_table(table_name: str, data_dir: Path, engine, truncate: bool = True):
    """Load a single table from CSV."""
    csv_file = CSV_FILES.get(table_name)
    if not csv_file:
        raise ValueError(f"Unknown table: {table_name}")
    
    csv_path = data_dir / csv_file
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Truncate existing data if requested
    if truncate:
        with engine.connect() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
            conn.commit()
    
    # Load data using pandas to_sql with chunking for large tables
    chunk_size = 10000 if len(df) > 50000 else None
    
    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        chunksize=chunk_size,
        method="multi",
    )
    
    return len(df)


def load_all_tables(data_dir: Optional[Path] = None, truncate: bool = True):
    """Load all tables in the correct order."""
    if data_dir is None:
        data_dir = find_data_directory()
    
    console.print(f"[bold blue]Loading data from: {data_dir}[/bold blue]")
    
    engine = get_engine()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        overall_task = progress.add_task(
            "[cyan]Loading tables...", total=len(LOAD_ORDER)
        )
        
        for table_name in LOAD_ORDER:
            progress.update(
                overall_task, 
                description=f"[cyan]Loading {table_name}..."
            )
            
            try:
                row_count = load_table(table_name, data_dir, engine, truncate)
                console.print(f"  [green]✓[/green] {table_name}: {row_count:,} rows")
            except FileNotFoundError as e:
                console.print(f"  [yellow]⚠[/yellow] {table_name}: {e}")
            except Exception as e:
                console.print(f"  [red]✗[/red] {table_name}: {e}")
                raise
            
            progress.advance(overall_task)
    
    console.print("\n[bold green]✓ Data loading complete![/bold green]")


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    "--table", "-t",
    type=click.Choice(LOAD_ORDER),
    help="Load only a specific table"
)
@click.option(
    "--data-dir", "-d",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing CSV files"
)
@click.option(
    "--no-truncate",
    is_flag=True,
    help="Append to existing data instead of truncating"
)
def main(table: Optional[str], data_dir: Optional[Path], no_truncate: bool):
    """Load CSV data into PostgreSQL database."""
    truncate = not no_truncate
    
    try:
        if data_dir is None:
            data_dir = find_data_directory()
        
        if table:
            console.print(f"[bold blue]Loading table: {table}[/bold blue]")
            engine = get_engine()
            row_count = load_table(table, data_dir, engine, truncate)
            console.print(f"[green]✓ Loaded {row_count:,} rows into {table}[/green]")
        else:
            load_all_tables(data_dir, truncate)
            
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error loading data: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
