"""
Database connection and management utilities.

Usage:
    python -m src.data.database init    # Create tables
    python -m src.data.database reset   # Drop and recreate tables
    python -m src.data.database check   # Verify connection and counts
"""

import os
import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
import click
from rich.console import Console
from rich.table import Table

# Load environment variables
load_dotenv()

console = Console()


def get_database_url() -> str:
    """Get database URL from environment variables."""
    # Try DATABASE_URL first
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    
    # Fall back to individual components
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "retain_dev")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "password")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine."""
    database_url = get_database_url()
    return create_engine(database_url, echo=False)


def get_session_factory() -> sessionmaker:
    """Create and return a session factory."""
    engine = get_engine()
    return sessionmaker(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_connection():
    """Context manager for raw database connections."""
    engine = get_engine()
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()


def get_schema_path() -> Path:
    """Get path to the SQL schema file."""
    # Look in multiple locations
    possible_paths = [
        Path("sql/schema.sql"),
        Path("retain_db_schema.sql"),
        Path(__file__).parent.parent.parent / "sql" / "schema.sql",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        "Could not find schema.sql. Expected locations:\n"
        + "\n".join(f"  - {p}" for p in possible_paths)
    )


def init_database():
    """Initialize database by creating all tables."""
    console.print("[bold blue]Initializing database...[/bold blue]")
    
    try:
        schema_path = get_schema_path()
        console.print(f"Using schema: {schema_path}")
        
        with open(schema_path) as f:
            schema_sql = f.read()
        
        engine = get_engine()
        with engine.connect() as conn:
            # Execute schema (split by semicolons for multiple statements)
            for statement in schema_sql.split(";"):
                statement = statement.strip()
                if statement and not statement.startswith("--"):
                    conn.execute(text(statement))
            conn.commit()
        
        console.print("[bold green]✓ Database initialized successfully![/bold green]")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error initializing database: {e}[/bold red]")
        sys.exit(1)


def reset_database():
    """Drop all tables and recreate them."""
    console.print("[bold yellow]Resetting database...[/bold yellow]")
    
    drop_sql = """
    DROP TABLE IF EXISTS streaming_events CASCADE;
    DROP TABLE IF EXISTS support_tickets CASCADE;
    DROP TABLE IF EXISTS payments CASCADE;
    DROP TABLE IF EXISTS subscriptions CASCADE;
    DROP TABLE IF EXISTS content_catalog CASCADE;
    DROP TABLE IF EXISTS accounts CASCADE;
    """
    
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(drop_sql))
            conn.commit()
        
        console.print("[yellow]Tables dropped.[/yellow]")
        init_database()
        
    except Exception as e:
        console.print(f"[bold red]Error resetting database: {e}[/bold red]")
        sys.exit(1)


def check_database():
    """Check database connection and show table counts."""
    console.print("[bold blue]Checking database connection...[/bold blue]")
    
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Test connection
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            console.print("[green]✓ Connection successful![/green]")
            
            # Get table counts
            tables = [
                "accounts",
                "subscriptions", 
                "content_catalog",
                "payments",
                "support_tickets",
                "streaming_events",
            ]
            
            table = Table(title="Table Row Counts")
            table.add_column("Table", style="cyan")
            table.add_column("Rows", justify="right", style="green")
            
            total_rows = 0
            for table_name in tables:
                try:
                    result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    )
                    count = result.fetchone()[0]
                    table.add_row(table_name, f"{count:,}")
                    total_rows += count
                except Exception:
                    table.add_row(table_name, "[red]Table not found[/red]")
            
            table.add_row("─" * 20, "─" * 10)
            table.add_row("[bold]Total[/bold]", f"[bold]{total_rows:,}[/bold]")
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[bold red]Error connecting to database: {e}[/bold red]")
        console.print("\n[yellow]Troubleshooting tips:[/yellow]")
        console.print("  1. Ensure PostgreSQL is running")
        console.print("  2. Check your .env file has correct credentials")
        console.print("  3. Ensure the database exists: createdb retain_dev")
        sys.exit(1)


# =============================================================================
# CLI
# =============================================================================

@click.group()
def cli():
    """Database management commands."""
    pass


@cli.command()
def init():
    """Create database tables."""
    init_database()


@cli.command()
def reset():
    """Drop and recreate all tables."""
    reset_database()


@cli.command()
def check():
    """Check database connection and show table counts."""
    check_database()


if __name__ == "__main__":
    cli()
