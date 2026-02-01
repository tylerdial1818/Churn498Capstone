"""
Analytics export for ad-hoc reporting.

This module produces a single, analyst-friendly CSV that combines:
- Account identifiers and metadata (email, signup date, plan, status)
- All engineered features
- Target variable and churn details

The output file uses a stable filename (retain_analytics.csv) so it
can be overwritten on each pipeline run, keeping notebooks and reports
pointed at a consistent path.

Usage:
    # As part of the pipeline CLI
    python -m src.features.build_features --export-analytics

    # Programmatically
    from src.features.export import export_analytics_csv
    path = export_analytics_csv(engine, result)
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .pipeline import PipelineResult

logger = logging.getLogger(__name__)

# Stable filename — intentionally not timestamped so notebooks
# and reports can reference it without updating paths.
ANALYTICS_FILENAME = "retain_analytics.csv"

# Timestamped snapshots go here for historical comparison.
SNAPSHOT_DIR = "snapshots"


def _fetch_account_metadata(engine: Engine) -> pd.DataFrame:
    """
    Pull human-readable account context from the database.

    These columns are NOT model features — they exist so an analyst
    can filter, slice, and identify accounts in the exported CSV.
    """
    query = """
    SELECT
        a.account_id,
        a.email,
        a.signup_date,
        a.country,
        a.age    AS account_age,
        a.gender,
        s.plan_type,
        s.status AS subscription_status,
        s.start_date AS subscription_start,
        s.end_date   AS subscription_end
    FROM accounts a
    JOIN subscriptions s ON a.account_id = s.account_id
    ORDER BY a.account_id
    """

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn).set_index("account_id")


def build_analytics_dataframe(
    engine: Engine,
    result: PipelineResult,
) -> pd.DataFrame:
    """
    Combine account metadata, features, and target into one DataFrame.

    Column order:
        1. Account identifiers & metadata  (who is this?)
        2. Target / churn details           (what happened?)
        3. Engineered features              (why did it happen?)

    Args:
        engine: Database connection for metadata query.
        result: Completed PipelineResult from the feature pipeline.

    Returns:
        A single DataFrame ready for CSV export.
    """
    logger.info("Building analytics dataset")

    # 1. Account metadata
    metadata = _fetch_account_metadata(engine)

    # 2. Target columns
    target = result.target if not result.target.empty else pd.DataFrame()

    # 3. Engineered features
    features = result.features

    # Merge: start with metadata, layer on target, then features.
    # Left join from metadata keeps every account even if a transformer
    # returned no rows for it (e.g. accounts with zero streaming events).
    analytics = metadata.join(target, how="left").join(features, how="left")

    # Sort for stable, reproducible output
    analytics = analytics.sort_index()

    logger.info(
        f"Analytics dataset built: "
        f"{len(analytics):,} rows × {len(analytics.columns)} columns"
    )

    return analytics


def export_analytics_csv(
    engine: Engine,
    result: PipelineResult,
    output_dir: str | Path = "data/processed",
    save_snapshot: bool = True,
) -> Path:
    """
    Export the analytics CSV to data/processed/.

    Creates two files:
        data/processed/retain_analytics.csv          ← stable path
        data/processed/snapshots/retain_analytics_20250601_143022.csv
                                                      ← timestamped backup

    The stable file is always overwritten so notebooks and reports
    can hard-code the path.  Snapshots let you compare across runs.

    Args:
        engine:         Database connection.
        result:         Completed PipelineResult.
        output_dir:     Base directory (default: data/processed).
        save_snapshot:  Also write a timestamped copy (default: True).

    Returns:
        Path to the stable CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analytics = build_analytics_dataframe(engine, result)

    # --- stable file (overwrite) ---
    stable_path = output_dir / ANALYTICS_FILENAME
    analytics.to_csv(stable_path)
    logger.info(f"Wrote analytics CSV → {stable_path}")

    # --- timestamped snapshot ---
    if save_snapshot:
        snapshot_dir = output_dir / SNAPSHOT_DIR
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"retain_analytics_{timestamp}.csv"
        snapshot_path = snapshot_dir / snapshot_name
        analytics.to_csv(snapshot_path)
        logger.info(f"Wrote snapshot   → {snapshot_path}")

    # --- summary to stdout ---
    _log_export_summary(analytics, stable_path)

    return stable_path


def _log_export_summary(df: pd.DataFrame, path: Path) -> None:
    """Log a human-readable summary of the exported dataset."""
    size_mb = path.stat().st_size / (1024 * 1024)

    churn_col = "churned"
    if churn_col in df.columns:
        n_churned = int(df[churn_col].sum())
        n_active = len(df) - n_churned
        churn_rate = n_churned / len(df) * 100
        target_summary = (
            f"  Active: {n_active:,}  |  Churned: {n_churned:,}  "
            f"({churn_rate:.1f}% churn rate)"
        )
    else:
        target_summary = "  Target column not present"

    logger.info(
        f"\n{'─' * 50}\n"
        f"  Analytics Export Summary\n"
        f"{'─' * 50}\n"
        f"  File:    {path}\n"
        f"  Size:    {size_mb:.1f} MB\n"
        f"  Rows:    {len(df):,}\n"
        f"  Columns: {len(df.columns)}\n"
        f"{target_summary}\n"
        f"{'─' * 50}"
    )
