"""
Shared utilities for the Retain agent system.

Formatting, validation, serialization, and context management helpers
used across all agent modules.
"""

import json
import logging
import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from .config import AgentConfig

logger = logging.getLogger("retain.agents.utils")


def format_dataframe_as_markdown(
    df: pd.DataFrame, max_rows: int = 20
) -> str:
    """Convert a DataFrame to a markdown table string.

    Args:
        df: DataFrame to format.
        max_rows: Maximum rows to include before truncation.

    Returns:
        Markdown-formatted table string.
    """
    if df.empty:
        return "*No data available.*"

    truncated = len(df) > max_rows
    display_df = df.head(max_rows)

    # Build markdown table
    headers = "| " + " | ".join(str(c) for c in display_df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display_df.columns) + " |"

    rows = []
    for _, row in display_df.iterrows():
        cells = " | ".join(_format_cell(v) for v in row.values)
        rows.append(f"| {cells} |")

    table = "\n".join([headers, separator, *rows])

    if truncated:
        table += f"\n\n*Showing {max_rows} of {len(df)} rows.*"

    return table


def _format_cell(value: Any) -> str:
    """Format a single cell value for markdown display."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def safe_json_serialize(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types.

    Handles numpy types, datetime, pandas objects, and other common
    non-serializable types encountered in ML pipelines.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "__dict__"):
        return safe_json_serialize(vars(obj))
    return obj


def validate_account_id(account_id: str) -> bool:
    """Check if an account ID matches the expected format ACC_XXXXXXXX.

    Args:
        account_id: Account ID string to validate.

    Returns:
        True if valid format, False otherwise.
    """
    if not account_id:
        return False
    return bool(re.match(r"^ACC_\d{8}$", account_id))


def get_reference_engine(config: AgentConfig | None = None) -> Engine:
    """Create a database engine using project configuration.

    Args:
        config: Agent configuration (unused for engine, but kept for API consistency).

    Returns:
        SQLAlchemy Engine instance.
    """
    from src.data.database import get_engine

    return get_engine()


def truncate_for_context(text: str, max_chars: int = 3000) -> str:
    """Safely truncate long text for LLM context windows.

    Args:
        text: Text to truncate.
        max_chars: Maximum character count.

    Returns:
        Truncated text with indicator if truncation occurred.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated, {len(text) - max_chars} chars omitted]"


def serialize_results(results: dict) -> str:
    """Serialize a results dict to a JSON string for tool output.

    Args:
        results: Dictionary of results.

    Returns:
        JSON-formatted string.
    """
    return json.dumps(safe_json_serialize(results), indent=2)
