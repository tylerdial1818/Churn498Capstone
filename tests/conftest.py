"""
Shared test fixtures for the Retain test suite.

Provides mock database engines, sample DataFrames matching expected schemas,
and mock LLM fixtures for agent testing.
"""

import os
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Database Fixtures
# =============================================================================


def _try_real_database():
    """Attempt to connect to the real retain_dev database."""
    try:
        from src.data.database import get_engine

        engine = get_engine()
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        return None


@pytest.fixture
def db_engine():
    """Database engine — uses real DB if available, otherwise skips.

    Tests using this fixture should be marked with @pytest.mark.db.
    """
    engine = _try_real_database()
    if engine is None:
        pytest.skip("Database not available")
    return engine


@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine for tests that don't need a real DB."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_accounts_df() -> pd.DataFrame:
    """Sample accounts matching the accounts table schema."""
    return pd.DataFrame({
        "account_id": [f"ACC_{i:08d}" for i in range(1, 6)],
        "email": [f"user{i}@example.com" for i in range(1, 6)],
        "signup_date": [date(2024, 1, 15 + i) for i in range(5)],
        "country": ["US", "UK", "CA", "US", "DE"],
        "age": [25, 34, 42, 28, 55],
        "gender": ["M", "F", "M", "F", "M"],
    })


@pytest.fixture
def sample_risk_scores_df() -> pd.DataFrame:
    """Sample risk scores matching ScoringResult.predictions schema."""
    return pd.DataFrame({
        "account_id": [f"ACC_{i:08d}" for i in range(1, 6)],
        "churn_probability": [0.85, 0.72, 0.55, 0.30, 0.12],
        "risk_tier": ["high", "high", "medium", "low", "low"],
        "model_version": [1, 1, 1, 1, 1],
        "scored_at": [datetime(2025, 12, 1)] * 5,
    })


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Sample feature matrix matching FeaturePipeline output."""
    np.random.seed(42)
    n = 20
    feature_names = [
        "watch_hours_30d", "watch_hours_90d", "login_count_30d",
        "unique_titles_30d", "avg_session_minutes", "payment_failures_90d",
        "support_tickets_30d", "days_since_last_watch", "plan_tier_encoded",
        "account_age_days",
    ]
    data = np.random.rand(n, len(feature_names))
    return pd.DataFrame(
        data,
        columns=feature_names,
        index=[f"ACC_{i:08d}" for i in range(1, n + 1)],
    )


@pytest.fixture
def sample_target_series() -> pd.Series:
    """Sample binary target matching training pipeline output."""
    np.random.seed(42)
    return pd.Series(
        np.random.choice([0, 1], size=20, p=[0.7, 0.3]),
        name="churned",
        index=[f"ACC_{i:08d}" for i in range(1, 21)],
    )


# =============================================================================
# Agent Fixtures
# =============================================================================


@pytest.fixture
def agent_config():
    """Default AgentConfig for testing."""
    from src.agents.config import AgentConfig

    return AgentConfig()


@pytest.fixture
def mock_llm():
    """Mock LLM that returns pre-defined responses.

    Returns a MagicMock that mimics ChatAnthropic behavior.
    Tool calls and content can be configured per test.
    """
    llm = MagicMock()

    # Default: return a response with no tool calls
    response = MagicMock()
    response.content = "Mock LLM response"
    response.tool_calls = []
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm

    return llm


@pytest.fixture
def mock_llm_with_tool_call():
    """Mock LLM that makes one tool call then returns a final response.

    Returns a factory function that can be configured with the tool name and args.
    """

    def _create(tool_name: str, tool_args: dict, final_content: str = "Done"):
        llm = MagicMock()

        # First call: tool call
        tool_response = MagicMock()
        tool_response.tool_calls = [
            {"name": tool_name, "args": tool_args, "id": "call_001"}
        ]
        tool_response.content = ""

        # Second call: final response
        final_response = MagicMock()
        final_response.tool_calls = []
        final_response.content = final_content

        llm.invoke.side_effect = [tool_response, final_response]
        llm.bind_tools.return_value = llm

        return llm

    return _create


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "db: marks tests as requiring database")
    config.addinivalue_line("markers", "llm: marks tests as requiring LLM API key")
