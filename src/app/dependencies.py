"""
FastAPI dependency injection for Retain.

Provides database engine and config as injectable dependencies.
"""

import logging
from functools import lru_cache

from sqlalchemy import Engine

from .config import AppConfig, get_app_config

logger = logging.getLogger(__name__)

_engine: Engine | None = None


@lru_cache
def get_config() -> AppConfig:
    """Return cached application config."""
    return get_app_config()


def get_demo_mode() -> bool:
    """Return whether app is running in demo mode."""
    return get_config().demo_mode


def get_db_engine() -> Engine:
    """Return the SQLAlchemy database engine.

    Uses the engine created at app startup via lifespan.
    Falls back to creating a new engine if needed.
    """
    global _engine
    if _engine is not None:
        return _engine

    try:
        from src.data.database import get_engine

        _engine = get_engine()
        return _engine
    except Exception as e:
        logger.error(f"Failed to get database engine: {e}")
        raise


def set_db_engine(engine: Engine | None) -> None:
    """Set the database engine (used by lifespan and tests)."""
    global _engine
    _engine = engine


@lru_cache
def get_agent_config():
    """Return cached AgentConfig for dependency injection."""
    from src.agents.config import AgentConfig

    return AgentConfig()
