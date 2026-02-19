"""
Application configuration for Retain web API.

Reads settings from environment variables via python-dotenv.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Configuration for the Retain FastAPI application."""

    database_url: str = ""
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:5173"]
    )
    demo_mode: bool = False
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


def get_app_config() -> AppConfig:
    """Load application config from environment variables."""
    return AppConfig(
        database_url=os.getenv("DATABASE_URL", ""),
        api_prefix=os.getenv("API_PREFIX", "/api/v1"),
        cors_origins=os.getenv(
            "CORS_ORIGINS", "http://localhost:5173"
        ).split(","),
        demo_mode=os.getenv("DEMO_MODE", "false").lower() == "true",
        debug=os.getenv("DEBUG", "false").lower() == "true",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
