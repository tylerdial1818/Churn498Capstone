"""Shared fixtures for API endpoint tests."""

import os

import pytest
from fastapi.testclient import TestClient

# Force demo mode before any app imports
os.environ["DEMO_MODE"] = "true"

from src.app.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def app():
    """Create a FastAPI app instance in demo mode."""
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Provide a TestClient for the demo-mode app."""
    with TestClient(app) as c:
        yield c
