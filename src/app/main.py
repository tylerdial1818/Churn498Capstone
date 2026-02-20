"""
FastAPI application factory for Retain.

Creates the app with lifespan, CORS, routers, and error handlers.

Usage:
    uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
    DEMO_MODE=true uvicorn src.app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_app_config
from .dependencies import set_db_engine
from .routers import accounts, agents, analytics, dashboard, interventions, prescriptions
from .routes import (
    analytics as agent_analytics,
    at_risk as agent_at_risk,
    dashboard as agent_dashboard,
    interventions as agent_interventions,
    prescriptions as agent_prescriptions,
)
from .schemas import ErrorResponse, HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Acquire database engine on startup, release on shutdown."""
    config = get_app_config()

    if not config.demo_mode:
        try:
            from src.data.database import get_engine

            engine = get_engine()
            set_db_engine(engine)
            logger.info("Database engine initialized")
        except Exception as e:
            logger.warning(f"Database unavailable, falling back to demo mode: {e}")
    else:
        logger.info("Running in demo mode — using fixture data")

    yield

    set_db_engine(None)
    logger.info("Database engine released")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_app_config()

    app = FastAPI(
        title="Retain API",
        description="Customer churn prediction and prevention platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    prefix = config.api_prefix
    app.include_router(dashboard.router, prefix=prefix)
    app.include_router(accounts.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)
    app.include_router(prescriptions.router, prefix=prefix)
    app.include_router(interventions.router, prefix=prefix)
    app.include_router(agents.router, prefix=prefix)

    # Agent-backed routes (under /api, separate from /api/v1)
    agent_prefix = "/api"
    app.include_router(agent_dashboard.router, prefix=agent_prefix)
    app.include_router(agent_at_risk.router, prefix=agent_prefix)
    app.include_router(agent_analytics.router, prefix=agent_prefix)
    app.include_router(agent_prescriptions.router, prefix=agent_prefix)
    app.include_router(agent_interventions.router, prefix=agent_prefix)

    # Health check
    @app.get(f"{prefix}/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        db_ok = False
        if not config.demo_mode:
            try:
                from .dependencies import get_db_engine

                engine = get_db_engine()
                with engine.connect() as conn:
                    conn.execute(__import__("sqlalchemy").text("SELECT 1"))
                db_ok = True
            except Exception:
                pass

        return HealthResponse(
            status="ok",
            demo_mode=config.demo_mode,
            db_connected=db_ok,
        )

    # Exception handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="Not Found",
                detail=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(422)
    async def validation_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="Validation Error",
                detail=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(500)
    async def server_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Internal server error: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                detail=str(exc) if config.debug else None,
            ).model_dump(),
        )

    return app


app = create_app()
