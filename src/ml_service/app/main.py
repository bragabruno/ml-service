from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("ml-service starting", environment=settings.environment, llm_provider=settings.llm_provider)
    app.state.ready = True
    app.state.start_time = time.monotonic()
    logger.info("ml-service ready")
    yield
    logger.info("ml-service shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ml-service",
        description="AI/ML plane for the Fraud Prevention Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from .api.routes import health, predict, investigate, model

    app.include_router(health.router, tags=["health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
    app.include_router(investigate.router, prefix="/api/v1", tags=["investigate"])
    app.include_router(model.router, prefix="/api/v1", tags=["model"])

    return app


app = create_app()
