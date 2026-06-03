from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .observability import configure_logging, get_logger, metrics_endpoint

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("startup")
    logger.info("ml-service starting", environment=settings.environment, llm_provider=settings.llm_provider)

    # Load the latest trained model into the serving singleton; /predict falls back to
    # the rules engine if no artifact is present (FRAUD-057 degraded mode).
    try:
        from ml_service.serving.model_registry import get_registry

        registry = get_registry()
        versions = registry.list_versions()
        if versions:
            registry.deploy(versions[-1])
            logger.info("model loaded", version=versions[-1])
        else:
            logger.warning("no trained model found; /predict will use the rules fallback")
    except Exception as exc:  # noqa: BLE001 - model load must never block startup
        logger.warning("model load failed; using rules fallback", error=str(exc))

    app.state.ready = True
    app.state.start_time = time.monotonic()
    logger.info("ml-service ready")
    yield
    logger.info("ml-service shutting down")


def create_app() -> FastAPI:
    get_settings()

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

    from .api.routes import events, feedback, health, investigate, model, predict

    app.include_router(health.router, tags=["health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
    app.include_router(investigate.router, prefix="/api/v1", tags=["investigate"])
    app.include_router(model.router, prefix="/api/v1", tags=["model"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
    app.include_router(events.router, prefix="/api/v1", tags=["events"])

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=metrics_endpoint(), media_type="text/plain")

    return app


app = create_app()
