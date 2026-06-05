from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ml_service.app.observability import get_logger
from ml_service.schemas.retraining import RetrainingRun
from ml_service.serving.model_registry import get_registry
from ml_service.serving.serving_model import get_serving_model

router = APIRouter()

logger = get_logger("model")


@router.get("/model/version")
async def model_version() -> dict[str, Any]:
    serving = get_serving_model()
    registry = get_registry()
    versions = registry.list_versions()
    return {
        "active_version": serving.version,
        "model_loaded": str(serving.is_loaded),
        "shadow_version": serving.shadow_version,
        "available_versions": versions,
        "framework": "xgboost",
    }


@router.post("/model/deploy/{version}")
async def deploy_model(version: str) -> dict[str, str]:
    registry = get_registry()
    serving = get_serving_model()
    try:
        old = registry.deploy(version, serving)
        logger.info("model_deployed", version=version, previous=old)
        return {"status": "deployed", "version": version, "previous": old}
    except FileNotFoundError as e:
        return {"status": "error", "detail": str(e)}


@router.post("/model/shadow/{version}")
async def load_shadow(version: str) -> dict[str, str]:
    """Load a registered version into the challenger (shadow) slot — scored alongside the
    champion on /predict for divergence telemetry; never affects responses."""
    registry = get_registry()
    serving = get_serving_model()
    try:
        model, metadata = registry.load(version)
    except FileNotFoundError as e:
        return {"status": "error", "detail": str(e)}
    serving.load_shadow(model, version, metadata=metadata)
    logger.info("shadow_loaded", version=version)
    return {"status": "shadow_loaded", "version": version}


@router.delete("/model/shadow")
async def clear_shadow() -> dict[str, str]:
    old = get_serving_model().clear_shadow()
    logger.info("shadow_cleared", previous=old)
    return {"status": "shadow_cleared", "previous": old}


@router.post("/retrain", response_model=RetrainingRun)
async def retrain() -> RetrainingRun:
    """Trigger a tracked retraining run (train → evaluate → register candidate + promotion check).

    Registers a candidate only; promotion to production stays a human action via
    `POST /model/deploy/{version}` (FRAUD-117 / FRAUD-080).
    """
    from ml_service.retraining.orchestrator import run_retraining
    from ml_service.schemas.retraining import TriggerType

    return run_retraining(trigger=TriggerType.SCHEDULE, reason="manual /retrain")
