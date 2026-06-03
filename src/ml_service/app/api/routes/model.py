from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ml_service.app.observability import get_logger
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


@router.post("/retrain")
async def retrain() -> dict[str, str]:
    return {
        "status": "triggered",
        "note": "Training pipeline runs via Airflow DAG (FRAUD-177)",
    }
