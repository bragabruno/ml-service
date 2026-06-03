from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/model/version")
async def model_version() -> dict[str, str]:
    return {
        "version": "stub-v0.1.0",
        "status": "stub",
        "framework": "xgboost",
        "note": "Real model loading will be implemented in FRAUD-061",
    }


@router.post("/retrain")
async def retrain() -> dict[str, str]:
    return {
        "status": "not_implemented",
        "note": "Training pipeline will be implemented in FRAUD-071",
    }
