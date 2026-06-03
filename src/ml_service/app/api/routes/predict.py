from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from ml_service.app.observability import PREDICTION_COUNT, PREDICTION_LATENCY, SCORE_HISTOGRAM, Timer, get_logger
from ml_service.features.transforms import compute_features, feature_importance_names
from ml_service.schemas.investigation import PredictRequest, PredictResponse
from ml_service.serving.serving_model import get_serving_model

router = APIRouter()

logger = get_logger("predict")


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    serving = get_serving_model()

    if not serving.is_loaded:
        return _rule_based_predict(request)

    with Timer() as timer:
        payload = request.model_dump()
        features = compute_features(payload)
        X = features.reshape(1, -1)
        proba = serving.predict(X)
        score = float(proba[0])

    SCORE_HISTOGRAM.observe(score)
    PREDICTION_LATENCY.observe(timer.elapsed)

    if score >= 0.7:
        risk_level = "HIGH"
        decision = "DECLINE"
    elif score >= 0.4:
        risk_level = "MEDIUM"
        decision = "REVIEW"
    else:
        risk_level = "LOW"
        decision = "APPROVE"

    PREDICTION_COUNT.labels(model_version=serving.version, decision=decision).inc()

    top_indices = list(np.argsort(np.abs(features))[-5:][::-1])
    factors = feature_importance_names(top_indices) if serving.is_loaded else ["NONE"]

    logger.info(
        "prediction",
        txn_id=request.transaction_id,
        score=round(score, 4),
        risk=risk_level,
        latency_ms=round(timer.elapsed * 1000, 2),
        model=serving.version,
    )

    return PredictResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(score, 4),
        risk_level=risk_level,
        model_version=serving.version,
        contributing_factors=factors if factors else ["NONE"],
    )


@router.post("/batch-predict")
async def batch_predict(requests: list[PredictRequest]) -> list[PredictResponse]:
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100")
    return [await predict(r) for r in requests]


def _rule_based_predict(request: PredictRequest) -> PredictResponse:
    risk_score = 0.0
    factors: list[str] = []

    if request.new_device:
        risk_score += 0.25
        factors.append("NEW_DEVICE")
    if request.failed_attempts > 3:
        risk_score += 0.20
        factors.append("HIGH_FAILED_ATTEMPTS")
    if request.amount > 5000:
        risk_score += 0.15
        factors.append("HIGH_AMOUNT")
    if request.country not in ("US", "GB", "CA"):
        risk_score += 0.20
        factors.append("FOREIGN_COUNTRY")

    risk_score = min(risk_score, 1.0)

    if risk_score >= 0.7:
        risk_level = "HIGH"
    elif risk_score >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(risk_score, 4),
        risk_level=risk_level,
        model_version="rules-fallback-v0.1.0",
        contributing_factors=factors if factors else ["NONE"],
    )
