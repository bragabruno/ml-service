from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ml_service.schemas.investigation import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
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
        model_version="stub-v0.1.0",
        contributing_factors=factors if factors else ["NONE"],
    )


@router.post("/batch-predict")
async def batch_predict(requests: list[PredictRequest]) -> list[PredictResponse]:
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100")
    return [await predict(r) for r in requests]
