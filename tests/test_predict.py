from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from ml_service.app.main import app
from ml_service.serving.serving_model import ServingModel


class MidBandModel:
    """Mock model that returns scores in the REVIEW range (0.4-0.7)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        scores = np.full(n, 0.55)
        return np.column_stack([1 - scores, scores])


class LowScoreModel:
    """Mock model that returns scores in the APPROVE range (<0.4)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        scores = np.full(n, 0.10)
        return np.column_stack([1 - scores, scores])


class HighScoreModel:
    """Mock model that returns scores in the DECLINE range (>=0.7)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        scores = np.full(n, 0.95)
        return np.column_stack([1 - scores, scores])


def _make_serving(model: object) -> ServingModel:
    sm = ServingModel()
    sm.load(model, "test-v1")  # type: ignore[arg-type]
    return sm


def _predict_payload() -> dict[str, object]:
    return {
        "transaction_id": "TXN-PRED-001",
        "amount": 5000.00,
        "country": "NG",
        "new_device": True,
        "failed_attempts": 2,
    }


@pytest.mark.asyncio
async def test_predict_mid_band_includes_agent_triage() -> None:
    with patch("ml_service.app.api.routes.predict.get_serving_model", return_value=_make_serving(MidBandModel())):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/predict", json=_predict_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_level"] == "MEDIUM"
    assert body["agent_triage"] is not None
    assert "disposition" in body["agent_triage"]
    assert "summary" in body["agent_triage"]
    assert "evidence" in body["agent_triage"]


@pytest.mark.asyncio
async def test_predict_low_band_no_triage() -> None:
    with patch("ml_service.app.api.routes.predict.get_serving_model", return_value=_make_serving(LowScoreModel())):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/predict", json=_predict_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_level"] == "LOW"
    assert body["agent_triage"] is None


@pytest.mark.asyncio
async def test_predict_high_band_no_triage() -> None:
    with patch("ml_service.app.api.routes.predict.get_serving_model", return_value=_make_serving(HighScoreModel())):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/predict", json=_predict_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_level"] == "HIGH"
    assert body["agent_triage"] is None


@pytest.mark.asyncio
async def test_predict_returns_contributing_factors() -> None:
    with patch("ml_service.app.api.routes.predict.get_serving_model", return_value=_make_serving(LowScoreModel())):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/predict", json=_predict_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert "contributing_factors" in body
    assert len(body["contributing_factors"]) > 0
    assert body["model_version"] == "test-v1"


@pytest.mark.asyncio
async def test_predict_missing_transaction_id() -> None:
    payload = {"amount": 100.0, "country": "US"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_predict_rules_fallback() -> None:
    """When no model is loaded, the rules fallback should return a prediction."""
    sm = ServingModel()
    assert not sm.is_loaded
    with patch("ml_service.app.api.routes.predict.get_serving_model", return_value=sm):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/predict", json=_predict_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "rules-fallback-v0.1.0"
    assert body["fraud_probability"] >= 0.0
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
