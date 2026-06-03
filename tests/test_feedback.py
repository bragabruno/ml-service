from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from ml_service.app.main import app
from ml_service.schemas.investigation import Decision


@pytest.fixture
def feedback_payload() -> dict[str, object]:
    return {
        "transaction_id": "TXN-FB-001",
        "case_id": "CASE-001",
        "original_disposition": "REVIEW",
        "analyst_decision": "DECLINE",
        "feedback_type": "override",
        "notes": "Analyst found additional fraud signals",
    }


@pytest.mark.asyncio
async def test_feedback_returns_accepted(feedback_payload: dict[str, object]) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/feedback", json=feedback_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["recorded"] is True


@pytest.mark.asyncio
async def test_feedback_without_case_id() -> None:
    payload = {
        "transaction_id": "TXN-FB-002",
        "original_disposition": "APPROVE",
        "analyst_decision": "APPROVE",
        "feedback_type": "confirm",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/feedback", json=payload)
    assert resp.status_code == 200
    assert resp.json()["recorded"] is True


@pytest.mark.asyncio
async def test_feedback_invalid_disposition() -> None:
    payload = {
        "transaction_id": "TXN-FB-003",
        "original_disposition": "INVALID",
        "analyst_decision": "APPROVE",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/feedback", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feedback_missing_required_fields() -> None:
    payload = {"transaction_id": "TXN-FB-004"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422


def test_analyst_feedback_schema() -> None:
    from ml_service.schemas.investigation import AnalystFeedback

    fb = AnalystFeedback(
        transaction_id="TXN-FB-005",
        original_disposition=Decision.APPROVE,
        analyst_decision=Decision.REVIEW,
    )
    assert fb.transaction_id == "TXN-FB-005"
    assert fb.feedback_type == "override"
    assert fb.case_id is None
    assert fb.notes is None


def test_analyst_feedback_with_all_fields() -> None:
    from ml_service.schemas.investigation import AnalystFeedback

    fb = AnalystFeedback(
        transaction_id="TXN-FB-006",
        case_id="CASE-006",
        original_disposition=Decision.REVIEW,
        analyst_decision=Decision.DECLINE,
        feedback_type="confirm",
        notes="Confirmed fraud",
    )
    assert fb.case_id == "CASE-006"
    assert fb.feedback_type == "confirm"
    assert fb.notes == "Confirmed fraud"
