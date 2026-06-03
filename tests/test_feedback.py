from __future__ import annotations

from ml_service.schemas.investigation import (
    AnalystFeedback,
    Decision,
    FeedbackResponse,
)


def test_feedback_schema_roundtrip() -> None:
    fb = AnalystFeedback(
        transaction_id="TXN-FB-001",
        case_id="CASE-001",
        original_disposition=Decision.REVIEW,
        analyst_decision=Decision.DECLINE,
        feedback_type="override",
        notes="Analyst found additional fraud signals",
    )
    assert fb.transaction_id == "TXN-FB-001"
    assert fb.case_id == "CASE-001"
    assert fb.original_disposition == Decision.REVIEW
    assert fb.analyst_decision == Decision.DECLINE
    assert fb.feedback_type == "override"
    assert fb.notes == "Analyst found additional fraud signals"


def test_feedback_without_case_id() -> None:
    fb = AnalystFeedback(
        transaction_id="TXN-FB-002",
        original_disposition=Decision.APPROVE,
        analyst_decision=Decision.APPROVE,
        feedback_type="confirm",
    )
    assert fb.case_id is None
    assert fb.notes is None


def test_feedback_response_schema() -> None:
    resp = FeedbackResponse(status="accepted", recorded=True)
    assert resp.status == "accepted"
    assert resp.recorded is True


def test_decision_enum_values() -> None:
    assert Decision.APPROVE.value == "APPROVE"
    assert Decision.REVIEW.value == "REVIEW"
    assert Decision.DECLINE.value == "DECLINE"
