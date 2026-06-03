from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from ml_service.app.observability import get_logger
from ml_service.integrations.case_management import get_case_client
from ml_service.schemas.investigation import AnalystFeedback, Decision, FeedbackResponse, LabelType

router = APIRouter()
logger = get_logger("feedback")


def derive_label(analyst_decision: Decision) -> LabelType | None:
    """Map an analyst disposition to a ground-truth label. REVIEW is inconclusive → no label."""
    if analyst_decision is Decision.DECLINE:
        return LabelType.FRAUD
    if analyst_decision is Decision.APPROVE:
        return LabelType.LEGITIMATE
    return None


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(feedback: AnalystFeedback) -> FeedbackResponse:
    label = derive_label(feedback.analyst_decision)
    logger.info(
        "analyst_feedback",
        txn_id=feedback.transaction_id,
        case_id=feedback.case_id,
        original=feedback.original_disposition.value,
        analyst=feedback.analyst_decision.value,
        feedback_type=feedback.feedback_type,
        label=label.value if label else None,
        notes=feedback.notes,
        timestamp=datetime.now(UTC).isoformat(),
    )
    if label is not None and feedback.case_id:
        get_case_client().record_label(feedback.case_id, feedback.transaction_id, label)
    return FeedbackResponse(status="accepted", recorded=True)
