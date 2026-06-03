from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from ml_service.app.observability import get_logger
from ml_service.schemas.investigation import AnalystFeedback, FeedbackResponse

router = APIRouter()
logger = get_logger("feedback")


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(feedback: AnalystFeedback) -> FeedbackResponse:
    logger.info(
        "analyst_feedback",
        txn_id=feedback.transaction_id,
        case_id=feedback.case_id,
        original=feedback.original_disposition.value,
        analyst=feedback.analyst_decision.value,
        feedback_type=feedback.feedback_type,
        notes=feedback.notes,
        timestamp=datetime.now(UTC).isoformat(),
    )
    return FeedbackResponse(status="accepted", recorded=True)
