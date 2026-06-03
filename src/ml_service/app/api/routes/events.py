from __future__ import annotations

from fastapi import APIRouter

from ml_service.agent.case_integration import handle_review_required
from ml_service.agent.llm.factory import get_llm_client
from ml_service.integrations.case_management import get_case_client
from ml_service.schemas.investigation import InvestigationReport, ReviewRequiredEvent

router = APIRouter()


@router.post("/events/review-required", response_model=InvestigationReport)
async def review_required(event: ReviewRequiredEvent) -> InvestigationReport:
    """Offline/test ingress for `fraud.review.required`: auto-draft + attach a report.

    The production path is the Kafka consumer (ml_service.events.consumer), which calls the
    same `handle_review_required` core.
    """
    return handle_review_required(event, get_llm_client(), get_case_client())
