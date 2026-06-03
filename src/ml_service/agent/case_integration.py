from __future__ import annotations

from typing import TYPE_CHECKING

from ml_service.agent.investigation_agent import investigate
from ml_service.schemas.investigation import InvestigateRequest, InvestigationReport, ReviewRequiredEvent

if TYPE_CHECKING:
    from ml_service.agent.llm.base import LLMClient
    from ml_service.integrations.case_management import CaseManagementClient


def handle_review_required(
    event: ReviewRequiredEvent,
    llm: LLMClient,
    case_client: CaseManagementClient,
) -> InvestigationReport:
    """Auto-draft an investigation report for a `fraud.review.required` case and attach it.

    Transport-agnostic core shared by the HTTP endpoint and the Kafka consumer. PII redaction
    and prompt-injection screening already happen inside `investigate()`.
    """
    inv_request = InvestigateRequest(
        transaction_id=event.transaction_id,
        case_id=event.case_id,
        amount=event.amount,
        currency=event.currency,
        country=event.country,
        new_device=event.new_device,
        failed_attempts=event.failed_attempts,
        user_id=event.user_id,
        merchant_id=event.merchant_id,
        device_id=event.device_id,
        untrusted_notes=event.untrusted_notes,
    )
    report, _trace = investigate(inv_request, llm)
    case_client.attach_report(event.case_id, report)
    return report
