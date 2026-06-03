from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ml_service.app.observability import get_logger
from ml_service.events.topics import Topic
from ml_service.schemas.retraining import RetrainingDecision, RetrainingRequestedEvent, TriggerType

if TYPE_CHECKING:
    from ml_service.events.producer import EventProducer
    from ml_service.retraining.audit import RetrainingAuditLedger

logger = get_logger("retraining_triggers")

DEFAULT_DEBOUNCE = timedelta(hours=24)


def drift_triggers_retraining(drift_result: dict[str, Any]) -> bool:
    """True when feature drift was detected (output of `evaluate_drift`)."""
    return bool(drift_result.get("drift_detected", False))


def label_volume_triggers_retraining(new_label_count: int, threshold: int) -> bool:
    """True when enough new analyst labels have accumulated to justify a refresh."""
    return new_label_count >= threshold


def request_retraining(
    trigger: TriggerType,
    reason: str,
    details: dict[str, Any],
    *,
    producer: EventProducer,
    ledger: RetrainingAuditLedger,
    model_version: str | None = None,
    debounce: timedelta = DEFAULT_DEBOUNCE,
    now: datetime | None = None,
) -> RetrainingDecision:
    """Request retraining: debounce per-trigger, emit `fraud.retraining.requested`, audit always.

    Every attempt is written to the audit ledger (emitted or suppressed). A request is suppressed
    when an *emitted* request for the same trigger already exists within the debounce window — so
    each trigger type debounces independently and storms are avoided.
    """
    moment = now or datetime.now(UTC)

    recent = ledger.recent(trigger.value, debounce, moment)
    if recent:
        ledger.append(
            {
                "request_id": str(uuid.uuid4()),
                "trigger": trigger.value,
                "reason": reason,
                "requested_at": moment.isoformat(),
                "emitted": False,
                "suppressed_reason": "debounced",
            }
        )
        logger.info("retraining_suppressed", trigger=trigger.value, reason="debounced")
        return RetrainingDecision(emitted=False, suppressed_reason="debounced")

    event = RetrainingRequestedEvent(
        request_id=str(uuid.uuid4()),
        trigger=trigger,
        reason=reason,
        requested_at=moment.isoformat(),
        details=details,
        model_version=model_version,
    )
    producer.publish(Topic.FRAUD_RETRAINING_REQUESTED.value, event.model_dump(mode="json"))
    ledger.append({**event.model_dump(mode="json"), "emitted": True})
    logger.info("retraining_requested", trigger=trigger.value, request_id=event.request_id, reason=reason)
    return RetrainingDecision(emitted=True, event=event)
