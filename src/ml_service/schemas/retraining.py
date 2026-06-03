from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TriggerType(str, Enum):
    """Why retraining was requested. Each fires independently through the same audited path."""

    DRIFT = "DRIFT"
    LABEL_VOLUME = "LABEL_VOLUME"
    SCHEDULE = "SCHEDULE"


class RetrainingRequestedEvent(BaseModel):
    """Payload published to `fraud.retraining.requested`."""

    request_id: str
    trigger: TriggerType
    reason: str
    requested_at: str
    details: dict[str, Any] = Field(default_factory=dict)
    model_version: str | None = None


class RetrainingDecision(BaseModel):
    """Outcome of a retraining request: emitted, or suppressed by debounce."""

    emitted: bool
    event: RetrainingRequestedEvent | None = None
    suppressed_reason: str | None = None
