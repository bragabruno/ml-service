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


class RunStatus(str, Enum):
    """Lifecycle of an orchestrated retraining run."""

    REQUESTED = "REQUESTED"
    TRAINED = "TRAINED"
    REGISTERED = "REGISTERED"
    GATE_FAILED = "GATE_FAILED"
    FAILED = "FAILED"


class PromotionDecision(BaseModel):
    """Auto-promotion guardrail verdict (FRAUD-117). Eligibility only — never auto-deploys."""

    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    candidate_pr_auc: float | None = None
    incumbent_pr_auc: float | None = None
    requires_human_approval: bool = True


class RetrainingRun(BaseModel):
    """Observable record of a request → train → evaluate → register run (FRAUD-116)."""

    run_id: str
    trigger: TriggerType
    status: RunStatus
    candidate_version: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    gate_passed: bool = False
    gate_violations: list[str] = Field(default_factory=list)
    promotion: PromotionDecision | None = None
    error: str | None = None
    started_at: str
    finished_at: str | None = None
