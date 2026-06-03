from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from ml_service.events.producer import MockEventProducer
from ml_service.events.topics import Topic
from ml_service.retraining.audit import RetrainingAuditLedger
from ml_service.retraining.triggers import (
    drift_triggers_retraining,
    label_volume_triggers_retraining,
    request_retraining,
)
from ml_service.schemas.retraining import TriggerType

if TYPE_CHECKING:
    from pathlib import Path

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_WINDOW = timedelta(hours=24)


def _ledger(tmp_path: Path) -> RetrainingAuditLedger:
    return RetrainingAuditLedger(tmp_path / "audit.jsonl")


def test_drift_trigger_evaluator() -> None:
    assert drift_triggers_retraining({"drift_detected": True}) is True
    assert drift_triggers_retraining({"drift_detected": False}) is False
    assert drift_triggers_retraining({}) is False


def test_label_volume_trigger_evaluator() -> None:
    assert label_volume_triggers_retraining(100, threshold=100) is True
    assert label_volume_triggers_retraining(99, threshold=100) is False


def test_request_retraining_emits_and_audits(tmp_path: Path) -> None:
    producer = MockEventProducer()
    ledger = _ledger(tmp_path)
    decision = request_retraining(
        TriggerType.DRIFT,
        reason="psi high",
        details={"psi": 0.4},
        producer=producer,
        ledger=ledger,
        now=_T0,
    )
    assert decision.emitted is True
    assert decision.event is not None
    assert len(producer.published) == 1
    assert producer.published[0].topic == Topic.FRAUD_RETRAINING_REQUESTED.value
    # Exactly one emitted record audited.
    assert len(ledger.recent(TriggerType.DRIFT.value, _WINDOW, _T0)) == 1


def test_request_retraining_debounced_within_window(tmp_path: Path) -> None:
    producer = MockEventProducer()
    ledger = _ledger(tmp_path)
    first = request_retraining(TriggerType.DRIFT, "a", {}, producer=producer, ledger=ledger, now=_T0)
    second = request_retraining(
        TriggerType.DRIFT, "b", {}, producer=producer, ledger=ledger, now=_T0 + timedelta(hours=1)
    )
    assert first.emitted is True
    assert second.emitted is False
    assert second.suppressed_reason == "debounced"
    assert len(producer.published) == 1  # second never published


def test_request_retraining_allowed_after_window(tmp_path: Path) -> None:
    producer = MockEventProducer()
    ledger = _ledger(tmp_path)
    request_retraining(TriggerType.DRIFT, "a", {}, producer=producer, ledger=ledger, now=_T0)
    later = request_retraining(
        TriggerType.DRIFT, "b", {}, producer=producer, ledger=ledger, now=_T0 + timedelta(hours=25)
    )
    assert later.emitted is True
    assert len(producer.published) == 2


def test_triggers_debounce_independently(tmp_path: Path) -> None:
    producer = MockEventProducer()
    ledger = _ledger(tmp_path)
    # All three trigger types fire within the same window, none debounced against each other.
    for trigger in (TriggerType.DRIFT, TriggerType.LABEL_VOLUME, TriggerType.SCHEDULE):
        decision = request_retraining(trigger, "x", {}, producer=producer, ledger=ledger, now=_T0)
        assert decision.emitted is True
    assert len(producer.published) == 3
