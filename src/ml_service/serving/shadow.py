from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ml_service.app.observability import SHADOW_DECISION_AGREEMENT, SHADOW_SCORE_DELTA, get_logger

logger = get_logger("shadow")

# Score bands shared by /predict and shadow comparison — single source of truth.
HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4


def decision_for_score(score: float) -> tuple[str, str]:
    """Map a fraud score to (risk_level, decision). The /predict band thresholds."""
    if score >= HIGH_THRESHOLD:
        return "HIGH", "DECLINE"
    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM", "REVIEW"
    return "LOW", "APPROVE"


def record_shadow_comparison(
    transaction_id: str,
    champion_score: float,
    shadow_score: float,
    *,
    champion_version: str,
    shadow_version: str,
    path: str | Path,
) -> dict[str, object]:
    """Record champion-vs-shadow divergence: score delta, decision agreement, metrics, audit row.

    Pure side-effect sink — returns the comparison record. Never raises into the caller's hot path;
    callers still guard it, but this keeps the failure surface tiny.
    """
    delta = abs(champion_score - shadow_score)
    _, champion_decision = decision_for_score(champion_score)
    _, shadow_decision = decision_for_score(shadow_score)
    agree = champion_decision == shadow_decision

    SHADOW_SCORE_DELTA.observe(delta)
    SHADOW_DECISION_AGREEMENT.labels(agreement="agree" if agree else "disagree").inc()

    record: dict[str, object] = {
        "transaction_id": transaction_id,
        "champion_version": champion_version,
        "shadow_version": shadow_version,
        "champion_score": round(champion_score, 4),
        "shadow_score": round(shadow_score, 4),
        "score_delta": round(delta, 4),
        "champion_decision": champion_decision,
        "shadow_decision": shadow_decision,
        "agree": agree,
        "recorded_at": datetime.now(UTC).isoformat(),
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    logger.info(
        "shadow_comparison",
        txn_id=transaction_id,
        delta=record["score_delta"],
        agree=agree,
        champion=champion_decision,
        shadow=shadow_decision,
    )
    return record
