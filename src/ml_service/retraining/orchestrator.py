from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from ml_service.app.observability import get_logger
from ml_service.features.contract import CONTRACT_VERSION
from ml_service.retraining.audit import RunLedger
from ml_service.retraining.promotion import evaluate_promotion
from ml_service.schemas.retraining import (
    PromotionDecision,
    RetrainingRun,
    RunStatus,
    TriggerType,
)
from ml_service.serving.model_registry import get_registry
from ml_service.serving.serving_model import get_serving_model
from ml_service.training.evaluate import evaluate
from ml_service.training.gate import check_gate
from ml_service.training.train import train_model

if TYPE_CHECKING:
    from ml_service.app.config import Settings
    from ml_service.serving.model_registry import ModelRegistry

logger = get_logger("retraining_orchestrator")

DataLoader = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def _incumbent_metrics(registry: ModelRegistry) -> dict[str, Any] | None:
    """Metrics of the currently-deployed production model, or None if nothing is serving."""
    serving = get_serving_model()
    if not serving.is_loaded:
        return None
    try:
        _model, meta = registry.load(serving.version)
    except FileNotFoundError:
        return None
    metrics = meta.get("metrics") if isinstance(meta, dict) else None
    return metrics if isinstance(metrics, dict) else None


def run_retraining(
    *,
    trigger: TriggerType,
    reason: str = "",
    data_loader: DataLoader | None = None,
    registry: ModelRegistry | None = None,
    run_ledger: RunLedger | None = None,
    model_name: str = "xgboost",
    settings: Settings | None = None,
    now: datetime | None = None,
) -> RetrainingRun:
    """Train → evaluate → gate → register a candidate, then assess promotion eligibility.

    Observable end-to-end: the run is appended to a JSONL ledger and logged. The candidate is
    registered but never deployed — promotion stays a human decision (FRAUD-117 / FRAUD-080).
    Dependencies are injectable so the flow is unit-testable offline (no duckdb / live model).
    """
    if settings is None:
        from ml_service.app.config import get_settings

        settings = get_settings()
    if data_loader is None:
        from ml_service.training.dataset import load_training_data

        data_loader = load_training_data
    registry = registry or get_registry()
    run_ledger = run_ledger or RunLedger(settings.retraining_runs_path)

    run_id = str(uuid.uuid4())
    started = (now or datetime.now(UTC)).isoformat()
    run = RetrainingRun(run_id=run_id, trigger=trigger, status=RunStatus.REQUESTED, started_at=started)
    logger.info("retraining_started", run_id=run_id, trigger=trigger.value, reason=reason)

    try:
        x_train, x_test, y_train, y_test, w_train, _w_test = data_loader()
        model = train_model(model_name, x_train, y_train, sample_weight=w_train)
        run.status = RunStatus.TRAINED

        result = evaluate(model, x_test, y_test, model_name)
        metrics = result.to_dict()
        gate = check_gate(result.pr_auc, result.roc_auc, result.recall, result.fp_rate, result.cost, len(y_test))
        run.metrics = metrics
        run.gate_passed = gate.passed
        run.gate_violations = gate.violations

        candidate_version = f"v{CONTRACT_VERSION}-candidate-{run_id[:8]}"
        registry.save(model, candidate_version, metadata={"metrics": metrics, "gate_passed": gate.passed})
        run.candidate_version = candidate_version
        run.status = RunStatus.REGISTERED if gate.passed else RunStatus.GATE_FAILED

        run.promotion = evaluate_promotion(
            metrics,
            _incumbent_metrics(registry),
            gate_passed=gate.passed,
            min_pr_auc_improvement=settings.promotion_min_pr_auc_improvement,
        )
    except Exception as exc:  # noqa: BLE001 - record failure, never crash the consumer loop
        run.status = RunStatus.FAILED
        run.error = str(exc)
        run.promotion = run.promotion or PromotionDecision(eligible=False, reasons=[f"run failed: {exc}"])
        logger.warning("retraining_failed", run_id=run_id, error=str(exc))

    run.finished_at = (now or datetime.now(UTC)).isoformat()
    run_ledger.append(run.model_dump(mode="json"))
    logger.info(
        "retraining_finished",
        run_id=run_id,
        status=run.status.value,
        candidate=run.candidate_version,
        eligible=run.promotion.eligible if run.promotion else None,
    )
    return run
