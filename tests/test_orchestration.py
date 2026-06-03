from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ml_service.app.config import Settings
from ml_service.retraining.audit import RunLedger
from ml_service.retraining.orchestrator import run_retraining
from ml_service.retraining.promotion import evaluate_promotion
from ml_service.schemas.retraining import RunStatus, TriggerType
from ml_service.serving.model_registry import ModelRegistry

if TYPE_CHECKING:
    from pathlib import Path


def _synthetic_loader():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(size=(n, 34))
    y = (rng.uniform(size=n) > 0.8).astype(int)
    # fraud rows correlate with feature 0 so the model learns a signal
    x[y == 1, 0] += 1.5
    w = np.where(y == 1, 5.0, 1.0)
    split = int(n * 0.7)
    return x[:split], x[split:], y[:split], y[split:], w[:split], w[split:]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        retraining_runs_path=str(tmp_path / "runs.jsonl"),
        promotion_min_pr_auc_improvement=0.0,
    )


# ── Orchestrator (FRAUD-116) ──────────────────────────────────────────────────


def test_run_retraining_completes_and_registers(tmp_path: Path) -> None:
    registry = ModelRegistry(models_dir=tmp_path / "models")
    ledger = RunLedger(tmp_path / "runs.jsonl")
    run = run_retraining(
        trigger=TriggerType.DRIFT,
        reason="psi high",
        data_loader=_synthetic_loader,
        registry=registry,
        run_ledger=ledger,
        settings=_settings(tmp_path),
    )
    assert run.status in (RunStatus.REGISTERED, RunStatus.GATE_FAILED)
    assert run.candidate_version is not None
    assert run.candidate_version in registry.list_versions()
    assert "pr_auc" in run.metrics
    assert run.promotion is not None
    # Observable: exactly one run record written.
    assert len(ledger.all()) == 1


def test_run_retraining_does_not_deploy(tmp_path: Path) -> None:
    from ml_service.serving.serving_model import get_serving_model

    before = get_serving_model().version
    registry = ModelRegistry(models_dir=tmp_path / "models")
    run_retraining(
        trigger=TriggerType.SCHEDULE,
        data_loader=_synthetic_loader,
        registry=registry,
        run_ledger=RunLedger(tmp_path / "runs.jsonl"),
        settings=_settings(tmp_path),
    )
    # The serving model is never swapped by the orchestrator.
    assert get_serving_model().version == before


def test_run_retraining_records_failure(tmp_path: Path) -> None:
    def _broken_loader():
        raise RuntimeError("no data")

    ledger = RunLedger(tmp_path / "runs.jsonl")
    run = run_retraining(
        trigger=TriggerType.LABEL_VOLUME,
        data_loader=_broken_loader,
        registry=ModelRegistry(models_dir=tmp_path / "models"),
        run_ledger=ledger,
        settings=_settings(tmp_path),
    )
    assert run.status is RunStatus.FAILED
    assert run.error == "no data"
    assert run.promotion is not None and run.promotion.eligible is False


# ── Promotion guardrail (FRAUD-117) ───────────────────────────────────────────


def test_promotion_cold_start_eligible() -> None:
    decision = evaluate_promotion({"pr_auc": 0.7}, None, gate_passed=True)
    assert decision.eligible is True
    assert decision.requires_human_approval is True


def test_promotion_blocked_when_gate_failed() -> None:
    decision = evaluate_promotion({"pr_auc": 0.9}, {"pr_auc": 0.5}, gate_passed=False)
    assert decision.eligible is False


def test_promotion_blocked_on_regression() -> None:
    decision = evaluate_promotion({"pr_auc": 0.70}, {"pr_auc": 0.80}, gate_passed=True)
    assert decision.eligible is False
    assert any("PR-AUC" in r for r in decision.reasons)


def test_promotion_eligible_on_improvement() -> None:
    decision = evaluate_promotion({"pr_auc": 0.85, "cost": 100.0}, {"pr_auc": 0.80, "cost": 120.0}, gate_passed=True)
    assert decision.eligible is True


def test_promotion_blocked_on_worse_cost() -> None:
    decision = evaluate_promotion({"pr_auc": 0.85, "cost": 200.0}, {"pr_auc": 0.80, "cost": 120.0}, gate_passed=True)
    assert decision.eligible is False
    assert any("cost" in r for r in decision.reasons)
