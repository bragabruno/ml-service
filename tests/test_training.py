from __future__ import annotations

import numpy as np

from ml_service.training.evaluate import EvalResult, evaluate
from ml_service.training.gate import check_gate
from ml_service.training.train import train_model


def test_evaluate_basic() -> None:
    class MockModel:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            n = X.shape[0]
            scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])[:n]
            return np.column_stack([1 - scores, scores])

    X = np.zeros((5, 34))
    y = np.array([0, 1, 0, 1, 0])
    result = evaluate(MockModel(), X, y, "mock")
    assert isinstance(result, EvalResult)
    assert result.model_name == "mock"
    assert 0.0 <= result.pr_auc <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0


def test_gate_passes() -> None:
    result = check_gate(pr_auc=0.85, roc_auc=0.95, recall=0.80, fp_rate=0.05, cost=100.0, n_test=100)
    assert result.passed
    assert result.violations == []


def test_gate_fails_on_low_recall() -> None:
    result = check_gate(pr_auc=0.85, roc_auc=0.95, recall=0.50, fp_rate=0.05, cost=100.0, n_test=100)
    assert not result.passed
    assert any("Recall" in v for v in result.violations)


def test_gate_fails_on_high_fp_rate() -> None:
    result = check_gate(pr_auc=0.85, roc_auc=0.95, recall=0.80, fp_rate=0.25, cost=100.0, n_test=100)
    assert not result.passed
    assert any("FP rate" in v for v in result.violations)


def test_train_xgboost() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(size=(100, 34))
    y = (rng.uniform(size=100) > 0.9).astype(int)
    model = train_model("xgboost", X, y)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)


def test_train_logreg() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(size=(100, 34))
    y = (rng.uniform(size=100) > 0.9).astype(int)
    model = train_model("logreg", X, y)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 2)
