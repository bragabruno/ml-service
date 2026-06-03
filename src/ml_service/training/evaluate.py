from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass
class EvalResult:
    model_name: str
    pr_auc: float
    roc_auc: float
    recall: float
    precision: float
    f1: float
    accuracy: float
    fp_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    cost: float
    threshold: float = 0.5
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model_name": self.model_name,
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "fp_rate": round(self.fp_rate, 4),
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "cost": round(self.cost, 2),
            "threshold": self.threshold,
        }


def evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    threshold: float = 0.5,
    cost_missed_fraud: float = 100.0,
    cost_false_positive: float = 10.0,
) -> EvalResult:
    proba = model.predict_proba(X_test)
    scores = proba[:, 1] if proba.ndim == 2 else proba

    preds = (scores >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_test, scores))
    roc_auc = float(roc_auc_score(y_test, scores))
    recall = float(recall_score(y_test, preds))
    precision = float(precision_score(y_test, preds))
    f1 = float(f1_score(y_test, preds))
    accuracy = float(accuracy_score(y_test, preds))

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fp_rate = fp / max(fp + tn, 1)

    cost = fn * cost_missed_fraud + fp * cost_false_positive

    return EvalResult(
        model_name=model_name,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        recall=recall,
        precision=precision,
        f1=f1,
        accuracy=accuracy,
        fp_rate=fp_rate,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        cost=cost,
        threshold=threshold,
    )


def main() -> int:
    """CLI entrypoint: evaluate the latest persisted model on a held-out split."""
    import json

    from ml_service.serving.model_registry import get_registry

    from .dataset import load_training_data

    try:
        _x_train, x_test, _y_train, y_test, _w_train, _w_test = load_training_data()
        registry = get_registry()
        versions = registry.list_versions()
        if not versions:
            print("No trained model found; run `make train` first.")
            return 1
        model, _meta = registry.load(versions[-1])
    except Exception as exc:  # noqa: BLE001 - surface data/model errors clearly
        print(f"ERROR: {exc}")
        return 1

    result = evaluate(model, x_test, y_test, "xgboost")
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
