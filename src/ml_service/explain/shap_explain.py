from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_service.features.contract import FEATURE_NAMES


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    contribution: float


def explain_prediction(model: Any, x: np.ndarray, *, top_k: int = 5) -> list[FeatureContribution]:
    """Top-k per-prediction feature contributions for a single input row.

    Uses SHAP TreeExplainer when available; falls back to the model's gain-based
    feature importances (FRAUD-084) if SHAP cannot explain the model.
    """
    row = np.asarray(x, dtype=np.float64).reshape(1, -1)
    try:
        import shap

        explainer: Any = shap.TreeExplainer(model)
        raw = explainer.shap_values(row)
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001 - SHAP may not support the model / not be installed
        values = _fallback_importances(model, row.shape[1])

    n = min(values.size, row.shape[1])
    contributions = [
        FeatureContribution(
            feature=FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}",
            value=float(row[0, i]),
            contribution=round(float(values[i]), 6),
        )
        for i in range(n)
    ]
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    return contributions[:top_k]


def _fallback_importances(model: Any, n_features: int) -> np.ndarray:
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        return np.asarray(importances, dtype=np.float64)
    return np.zeros(n_features, dtype=np.float64)
