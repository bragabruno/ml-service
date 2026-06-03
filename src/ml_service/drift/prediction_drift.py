from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feature_drift import ks_statistic, population_stability_index


@dataclass(frozen=True)
class PredictionDrift:
    psi: float
    ks_statistic: float
    drifted: bool


def compute_prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    psi_threshold: float = 0.2,
) -> PredictionDrift:
    """Drift between a baseline score distribution and the live production scores."""
    psi = population_stability_index(reference_scores, current_scores)
    ks = ks_statistic(
        np.asarray(reference_scores, dtype=np.float64),
        np.asarray(current_scores, dtype=np.float64),
    )
    return PredictionDrift(psi=round(psi, 4), ks_statistic=round(ks, 4), drifted=psi > psi_threshold)
