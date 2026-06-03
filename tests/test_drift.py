from __future__ import annotations

import numpy as np

from ml_service.drift.feature_drift import (
    compute_feature_drift,
    ks_statistic,
    population_stability_index,
)
from ml_service.drift.prediction_drift import compute_prediction_drift


def test_psi_near_zero_for_identical() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    assert population_stability_index(x, x) < 0.01


def test_psi_high_for_shifted() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=1000)
    cur = rng.normal(3, 1, size=1000)
    assert population_stability_index(ref, cur) > 0.25


def test_ks_statistic_bounds() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=500)
    assert ks_statistic(a, a) == 0.0
    b = rng.normal(5, 1, size=500)
    assert 0.0 < ks_statistic(a, b) <= 1.0


def test_compute_feature_drift_flags_only_shifted_column() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(500, 3))
    cur = ref.copy()
    cur[:, 1] += 4.0
    by_name = {d.feature: d for d in compute_feature_drift(ref, cur, ["a", "b", "c"])}
    assert not by_name["a"].drifted
    assert by_name["b"].drifted


def test_prediction_drift_detects_score_shift() -> None:
    rng = np.random.default_rng(3)
    ref = rng.uniform(0.0, 0.3, size=1000)
    cur = rng.uniform(0.5, 1.0, size=1000)
    assert compute_prediction_drift(ref, cur).drifted
