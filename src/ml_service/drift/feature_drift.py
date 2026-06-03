from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    drifted: bool


def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between a reference and a current distribution.

    Rule of thumb: PSI < 0.1 = no shift, 0.1–0.25 = moderate, > 0.25 = major shift.
    Bins are derived from quantiles of the reference distribution.
    """
    expected_arr = np.asarray(expected, dtype=np.float64)
    actual_arr = np.asarray(actual, dtype=np.float64)
    if expected_arr.size == 0 or actual_arr.size == 0:
        return 0.0

    edges = np.percentile(expected_arr, np.linspace(0, 100, buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected_arr, bins=edges)
    act_counts, _ = np.histogram(actual_arr, bins=edges)

    exp_frac = np.clip(exp_counts / expected_arr.size, 1e-6, None)
    act_frac = np.clip(act_counts / actual_arr.size, 1e-6, None)

    return float(np.sum((act_frac - exp_frac) * np.log(act_frac / exp_frac)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max CDF gap), pure-numpy.

    Implemented without scipy so the drift module has no extra dependency.
    """
    a_sorted = np.sort(np.asarray(a, dtype=np.float64))
    b_sorted = np.sort(np.asarray(b, dtype=np.float64))
    if a_sorted.size == 0 or b_sorted.size == 0:
        return 0.0
    grid = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, grid, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, grid, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compute_feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str],
    *,
    psi_threshold: float = 0.25,
) -> list[FeatureDrift]:
    """Per-feature PSI + KS between a reference and a current feature matrix (rows=samples)."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    results: list[FeatureDrift] = []
    for i, name in enumerate(feature_names):
        ref_col = ref[:, i]
        cur_col = cur[:, i]
        psi = population_stability_index(ref_col, cur_col)
        results.append(
            FeatureDrift(
                feature=name,
                psi=round(psi, 4),
                ks_statistic=round(ks_statistic(ref_col, cur_col), 4),
                drifted=psi > psi_threshold,
            )
        )
    return results
