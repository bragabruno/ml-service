from __future__ import annotations

import numpy as np

from ml_service.features.contract import FEATURE_NAMES, NUM_FEATURES
from ml_service.features.transforms import compute_features, feature_importance_names


def test_compute_features_shape() -> None:
    payload = {"amount": 100.0, "country": "US", "created_at": "2026-01-15T10:30:00+00:00"}
    features = compute_features(payload)
    assert features.shape == (NUM_FEATURES,)


def test_compute_features_deterministic() -> None:
    payload = {"amount": 500.0, "country": "BR"}
    ctx = {"velocity_5m": 2, "velocity_24h": 10, "merchant_risk_tier": "HIGH"}
    f1 = compute_features(payload, ctx)
    f2 = compute_features(payload, ctx)
    np.testing.assert_array_equal(f1, f2)


def test_compute_features_amount_log() -> None:
    payload = {"amount": 99.0}
    features = compute_features(payload)
    idx = FEATURE_NAMES.index("amount")
    idx_log = FEATURE_NAMES.index("amount_log")
    assert features[idx] == 99.0
    assert abs(features[idx_log] - np.log1p(99.0)) < 1e-6


def test_compute_features_weekend() -> None:
    payload = {"amount": 50.0, "created_at": "2026-01-17T12:00:00+00:00"}
    features = compute_features(payload)
    idx = FEATURE_NAMES.index("is_weekend")
    assert features[idx] == 1.0


def test_compute_features_night() -> None:
    payload = {"amount": 50.0, "created_at": "2026-01-15T02:00:00+00:00"}
    features = compute_features(payload)
    idx = FEATURE_NAMES.index("is_night")
    assert features[idx] == 1.0


def test_feature_importance_names() -> None:
    names = feature_importance_names([0, 1, 2])
    assert names == ["amount", "amount_log", "amount_usd"]


def test_compute_features_foreign_country() -> None:
    payload = {"amount": 100.0, "country": "NG"}
    ctx = {"home_country": "US"}
    features = compute_features(payload, ctx)
    idx = FEATURE_NAMES.index("is_foreign_country")
    assert features[idx] == 1.0


def test_compute_features_new_device() -> None:
    payload = {"amount": 100.0, "created_at": "2026-01-15T10:00:00+00:00"}
    ctx = {"device_first_seen": "2026-01-14T10:00:00+00:00"}
    features = compute_features(payload, ctx)
    idx = FEATURE_NAMES.index("device_is_new")
    assert features[idx] == 1.0
