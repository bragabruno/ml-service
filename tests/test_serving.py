from __future__ import annotations

import numpy as np

from ml_service.serving.serving_model import ServingModel


class MockModel:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        scores = np.random.default_rng(42).uniform(0.1, 0.9, size=n)
        return np.column_stack([1 - scores, scores])


def test_serving_model_not_loaded() -> None:
    sm = ServingModel()
    assert not sm.is_loaded
    assert sm.version == "none"


def test_serving_model_load() -> None:
    sm = ServingModel()
    sm.load(MockModel(), "v1.0.0")
    assert sm.is_loaded
    assert sm.version == "v1.0.0"


def test_serving_model_predict() -> None:
    sm = ServingModel()
    sm.load(MockModel(), "v1.0.0")
    X = np.random.default_rng(0).uniform(size=(3, 34))
    proba = sm.predict_proba(X)
    assert proba.shape == (3, 2)


def test_serving_model_swap() -> None:
    sm = ServingModel()
    sm.load(MockModel(), "v1.0.0")
    old = sm.swap(MockModel(), "v2.0.0")
    assert old == "v1.0.0"
    assert sm.version == "v2.0.0"


def test_serving_model_unload() -> None:
    sm = ServingModel()
    sm.load(MockModel(), "v1.0.0")
    old = sm.unload()
    assert old == "v1.0.0"
    assert not sm.is_loaded


def test_serving_model_predict_raises_when_empty() -> None:
    sm = ServingModel()
    X = np.zeros((1, 34))
    try:
        sm.predict_proba(X)
        raise AssertionError("Should have raised")
    except RuntimeError:
        pass
