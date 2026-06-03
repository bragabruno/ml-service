from __future__ import annotations

import numpy as np

from ml_service.agent.llm.mock_client import MockLLM
from ml_service.explain.narrator import narrate_explanation
from ml_service.explain.shap_explain import explain_prediction
from ml_service.features.contract import FEATURE_NAMES, NUM_FEATURES
from ml_service.training.train import train_model


def test_explain_prediction_returns_top_k_named_features() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(size=(80, NUM_FEATURES))
    y = (rng.uniform(size=80) > 0.8).astype(int)
    model = train_model("xgboost", x, y)

    contributions = explain_prediction(model, x[0], top_k=5)
    assert 0 < len(contributions) <= 5
    assert all(c.feature in FEATURE_NAMES for c in contributions)
    # Sorted by absolute contribution, descending.
    abs_vals = [abs(c.contribution) for c in contributions]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_narrator_produces_text() -> None:
    narrative = narrate_explanation(
        MockLLM(),
        model_version="v1.0.0",
        fraud_probability=0.78,
        risk_level="HIGH",
        features=[{"name": "new_device", "value": 1.0, "contribution": 0.31}],
        rule_hits=[{"name": "NEW_DEVICE_FOREIGN_COUNTRY", "description": "New device from a foreign country"}],
    )
    assert isinstance(narrative, str)
    assert len(narrative) > 0
