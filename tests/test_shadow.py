from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ml_service.serving.serving_model import ServingModel
from ml_service.serving.shadow import decision_for_score, record_shadow_comparison

if TYPE_CHECKING:
    from pathlib import Path


class _FixedModel:
    """Returns a constant fraud probability for every row."""

    def __init__(self, score: float) -> None:
        self._score = score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack([np.full(n, 1 - self._score), np.full(n, self._score)])


class _BrokenModel:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("shadow model exploded")


def test_decision_for_score_bands() -> None:
    assert decision_for_score(0.95) == ("HIGH", "DECLINE")
    assert decision_for_score(0.5) == ("MEDIUM", "REVIEW")
    assert decision_for_score(0.1) == ("LOW", "APPROVE")


def test_shadow_slot_load_predict_clear() -> None:
    serving = ServingModel()
    assert serving.is_shadow_loaded is False
    serving.load_shadow(_FixedModel(0.9), "v-challenger")
    assert serving.is_shadow_loaded is True
    assert serving.shadow_version == "v-challenger"
    scores = serving.predict_shadow(np.zeros((3, 34)))
    assert scores.shape == (3,)
    assert abs(float(scores[0]) - 0.9) < 1e-9
    assert serving.clear_shadow() == "v-challenger"
    assert serving.is_shadow_loaded is False


def test_champion_unaffected_by_shadow() -> None:
    serving = ServingModel()
    serving.load(_FixedModel(0.2), "champion-v1")
    serving.load_shadow(_FixedModel(0.95), "challenger-v2")
    # The champion's own prediction is unchanged by the presence of a shadow.
    champ = float(serving.predict(np.zeros((1, 34)))[0])
    assert abs(champ - 0.2) < 1e-9
    assert serving.version == "champion-v1"


def test_record_shadow_comparison_writes_and_computes(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    rec = record_shadow_comparison(
        "TXN-1",
        champion_score=0.2,
        shadow_score=0.85,
        champion_version="c1",
        shadow_version="s2",
        path=path,
    )
    assert rec["score_delta"] == 0.65
    assert rec["champion_decision"] == "APPROVE"
    assert rec["shadow_decision"] == "DECLINE"
    assert rec["agree"] is False
    assert path.read_text().strip().count("\n") == 0  # exactly one record
    assert '"transaction_id": "TXN-1"' in path.read_text()


def test_record_shadow_comparison_agreement(tmp_path: Path) -> None:
    rec = record_shadow_comparison(
        "TXN-2",
        champion_score=0.80,
        shadow_score=0.92,
        champion_version="c1",
        shadow_version="s2",
        path=tmp_path / "s.jsonl",
    )
    assert rec["agree"] is True  # both DECLINE


def test_predict_survives_broken_shadow() -> None:
    # A raising shadow model must not break the champion prediction path.
    serving = ServingModel()
    serving.load(_FixedModel(0.3), "champion-v1")
    serving.load_shadow(_BrokenModel(), "broken-v2")
    from ml_service.app.api.routes.predict import _maybe_shadow_score

    # Should swallow the shadow error and return None without raising.
    assert _maybe_shadow_score(serving, np.zeros((1, 34)), "TXN-3", 0.3) is None
