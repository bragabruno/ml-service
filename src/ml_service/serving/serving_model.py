from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class Predictable(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class LoadedModel:
    model: Predictable
    version: str
    feature_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ServingModel:
    def __init__(self) -> None:
        self._current: LoadedModel | None = None
        self._shadow: LoadedModel | None = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._current is not None

    @property
    def version(self) -> str:
        with self._lock:
            if self._current is None:
                return "none"
            return self._current.version

    def load(self, model: Predictable, version: str, **kwargs: Any) -> None:
        loaded = LoadedModel(model=model, version=version, **kwargs)
        with self._lock:
            self._current = loaded

    def swap(self, model: Predictable, version: str, **kwargs: Any) -> str:
        old_version = self.version
        self.load(model, version, **kwargs)
        return old_version

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._current is None:
                raise RuntimeError("No model loaded")
            return self._current.model.predict_proba(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(features)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    def unload(self) -> str:
        with self._lock:
            if self._current is None:
                return "none"
            old = self._current.version
            self._current = None
            return old

    # ── Shadow (challenger) slot ──────────────────────────────────────────────
    # A challenger scored alongside the champion for divergence telemetry. It NEVER
    # affects predict()/predict_proba() — the champion alone serves real decisions.

    @property
    def is_shadow_loaded(self) -> bool:
        with self._lock:
            return self._shadow is not None

    @property
    def shadow_version(self) -> str:
        with self._lock:
            return self._shadow.version if self._shadow is not None else "none"

    def load_shadow(self, model: Predictable, version: str, **kwargs: Any) -> None:
        loaded = LoadedModel(model=model, version=version, **kwargs)
        with self._lock:
            self._shadow = loaded

    def clear_shadow(self) -> str:
        with self._lock:
            if self._shadow is None:
                return "none"
            old = self._shadow.version
            self._shadow = None
            return old

    def predict_shadow(self, features: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._shadow is None:
                raise RuntimeError("No shadow model loaded")
            proba = self._shadow.model.predict_proba(features)
        return proba[:, 1] if proba.ndim == 2 else proba


_serving_model: ServingModel | None = None


def get_serving_model() -> ServingModel:
    global _serving_model
    if _serving_model is None:
        _serving_model = ServingModel()
    return _serving_model
