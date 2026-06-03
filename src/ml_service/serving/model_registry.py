from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from ml_service.serving.serving_model import Predictable, ServingModel, get_serving_model


class ModelRegistry:
    def __init__(self, models_dir: str | Path = "./models") -> None:
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: Predictable, version: str, metadata: dict[str, Any] | None = None) -> Path:
        path = self._models_dir / f"model_{version}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "version": version, "metadata": metadata or {}}, f)
        return path

    def load(self, version: str) -> tuple[Predictable, dict[str, Any]]:
        path = self._models_dir / f"model_{version}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model version {version} not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["model"], data.get("metadata", {})

    def list_versions(self) -> list[str]:
        return sorted(p.stem.replace("model_", "") for p in self._models_dir.glob("model_*.pkl"))

    def deploy(self, version: str, serving: ServingModel | None = None) -> str:
        srv = serving or get_serving_model()
        model, metadata = self.load(version)
        old = srv.swap(model, version, metadata=metadata)
        return str(old)


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
