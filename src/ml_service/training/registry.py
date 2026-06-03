from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from ml_service.app.config import get_settings


def log_to_mlflow(
    model: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    model_name: str = "fraud-detector",
    version: str = "v1.0.0",
) -> str:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return "mlflow-not-available"

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name=f"{model_name}-{version}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        return str(run.info.run_id)


def register_model(
    model: Any,
    version: str,
    metrics: dict[str, float],
    output_dir: str | Path = "./models",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"model_{version}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "version": version, "metrics": metrics}, f)
    return path


def main() -> int:
    """CLI entrypoint: register the latest persisted model with MLflow (best-effort)."""
    from ml_service.serving.model_registry import get_registry

    registry = get_registry()
    versions = registry.list_versions()
    if not versions:
        print("No trained model found; run `make train` first.")
        return 1

    model, meta = registry.load(versions[-1])
    metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    try:
        run_id = log_to_mlflow(model, numeric, {"version": versions[-1]}, version=versions[-1])
        print(f"Registered {versions[-1]} (mlflow run: {run_id})")
    except Exception as exc:  # noqa: BLE001 - MLflow server is optional locally
        print(f"(mlflow registration skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
