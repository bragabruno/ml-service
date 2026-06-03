from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class TrainConfig:
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 200
    xgb_scale_pos_weight: float | None = None
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: int = 3
    xgb_gamma: float = 0.0
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    logreg_c: float = 1.0
    random_state: int = 42


def build_xgboost(config: TrainConfig, sample_weight: np.ndarray | None = None) -> XGBClassifier:
    scale_pos = config.xgb_scale_pos_weight
    if scale_pos is None and sample_weight is not None:
        n_pos = int(np.sum(sample_weight > 1.0))
        n_neg = len(sample_weight) - n_pos
        scale_pos = n_neg / max(n_pos, 1)

    return XGBClassifier(
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        n_estimators=config.xgb_n_estimators,
        scale_pos_weight=scale_pos or 1.0,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        min_child_weight=config.xgb_min_child_weight,
        gamma=config.xgb_gamma,
        reg_alpha=config.xgb_reg_alpha,
        reg_lambda=config.xgb_reg_lambda,
        random_state=config.random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )


def build_logreg(config: TrainConfig) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=config.logreg_c,
                    max_iter=1000,
                    random_state=config.random_state,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    config: TrainConfig | None = None,
) -> Any:
    cfg = config or TrainConfig()

    if model_name == "xgboost":
        model = build_xgboost(cfg, sample_weight)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    elif model_name == "logreg":
        model = build_logreg(cfg)
        # Pipeline.fit needs the step-prefixed name to route sample_weight to the classifier.
        model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main() -> int:
    """CLI entrypoint: load the dbt training mart, train XGBoost + a LogReg baseline,
    evaluate, run the promotion gate, persist the artifact, and (best-effort) log to MLflow.
    """
    import json

    from ml_service.features.contract import CONTRACT_VERSION
    from ml_service.serving.model_registry import get_registry

    from .dataset import load_training_data
    from .evaluate import evaluate
    from .gate import check_gate
    from .registry import log_to_mlflow

    try:
        x_train, x_test, y_train, y_test, w_train, _w_test = load_training_data()
    except Exception as exc:  # noqa: BLE001 - surface any data/connection error clearly
        print(f"ERROR: could not load training data: {exc}")
        print("Run `make generate && make dbt` first to build the dbt training_dataset.")
        return 1

    cfg = TrainConfig()
    print(f"Training on {len(x_train)} rows ({int(y_train.sum())} fraud) ...")
    baseline = train_model("logreg", x_train, y_train, sample_weight=w_train, config=cfg)
    model = train_model("xgboost", x_train, y_train, sample_weight=w_train, config=cfg)

    base_eval = evaluate(baseline, x_test, y_test, "logreg-baseline")
    xgb_eval = evaluate(model, x_test, y_test, "xgboost")
    print(f"baseline PR-AUC={base_eval.pr_auc:.4f} | xgboost PR-AUC={xgb_eval.pr_auc:.4f}")

    gate = check_gate(xgb_eval.pr_auc, xgb_eval.roc_auc, xgb_eval.recall, xgb_eval.fp_rate, xgb_eval.cost, len(y_test))

    version = f"v{CONTRACT_VERSION}"
    metrics = xgb_eval.to_dict()
    artifact = get_registry().save(model, version, metadata={"metrics": metrics, "gate_passed": gate.passed})

    run_id = "mlflow-skipped"
    numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    try:
        run_id = log_to_mlflow(
            model, numeric_metrics, {"model": "xgboost", "contract_version": CONTRACT_VERSION}, version=version
        )
    except Exception as exc:  # noqa: BLE001 - the MLflow server is optional for local runs
        print(f"(mlflow logging skipped: {exc})")

    print(
        json.dumps(
            {
                "version": version,
                "artifact": str(artifact),
                "mlflow_run": run_id,
                "gate_passed": gate.passed,
                "violations": gate.violations,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
