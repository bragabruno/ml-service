from __future__ import annotations

from typing import TYPE_CHECKING

import optuna

from ml_service.training.evaluate import evaluate
from ml_service.training.train import TrainConfig, train_model

if TYPE_CHECKING:
    import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_trials: int = 20,
    random_state: int = 42,
) -> TrainConfig:
    def objective(trial: optuna.Trial) -> float:
        config = TrainConfig(
            xgb_max_depth=trial.suggest_int("max_depth", 3, 10),
            xgb_learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            xgb_n_estimators=trial.suggest_int("n_estimators", 50, 500),
            xgb_subsample=trial.suggest_float("subsample", 0.5, 1.0),
            xgb_colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            xgb_min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            xgb_gamma=trial.suggest_float("gamma", 0.0, 5.0),
            xgb_reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            xgb_reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            random_state=random_state,
        )
        model = train_model("xgboost", X_train, y_train, sample_weight, config)
        result = evaluate(model, X_val, y_val, "xgboost-tuned")
        return float(-result.pr_auc)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    return TrainConfig(
        xgb_max_depth=best["max_depth"],
        xgb_learning_rate=best["learning_rate"],
        xgb_n_estimators=best["n_estimators"],
        xgb_subsample=best["subsample"],
        xgb_colsample_bytree=best["colsample_bytree"],
        xgb_min_child_weight=best["min_child_weight"],
        xgb_gamma=best["gamma"],
        xgb_reg_alpha=best["reg_alpha"],
        xgb_reg_lambda=best["reg_lambda"],
        random_state=random_state,
    )
