from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_service.features.contract import FEATURE_NAMES
from ml_service.features.offline_store import get_offline_store


def load_training_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    store = get_offline_store()
    df = store.get_training_dataset()

    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].fillna(0.0).values.astype(np.float64)
    y = df["is_fraud"].values.astype(np.int32)
    weights = df["sample_weight"].values.astype(np.float64)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, w_train, w_test


def load_from_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].fillna(0.0).values.astype(np.float64)
    y = df["is_fraud"].values.astype(np.int32)
    weights = df.get("sample_weight", pd.Series(np.ones(len(df)))).values.astype(np.float64)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, w_train, w_test
