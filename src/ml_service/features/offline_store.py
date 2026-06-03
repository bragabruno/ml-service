from __future__ import annotations

from typing import TYPE_CHECKING, Any

import duckdb

from ml_service.app.config import get_settings

if TYPE_CHECKING:
    import pandas as pd


class OfflineFeatureStore:
    def __init__(self, duckdb_path: str | None = None) -> None:
        settings = get_settings()
        self._path = duckdb_path or settings.db_duckdb_path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._path, read_only=True)

    def get_training_dataset(self) -> pd.DataFrame:
        conn = self._connect()
        try:
            return conn.execute("SELECT * FROM training_dataset").fetchdf()
        finally:
            conn.close()

    def get_features_for_transaction(self, transaction_id: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            result = conn.execute(
                "SELECT * FROM fct_transaction_features WHERE transaction_id = ?",
                [transaction_id],
            ).fetchone()
            if result is None:
                return {}
            columns = [desc[0] for desc in (conn.description or [])]
            return dict(zip(columns, result, strict=False))
        finally:
            conn.close()

    def get_feature_names(self) -> list[str]:
        conn = self._connect()
        try:
            result = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'fct_transaction_features' ORDER BY ordinal_position"
            ).fetchall()
            return [r[0] for r in result]
        finally:
            conn.close()

    def list_tables(self) -> list[str]:
        conn = self._connect()
        try:
            result = conn.execute("SHOW TABLES").fetchall()
            return [r[0] for r in result]
        finally:
            conn.close()


_store: OfflineFeatureStore | None = None


def get_offline_store() -> OfflineFeatureStore:
    global _store
    if _store is None:
        _store = OfflineFeatureStore()
    return _store
