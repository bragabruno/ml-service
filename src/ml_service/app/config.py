from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "mock"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    db_duckdb_path: str = "./data/fraud.duckdb"
    db_postgres_host: str = "localhost"
    db_postgres_port: int = 5433
    db_postgres_db: str = "fraud_db"
    db_postgres_user: str = "fraud_user"
    db_postgres_password: str = "fraud_pass"

    redis_host: str = "localhost"
    redis_port: int = 6380

    mlflow_tracking_uri: str = "http://localhost:5001"

    kafka_bootstrap_servers: str = "localhost:9093"

    service_host: str = "0.0.0.0"
    service_port: int = 8000
    log_level: str = "INFO"
    environment: str = "local"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.db_postgres_user}:{self.db_postgres_password}"
            f"@{self.db_postgres_host}:{self.db_postgres_port}/{self.db_postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
