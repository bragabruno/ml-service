from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "mock"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    db_duckdb_path: str = "./data/fraud.duckdb"
    db_postgres_host: str = "localhost"
    db_postgres_port: int = 5432
    db_postgres_db: str = "fraud_db"
    db_postgres_user: str = "fraud_user"
    db_postgres_password: str = "fraud_pass"

    redis_host: str = "localhost"
    redis_port: int = 6379

    mlflow_tracking_uri: str = "http://localhost:5001"

    kafka_bootstrap_servers: str = "localhost:9092"

    # Case-management integration (HITL). mock = in-memory/log (default, offline);
    # http = POST drafted reports/labels to the case-management service.
    case_mgmt_provider: str = "mock"
    case_mgmt_url: str = ""

    # Retraining-event producer. mock = in-memory/log (default, offline); kafka = publish to broker.
    event_producer_provider: str = "mock"
    retraining_audit_path: str = "./data/retraining_audit.jsonl"
    retraining_debounce_hours: float = 24.0
    # Retraining orchestration + promotion guardrails.
    retraining_runs_path: str = "./data/retraining_runs.jsonl"
    promotion_min_pr_auc_improvement: float = 0.0

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
