from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "ml-service",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="feature_pipeline",
    default_args=default_args,
    description="Refresh feature and training marts via dbt",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["dbt", "features", "etl"],
) as dag:

    dbt_seed = BashOperator(
        task_id="dbt_seed",
        bash_command="cd /opt/ml-service/dbt && dbt seed --target duckdb",
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command="cd /opt/ml-service/dbt && dbt run --target duckdb",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command="cd /opt/ml-service/dbt && dbt test --target duckdb",
    )

    parity_check = BashOperator(
        task_id="parity_check",
        bash_command="cd /opt/ml-service && python -m ml_service.features.parity",
    )

    dbt_seed >> dbt_run >> dbt_test >> parity_check
