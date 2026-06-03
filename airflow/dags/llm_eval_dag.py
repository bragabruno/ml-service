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
    dag_id="llm_eval",
    default_args=default_args,
    description="Run LLM evaluation suite and publish report",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["eval", "llm", "agent"],
) as dag:

    run_eval = BashOperator(
        task_id="run_eval",
        bash_command="cd /opt/ml-service && python -m eval.runner",
    )

    eval_gate = BashOperator(
        task_id="eval_gate",
        bash_command="cd /opt/ml-service && python -m eval.gate",
    )

    publish_report = BashOperator(
        task_id="publish_report",
        bash_command="cd /opt/ml-service && python -m eval.report",
    )

    run_eval >> eval_gate >> publish_report
