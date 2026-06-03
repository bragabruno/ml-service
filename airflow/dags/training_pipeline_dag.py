from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ml-service",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_gate(**kwargs):
    from ml_service.training.gate import GateThresholds, check_gate
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_model")
    if metrics is None:
        raise ValueError("No evaluation metrics found")
    result = check_gate(
        pr_auc=metrics["pr_auc"],
        roc_auc=metrics["roc_auc"],
        recall=metrics["recall"],
        fp_rate=metrics["fp_rate"],
        cost=metrics["cost"],
        n_test=metrics.get("n_test", 1000),
    )
    if not result.passed:
        raise ValueError(f"Gate failed: {result.violations}")
    return result.to_dict()


with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Train, evaluate, gate, and register fraud detection model",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["training", "mlflow", "model"],
) as dag:

    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command="cd /opt/ml-service/dbt && dbt build --target duckdb",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/ml-service && python -m ml_service.training.train",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="cd /opt/ml-service && python -m ml_service.training.evaluate",
    )

    gate_check = PythonOperator(
        task_id="gate_check",
        python_callable=check_gate,
        provide_context=True,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="cd /opt/ml-service && python -m ml_service.training.registry",
    )

    dbt_build >> train_model >> evaluate_model >> gate_check >> register_model
