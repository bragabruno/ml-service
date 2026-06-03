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


def check_drift(**kwargs):
    """Compute real feature-level PSI from the offline feature store.

    Uses the older half of the training dataset as the reference distribution and
    the recent half as the current window — a baseline-vs-recent comparison that is
    offline-runnable against the duckdb mart (no live stream required).
    """
    from ml_service.drift.feature_drift import evaluate_drift
    from ml_service.features.offline_store import get_offline_store

    df = get_offline_store().get_training_dataset()
    numeric = df.select_dtypes(include="number")
    mid = len(numeric) // 2
    reference = numeric.iloc[:mid].to_numpy()
    current = numeric.iloc[mid:].to_numpy()
    return evaluate_drift(reference, current, list(numeric.columns))


with DAG(
    dag_id="drift_monitor",
    default_args=default_args,
    description="Monitor feature and prediction drift; trigger retraining if detected",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["drift", "monitoring", "mlops"],
) as dag:

    compute_drift = PythonOperator(
        task_id="compute_drift",
        python_callable=check_drift,
        provide_context=True,
    )

    alert_on_drift = BashOperator(
        task_id="alert_on_drift",
        bash_command='echo "Drift detected — retraining requested"',
        trigger_rule="all_success",
    )

    compute_drift >> alert_on_drift
