from __future__ import annotations

from datetime import datetime

from airflow import DAG

from config import KEDRO_ENV, PROJECT_CONTAINER_PATH, default_args
from operators.kedro_operator import KedroOperator


with DAG(
    dag_id="modelo_regresion",
    default_args=default_args,
    description="Ejecución individual del pipeline de regresión",
    schedule_interval=None,  # manual por defecto; programar desde la UI si se desea
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    concurrency=2,
    tags=["kedro", "regression", "manual", "rainbow_six"],
) as dag:

    train_regression = KedroOperator(
        task_id="train_regression",
        pipeline_name="modelo_regresion",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    train_regression
