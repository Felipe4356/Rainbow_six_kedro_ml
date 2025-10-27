from __future__ import annotations

from datetime import datetime

from airflow import DAG

from config import KEDRO_ENV, PROJECT_CONTAINER_PATH, default_args
from operators.kedro_operator import KedroOperator


with DAG(
    dag_id="modelo_clasificacion",
    default_args=default_args,
    description="Ejecución individual del pipeline de clasificación",
    schedule_interval=None,  # manual por defecto; programar desde la UI si se desea
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    concurrency=2,
    tags=["kedro", "classification", "manual", "rainbow_six"],
) as dag:

    train_classification = KedroOperator(
        task_id="train_classification",
        pipeline_name="modelo_clasificacion",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    train_classification
