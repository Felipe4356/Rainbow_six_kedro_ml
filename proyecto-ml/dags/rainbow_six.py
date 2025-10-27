from __future__ import annotations

from datetime import datetime

from airflow import DAG

from config import KEDRO_ENV, PROJECT_CONTAINER_PATH, default_args
from operators.kedro_operator import KedroOperator


with DAG(
    dag_id="rainbow_six",
    default_args=default_args,
    description="Ejecución individual del pipeline de procesamiento de datos 'rainbow_six'",
    schedule_interval=None,  # manual por defecto; programable desde la UI
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    concurrency=2,
    tags=["kedro", "data", "manual", "rainbow_six"],
) as dag:

    process_data = KedroOperator(
        task_id="process_data",
        pipeline_name="rainbow_six",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    process_data
