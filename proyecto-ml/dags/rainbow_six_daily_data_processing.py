from __future__ import annotations

from datetime import datetime

from airflow import DAG

from config import KEDRO_ENV, PROJECT_CONTAINER_PATH, SCHEDULES, default_args
from operators.kedro_operator import KedroOperator


with DAG(
    dag_id="rainbow_six_daily_data_processing",
    default_args=default_args,
    description="Procesamiento de datos cada 4 horas usando el pipeline 'rainbow_six'",
    schedule_interval=SCHEDULES["every_4h"],
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["kedro", "data", "4h", "rainbow_six"],
) as dag:

    process_data = KedroOperator(
        task_id="process_data",
        pipeline_name="rainbow_six",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    process_data
