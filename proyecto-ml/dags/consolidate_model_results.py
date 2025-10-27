from __future__ import annotations

from datetime import datetime

from airflow import DAG

from config import KEDRO_ENV, PROJECT_CONTAINER_PATH, default_args
from operators.kedro_operator import KedroOperator


with DAG(
    dag_id="consolidate_model_results",
    default_args=default_args,
    description="Consolidación/comparación de resultados de modelos",
    schedule_interval=None,  # manual por defecto; programar desde la UI si se desea
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    concurrency=2,
    tags=["kedro", "compare", "manual", "rainbow_six"],
) as dag:

    compare_models = KedroOperator(
        task_id="compare_models",
        pipeline_name="model_comparison",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    compare_models
