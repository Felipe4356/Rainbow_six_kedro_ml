from __future__ import annotations

from datetime import datetime

from airflow import DAG

from config import KEDRO_ENV, PROJECT_CONTAINER_PATH, default_args
from operators.kedro_operator import KedroOperator


with DAG(
    dag_id="rainbow_six_on_demand",
    default_args=default_args,
    description="DAG manual para ejecutar procesamiento, entrenamiento y comparación",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["kedro", "manual", "rainbow_six"],
) as dag:

    process_data = KedroOperator(
        task_id="process_data",
        pipeline_name="rainbow_six",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    train_classification = KedroOperator(
        task_id="train_classification",
        pipeline_name="modelo_clasificacion",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    train_regression = KedroOperator(
        task_id="train_regression",
        pipeline_name="modelo_regresion",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    compare_models = KedroOperator(
        task_id="compare_models",
        pipeline_name="model_comparison",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    process_data >> [train_classification, train_regression] >> compare_models
