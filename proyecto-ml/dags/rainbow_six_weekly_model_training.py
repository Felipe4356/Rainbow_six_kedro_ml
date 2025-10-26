from __future__ import annotations

from datetime import datetime

from airflow import DAG

from config import KEDRO_ENV, PROJECT_CONTAINER_PATH, SCHEDULES, default_args
from operators.kedro_operator import KedroOperator


with DAG(
    dag_id="rainbow_six_weekly_model_training",
    default_args=default_args,
    description="Reentrenamiento semanal de modelos y consolidación",
    schedule_interval=SCHEDULES["weekly"],
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["kedro", "training", "weekly", "rainbow_six"],
) as dag:

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

    [train_classification, train_regression] >> compare_models
