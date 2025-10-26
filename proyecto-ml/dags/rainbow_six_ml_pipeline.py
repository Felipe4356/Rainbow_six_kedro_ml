from __future__ import annotations

from datetime import datetime

from airflow import DAG

from config import KEDRO_ENV, PROJECT_CONTAINER_PATH, SCHEDULES, default_args
from operators.kedro_operator import KedroOperator


with DAG(
    dag_id="rainbow_six_ml_pipeline",
    default_args=default_args,
    description="Flujo ML completo diario: procesamiento de datos, entrenamiento y comparación",
    schedule_interval=SCHEDULES["daily_ml"],
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    concurrency=2,
    tags=["kedro", "ml", "daily", "rainbow_six"],
) as dag:

    # 1) Procesamiento de datos (ETL)
    process_data = KedroOperator(
        task_id="process_data",
        pipeline_name="rainbow_six",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    # 2) Entrenamiento de modelos
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

    # 3) Comparación y consolidación de resultados
    compare_models = KedroOperator(
        task_id="compare_models",
        pipeline_name="model_comparison",
        project_path=PROJECT_CONTAINER_PATH,
        env=KEDRO_ENV,
    )

    process_data >> [train_classification, train_regression] >> compare_models
