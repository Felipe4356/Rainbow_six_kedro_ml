from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project


class KedroOperator(BaseOperator):
    @apply_defaults
    def __init__(
        self,
        package_name: str,
        pipeline_name: str,
        node_name: str | list[str],
        project_path: str | Path,
        env: str,
        conf_source: str,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.package_name = package_name
        self.pipeline_name = pipeline_name
        self.node_name = node_name
        self.project_path = project_path
        self.env = env
        self.conf_source = conf_source

    def execute(self, context):
        configure_project(self.package_name)
        with KedroSession.create(self.project_path, env=self.env, conf_source=self.conf_source) as session:
            if isinstance(self.node_name, str):
                self.node_name = [self.node_name]
            session.run(self.pipeline_name, node_names=self.node_name)

# Kedro settings required to run your pipeline
env = "base"
pipeline_name = "__default__"
project_path = Path.cwd()
package_name = "proteyecto_ml"
conf_source = "" or Path.cwd() / "conf"


# Using a DAG context manager, you don't have to specify the dag property of each task
with DAG(
    dag_id="proteyecto-ml",
    start_date=datetime(2023,1,1),
    max_active_runs=3,
    # https://airflow.apache.org/docs/stable/scheduler.html#dag-runs
    schedule_interval="@once",
    catchup=False,
    # Default settings applied to all tasks
    default_args=dict(
        owner="airflow",
        depends_on_past=False,
        email_on_failure=False,
        email_on_retry=False,
        retries=1,
        retry_delay=timedelta(minutes=5)
    )
) as dag:
    tasks = {
        "combinar-raw-node": KedroOperator(
            task_id="combinar-raw-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="combinar_raw_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "limpiar-datos-node": KedroOperator(
            task_id="limpiar-datos-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="limpiar_datos_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "eliminar-atipicos-node": KedroOperator(
            task_id="eliminar-atipicos-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="eliminar_atipicos_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "preparar-datos-node": KedroOperator(
            task_id="preparar-datos-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="preparar_datos_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "crear-kdr-node": KedroOperator(
            task_id="crear-kdr-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="crear_kdr_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "crear-impact-score-node": KedroOperator(
            task_id="crear-impact-score-node",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="crear_impact_score_node",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-decision-tree": KedroOperator(
            task_id="train-decision-tree",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_decision_tree",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-decision-tree-classifier": KedroOperator(
            task_id="train-decision-tree-classifier",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_decision_tree_classifier",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-knn-classifier": KedroOperator(
            task_id="train-knn-classifier",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_knn_classifier",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-linear-regression": KedroOperator(
            task_id="train-linear-regression",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_linear_regression",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-logistic-regression": KedroOperator(
            task_id="train-logistic-regression",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_logistic_regression",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-random-forest": KedroOperator(
            task_id="train-random-forest",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_random_forest",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-random-forest-classifier": KedroOperator(
            task_id="train-random-forest-classifier",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_random_forest_classifier",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-svm-classifier": KedroOperator(
            task_id="train-svm-classifier",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_svm_classifier",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-svr": KedroOperator(
            task_id="train-svr",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_svr",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "train-xgboost": KedroOperator(
            task_id="train-xgboost",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_xgboost",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "consolidate-classification-metrics": KedroOperator(
            task_id="consolidate-classification-metrics",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="consolidate_classification_metrics",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "consolidate-regression-metrics": KedroOperator(
            task_id="consolidate-regression-metrics",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="consolidate_regression_metrics",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "create-classification-visualization": KedroOperator(
            task_id="create-classification-visualization",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="create_classification_visualization",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "create-regression-visualization": KedroOperator(
            task_id="create-regression-visualization",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="create_regression_visualization",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        ),
        "generate-final-report": KedroOperator(
            task_id="generate-final-report",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="generate_final_report",
            project_path=project_path,
            env=env,
            conf_source=conf_source,
        )
    }
    tasks["combinar-raw-node"] >> tasks["limpiar-datos-node"]
    tasks["limpiar-datos-node"] >> tasks["eliminar-atipicos-node"]
    tasks["eliminar-atipicos-node"] >> tasks["preparar-datos-node"]
    tasks["preparar-datos-node"] >> tasks["crear-kdr-node"]
    tasks["crear-kdr-node"] >> tasks["crear-impact-score-node"]
    tasks["crear-impact-score-node"] >> tasks["train-decision-tree"]
    tasks["crear-impact-score-node"] >> tasks["train-decision-tree-classifier"]
    tasks["crear-impact-score-node"] >> tasks["train-knn-classifier"]
    tasks["crear-impact-score-node"] >> tasks["train-linear-regression"]
    tasks["crear-impact-score-node"] >> tasks["train-logistic-regression"]
    tasks["crear-impact-score-node"] >> tasks["train-random-forest"]
    tasks["crear-impact-score-node"] >> tasks["train-random-forest-classifier"]
    tasks["crear-impact-score-node"] >> tasks["train-svm-classifier"]
    tasks["crear-impact-score-node"] >> tasks["train-svr"]
    tasks["crear-impact-score-node"] >> tasks["train-xgboost"]
    tasks["train-knn-classifier"] >> tasks["consolidate-classification-metrics"]
    tasks["train-random-forest-classifier"] >> tasks["consolidate-classification-metrics"]
    tasks["train-decision-tree-classifier"] >> tasks["consolidate-classification-metrics"]
    tasks["train-svm-classifier"] >> tasks["consolidate-classification-metrics"]
    tasks["train-logistic-regression"] >> tasks["consolidate-classification-metrics"]
    tasks["train-linear-regression"] >> tasks["consolidate-regression-metrics"]
    tasks["train-random-forest"] >> tasks["consolidate-regression-metrics"]
    tasks["train-xgboost"] >> tasks["consolidate-regression-metrics"]
    tasks["train-decision-tree"] >> tasks["consolidate-regression-metrics"]
    tasks["train-svr"] >> tasks["consolidate-regression-metrics"]
    tasks["consolidate-classification-metrics"] >> tasks["create-classification-visualization"]
    tasks["consolidate-regression-metrics"] >> tasks["create-regression-visualization"]
    tasks["consolidate-regression-metrics"] >> tasks["generate-final-report"]
    tasks["consolidate-classification-metrics"] >> tasks["generate-final-report"]