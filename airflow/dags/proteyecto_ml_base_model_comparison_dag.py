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
pipeline_name = "model_comparison"
project_path = Path.cwd()
package_name = "proteyecto_ml"
conf_source = "" or Path.cwd() / "conf"


# Using a DAG context manager, you don't have to specify the dag property of each task
with DAG(
    dag_id="proteyecto-ml-model-comparison",
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
    tasks["consolidate-classification-metrics"] >> tasks["create-classification-visualization"]
    tasks["consolidate-regression-metrics"] >> tasks["create-regression-visualization"]
    tasks["consolidate-regression-metrics"] >> tasks["generate-final-report"]
    tasks["consolidate-classification-metrics"] >> tasks["generate-final-report"]