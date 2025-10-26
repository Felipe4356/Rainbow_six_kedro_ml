from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from airflow.models import BaseOperator


class KedroOperator(BaseOperator):
    """
    Airflow operator to run a Kedro pipeline inside the Airflow worker.

    Parameters:
        pipeline_name: Name of the Kedro pipeline to run (as registered in pipeline_registry).
        project_path: Absolute path to the Kedro project inside the container.
        env: Kedro environment to use (e.g., "base", "local").
        extra_params: Optional extra parameters to pass to session.run.
    """

    def __init__(
        self,
        *,
        pipeline_name: str,
        project_path: Optional[str] = None,
        env: str = "base",
        extra_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pipeline_name = pipeline_name
        self.project_path = project_path or os.environ.get(
            "KEDRO_PROJECT_PATH", "/opt/airflow/kedro_project"
        )
        self.env = env
        self.extra_params = extra_params or {}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Set safe defaults to reduce memory/CPU pressure inside containers
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MPLBACKEND", "Agg")

        # Ensure the Kedro project source directory is on sys.path
        src_path = os.path.join(self.project_path, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # Import here so Airflow scheduler doesn't require Kedro on import time
        from kedro.framework.session import KedroSession

        self.log.info(
            "Running Kedro pipeline '%s' [env=%s] at project_path=%s",
            self.pipeline_name,
            self.env,
            self.project_path,
        )

        # Create a Kedro session and run the pipeline
        with KedroSession.create(project_path=self.project_path, env=self.env) as session:
            session.run(pipeline_name=self.pipeline_name, extra_params=self.extra_params)

        result = {
            "pipeline_name": self.pipeline_name,
            "env": self.env,
            "project_path": self.project_path,
        }
        self.log.info("Kedro pipeline finished: %s", result)
        return result
