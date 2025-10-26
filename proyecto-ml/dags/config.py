from __future__ import annotations

import os
from datetime import timedelta

# Base paths and environment
PROJECT_CONTAINER_PATH = os.environ.get("KEDRO_PROJECT_PATH", "/opt/airflow/kedro_project")
KEDRO_ENV = os.environ.get("KEDRO_ENV", "base")

# Default args shared by DAGs
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Common schedules
SCHEDULES = {
    "daily_ml": "0 2 * * *",  # daily at 02:00
    "every_4h": "0 */4 * * *",  # every 4 hours
    "weekly": "0 3 * * 1",  # weekly on Monday at 03:00
}
