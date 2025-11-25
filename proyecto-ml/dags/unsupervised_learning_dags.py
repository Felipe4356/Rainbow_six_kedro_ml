"""DAG para ejecutar el pipeline `clustering` de Kedro

Se ejecuta en el entorno `local` para usar `conf/local/parameters.yml`.
"""
from datetime import datetime, timedelta
import os

import importlib

# Intentamos cargar Airflow dinámicamente para evitar errores en editores sin el paquete
_airflow_spec = importlib.util.find_spec('airflow')
if _airflow_spec is not None:
	airflow = importlib.import_module('airflow')
	operators_bash = importlib.import_module('airflow.operators.bash')
	DAG = airflow.DAG
	BashOperator = operators_bash.BashOperator
else:
	from ._airflow_stubs import DAG, BashOperator  # type: ignore

DEFAULT_ARGS = {
	'owner': 'airflow',
	'depends_on_past': False,
	'retries': 1,
	'retry_delay': timedelta(minutes=5),
}

PROJECT_DIR = os.environ.get('PROJECT_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with DAG(
	dag_id='unsupervised_learning_pipeline',
	default_args=DEFAULT_ARGS,
	description='Ejecuta el pipeline de unsupervised learning',
	schedule_interval=None,
	start_date=datetime(2025, 1, 1),
	catchup=False,
) as dag:

	bash = (
		f'set -euxo pipefail; '
		f'echo PROJECT_DIR={PROJECT_DIR}; '
		'echo PATH=$PATH; '
		f'if [ -f "{PROJECT_DIR}/pyproject.toml" ]; then cd "{PROJECT_DIR}"; '
		'elif [ -f "/app/proyecto-ml/pyproject.toml" ]; then cd /app/proyecto-ml; '
		'elif [ -f "/opt/airflow/proyecto-ml/pyproject.toml" ]; then cd /opt/airflow/proyecto-ml; '
		'elif [ -f "/opt/airflow/pyproject.toml" ]; then cd /opt/airflow; '
		'elif [ -f "/home/airflow/proyecto-ml/pyproject.toml" ]; then cd /home/airflow/proyecto-ml; '
		'else echo "pyproject.toml not found in PROJECT_DIR or known paths"; ls -la "{PROJECT_DIR}" || true; ls -la /opt/airflow || true; exit 2; fi; '
		'command -v kedro >/dev/null 2>&1 || { echo "kedro not found in PATH"; exit 1; }; '
		'kedro run --pipeline=unsupervised_learning --env local'
	)

	run_unsupervised = BashOperator(
		task_id='run_unsupervised_learning_pipeline',
		bash_command=bash,
	)

	run_unsupervised