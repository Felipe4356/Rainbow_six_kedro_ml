"""Stubs mínimos para permitir edición y análisis estático cuando Airflow no está instalado.

Estos stubs no se usan en producción: en tiempo de ejecución los DAGs importan Airflow
dinámicamente si está disponible; si no, estos stubs evitan errores de editor/Pylance.
"""
from datetime import datetime
from typing import Any, Dict


class DAG:
    def __init__(self, dag_id: str = "", default_args: Dict[str, Any] = None, description: str = "", schedule_interval=None, start_date: datetime = None, catchup: bool = False):
        self.dag_id = dag_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class BashOperator:
    def __init__(self, task_id: str, bash_command: str, **kwargs):
        self.task_id = task_id
        self.bash_command = bash_command

    def __repr__(self):
        return f"<BashOperator {self.task_id}>"
