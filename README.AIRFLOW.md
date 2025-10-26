# ✈️ Apache Airflow - Instalación y Uso (Rainbow Six ML)

Guía práctica para levantar Apache Airflow con Docker y orquestar tus pipelines de Kedro en este proyecto.

---

## 🧰 Prerrequisitos

- Windows 10/11 con Docker Desktop (WSL2 recomendado)
- 6–8 GB de RAM asignada a Docker (Settings → Resources)
- Puertos libres: 8080 (Airflow Web UI)

Estructura relevante del repo:
- `docker-compose.airflow.yml` (orquestación de Airflow)
- `docker/Dockerfile.airflow` (imagen base de Airflow con requirements)
- `proyecto-ml/dags/` (DAGs de Airflow del proyecto)
- `proyecto-ml/src/`, `proyecto-ml/conf/`, `proyecto-ml/data/`, `proyecto-ml/logs/` (mapeados al contenedor)

---

## 🚀 Inicio Rápido

Todos los comandos son para PowerShell en la carpeta raíz del repo.

1) Construir imágenes

```powershell
docker compose -f docker-compose.airflow.yml build
```

2) Levantar el stack (DB, init, webserver, scheduler)

```powershell
docker compose -f docker-compose.airflow.yml up -d
```

3) Acceso a la UI

- URL: http://localhost:8080
- Usuario: `admin`
- Password: `admin`

4) Verificar DAGs cargados

```powershell
docker compose -f docker-compose.airflow.yml exec airflow-webserver airflow dags list
```

Deberías ver:
- `modelo_clasificacion`
- `modelo_regresion`
- `consolidate_model_results`

---

## 📦 ¿Qué contiene la imagen de Airflow?

- Base: `apache/airflow:2.8.4-python3.11`
- Instala `requirements.txt` usando constraints oficiales de Airflow (compatibilidad segura)
- Variables de entorno preparadas para Kedro dentro del contenedor:
  - `KEDRO_HOME=/app/proyecto-ml`
  - `PYTHONPATH=/app/proyecto-ml/src`
  - `KEDRO_ENV=production`

Bind mounts (directorios del host mapeados al contenedor):
- `./proyecto-ml/dags -> /opt/airflow/dags`
- `./proyecto-ml/logs -> /opt/airflow/logs`
- `./proyecto-ml/src -> /app/proyecto-ml/src`
- `./proyecto-ml/conf -> /app/proyecto-ml/conf`
- `./proyecto-ml/data -> /app/proyecto-ml/data`

---

## 🗂️ DAGs del proyecto

Los DAGs actuales usan BashOperator para ejecutar pipelines de Kedro:
- `proyecto-ml/dags/modelo_clasificacion_dag.py` → `kedro run --pipeline modelo_clasificacion`
- `proyecto-ml/dags/modelo_regresion_dag.py` → `kedro run --pipeline modelo_regresion`
- `proyecto-ml/dags/consolidate_results_dag.py` → Espera a los dos anteriores y consolida JSONs en `08_reporting/model_comparison.json`

Nota: Los DAGs leen `KEDRO_PROJECT_PATH=/app/proyecto-ml` (ya configurado en el compose) y exportan `PYTHONPATH` a `/app/proyecto-ml/src`.

---

## ▶️ Ejecutar y programar DAGs

- Despausar un DAG desde la UI (switch en la vista del DAG).
- Ejecutar manualmente desde la UI (Trigger DAG).

CLI opcional:

```powershell
# Ejecutar un test de DAG (parse + run lógico en fecha dada)
docker compose -f docker-compose.airflow.yml exec airflow-webserver `
  airflow dags test modelo_clasificacion 2025-10-01

# Mostrar estructura del DAG
docker compose -f docker-compose.airflow.yml exec airflow-webserver `
  airflow dags show modelo_regresion
```

Programación (schedule_interval) dentro de cada DAG:
- `@once`, `@hourly`, `@daily`, `@weekly`, `@monthly`
- Cron, p.ej. `"0 */2 * * *"` (cada 2 horas)

---

## 🧪 Crear un DAG nuevo (rápido)

Crea `proyecto-ml/dags/mi_nuevo_dag.py` con un BashOperator simple:

```python
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_CONTAINER_PATH = os.environ.get("KEDRO_PROJECT_PATH", "/app/proyecto-ml")

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="mi_nuevo_dag",
    start_date=datetime(2025, 10, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    run_pipeline = BashOperator(
        task_id="run_pipeline",
        bash_command=f"cd {PROJECT_CONTAINER_PATH} && kedro run --pipeline rainbow_six",
        env={"PYTHONPATH": f"{PROJECT_CONTAINER_PATH}/src"},
    )
```

Guarda el archivo y Airflow lo cargará automáticamente (los DAGs están bind-montados).

---

## 🔍 Diagnóstico y logs

Comandos útiles:

```powershell
# Ver logs del init (migraciones y creación de usuario)
docker compose -f docker-compose.airflow.yml logs --tail=200 airflow-init

# Ver carpeta de DAGs en el contenedor
docker compose -f docker-compose.airflow.yml exec airflow-webserver ls -la /opt/airflow/dags

# Probar sintaxis de un DAG específico
docker compose -f docker-compose.airflow.yml exec airflow-webserver `
  python /opt/airflow/dags/modelo_clasificacion_dag.py

# Ver logs de una tarea
docker compose -f docker-compose.airflow.yml exec airflow-scheduler `
  airflow tasks logs modelo_clasificacion run_modelo_clasificacion_task 2025-10-01
```

---

## 🧯 Troubleshooting

1) Error de migración/atributos (p.ej., `AttributeError: execution_date`)

Reset completo del stack (borra metadata de Airflow):
```powershell
docker compose -f docker-compose.airflow.yml down -v
docker compose -f docker-compose.airflow.yml up -d
```

2) Los DAGs no aparecen en la UI
```powershell
# Ver carpeta DAGs en contenedor
docker compose -f docker-compose.airflow.yml exec airflow-webserver ls -la /opt/airflow/dags

# Probar sintaxis
docker compose -f docker-compose.airflow.yml exec airflow-webserver `
  python /opt/airflow/dags/modelo_regresion_dag.py

# Reiniciar scheduler
docker compose -f docker-compose.airflow.yml restart airflow-scheduler
```

3) Conflicto de puertos (8080 en uso)
- Edita `docker-compose.airflow.yml` para publicar otro puerto, p.ej. `"8081:8080"`.

4) Contenedores huérfanos (warning de orphans)
```powershell
docker compose -f docker-compose.airflow.yml up -d --remove-orphans
```

---

## 🧩 Integración con Kedro

- Dentro del contenedor, los DAGs ejecutan `kedro run` con `PYTHONPATH=/app/proyecto-ml/src`.
- Si quieres generar DAGs desde pipelines automáticamente, puedes usar `kedro-airflow` (ya en `requirements.txt`):

```powershell
# (Opcional) dentro del contenedor de Airflow o en tu venv
kedro airflow create --pipeline data_processing
```

---

## 🧹 Parar y limpiar

```powershell
# Parar servicios
docker compose -f docker-compose.airflow.yml down

# Parar y borrar volúmenes (reset completo)
docker compose -f docker-compose.airflow.yml down -v
```

---

## ✅ Resumen

- `docker compose -f docker-compose.airflow.yml build`
- `docker compose -f docker-compose.airflow.yml up -d`
- UI en `http://localhost:8080` (admin/admin)
- DAGs en `proyecto-ml/dags` → montados en `/opt/airflow/dags`
- Pipelines de Kedro se ejecutan vía BashOperator con el entorno ya configurado.
