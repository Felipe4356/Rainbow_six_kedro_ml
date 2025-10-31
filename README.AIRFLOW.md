# ✈️ Docker + Airflow + DVC (Rainbow Six ML)

Guía práctica para levantar Apache Airflow con Docker, orquestar pipelines de Kedro y reproducir resultados con DVC en este proyecto.

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

* classification_pipeline 
* model_comparison_pipeline 
* regression_pipeline   
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

Los DAGs actuales (archivos en `proyecto-ml/dags/`) son:

- `proyecto-ml/dags/classification_pipeline_dag.py` → DAG para el pipeline de clasificación: limpieza de datos, entrenamiento y evaluación del modelo de clasificación.
- `proyecto-ml/dags/regression_pipeline_dag.py` → DAG para el pipeline de regresión: preparación, entrenamiento y evaluación del modelo de regresión.
- `proyecto-ml/dags/model_comparison_dag.py` → DAG que compara resultados entre modelos (consolida métricas y genera reportes).

Todos los DAGs están montados en `/opt/airflow/dags` dentro del contenedor y, por convención, exponen los siguientes dag_ids (tal como aparecen en la UI y en los comandos de Airflow):

- `classification_pipeline`
- `regression_pipeline`
- `model_comparison_pipeline`

Nota: Los DAGs usan `KEDRO_PROJECT_PATH=/app/proyecto-ml` (configurado en el compose) y agarran `PYTHONPATH=/app/proyecto-ml/src` cuando ejecutan `kedro run`.

---

## ▶️ Ejecutar y programar DAGs

- Despausar un DAG desde la UI (switch en la vista del DAG).
- Ejecutar manualmente desde la UI (Trigger DAG).

CLI opcional:

```powershell
# Ejecutar un test de DAG (parse + run lógico en fecha dada)
docker compose -f docker-compose.airflow.yml exec airflow-webserver `
  airflow dags test classification_pipeline 2025-10-01

# Mostrar estructura del DAG
docker compose -f docker-compose.airflow.yml exec airflow-webserver `
  airflow dags show regression_pipeline
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
  python /opt/airflow/dags/classification_pipeline_dag.py

# Ver logs de una tarea
docker compose -f docker-compose.airflow.yml exec airflow-scheduler `
  airflow tasks logs classification_pipeline run_pipeline 2025-10-01
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
  python /opt/airflow/dags/regression_pipeline_dag.py

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

## 🏃‍♂️ Ejecución con Kedro (local, opcional)

Puedes ejecutar los pipelines desde tu máquina (fuera de Docker):

```powershell
# Ejecutar por etapas
kedro run --pipeline=rainbow_six
kedro run --pipeline=modelo_clasificacion
kedro run --pipeline=modelo_regresion
kedro run --pipeline=model_comparison
```

---

## 🔁 Reproducibilidad con DVC (opcional)

Reproduce el flujo y consulta métricas versionadas:

```powershell
# Reproducir todo el flujo hasta comparación
dvc repro model_comparison

# Ver métricas rastreadas
# (si configuraste DVC previamente: dvc init / dvc remote add ...)
dvc metrics show
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
