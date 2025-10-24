# Airflow: Guía rápida y comandos (Windows/PowerShell)

Este README explica cómo levantar, administrar y diagnosticar Apache Airflow para este proyecto. Todo está orquestado con Docker Compose y los DAGs viven en `airflow/dags`.

URL UI: http://localhost:8080  
Credenciales: usuario `admin` · contraseña `admin`

## Servicios y archivos clave
- Compose: `docker-compose.airflow.yml`
- Imagen: `docker/Dockerfile.airflow`
- DAGs: `airflow/dags/*.py`
- Volúmenes: Postgres (metadatos), logs y plugins

DAG IDs actuales en este repo:
- `proteyecto-ml` (pipeline completo)
- `proteyecto-ml-rainbow-six` (preparación de datos)
- `proteyecto-ml-clasificacion` (modelo clasificación)
- `proteyecto-ml-regresion` (modelo regresión)
- `proteyecto-ml-model-comparison` (consolidación/reportes)

## Levantar y detener Airflow

```powershell
# Arrancar Airflow (webserver + scheduler + postgres)
docker compose -f docker-compose.airflow.yml up -d --build

# Ver estado de contenedores
docker compose -f docker-compose.airflow.yml ps

# Ver logs del webserver (salir con Ctrl+C)
docker compose -f docker-compose.airflow.yml logs airflow-webserver --no-color -f

# Detener sólo Airflow (mantener volúmenes)
docker compose -f docker-compose.airflow.yml stop

# Apagar y borrar contenedores (conservar volúmenes)
docker compose -f docker-compose.airflow.yml down

# Apagar y borrar TODO (incluye volúmenes de Postgres: ¡pierde histórico!)
docker compose -f docker-compose.airflow.yml down -v
```

## Comandos útiles dentro del contenedor
Usa el webserver para ejecutar la CLI de Airflow:

```powershell
# Listar DAGs
docker exec rainbow-six-airflow-webserver airflow dags list

# Ver errores de importación de DAGs
docker exec rainbow-six-airflow-webserver airflow dags list-import-errors

# Pausar / despausar un DAG
docker exec rainbow-six-airflow-webserver airflow dags pause proteyecto-ml-regresion
docker exec rainbow-six-airflow-webserver airflow dags unpause proteyecto-ml-regresion

# Disparar un DAG manualmente
docker exec rainbow-six-airflow-webserver airflow dags trigger proteyecto-ml-rainbow-six

# Listar ejecuciones de un DAG
docker exec rainbow-six-airflow-webserver airflow dags list-runs -d proteyecto-ml-rainbow-six

# Ver tareas de un DAG (Task IDs)
docker exec rainbow-six-airflow-webserver airflow tasks list proteyecto-ml

# Log de una tarea específica (reemplaza con run_id y task_id reales)
docker exec rainbow-six-airflow-webserver airflow tasks logs proteyecto-ml <task_id> --dag-run-id <run_id>

# Re-ejecutar histórico (backfill por fecha de inicio)
docker exec rainbow-six-airflow-webserver airflow dags backfill -s 2023-01-01 proteyecto-ml

# Limpiar estados de tareas (volver a encolar fallidas)
docker exec rainbow-six-airflow-webserver airflow tasks clear -y proteyecto-ml
```

## Variables y Conexiones (opcional)

```powershell
# Variables
# Importar desde JSON (dentro del contenedor)
# docker exec -i rainbow-six-airflow-webserver airflow variables import /path/vars.json

# Setear una variable simple
docker exec rainbow-six-airflow-webserver airflow variables set ENV base

# Conexiones
# Crear conexión (ejemplo Postgres)
docker exec rainbow-six-airflow-webserver airflow connections add my_postgres ^
  --conn-type postgres ^
  --conn-host airflow-postgres ^
  --conn-schema airflow ^
  --conn-login airflow ^
  --conn-password airflow ^
  --conn-port 5432
```

## Recarga de DAGs y sincronización de código
- Los DAGs se montan por volumen desde `airflow/dags`; los cambios se detectan automáticamente (espera ~30-60s o refresca la UI).
- Para forzar refresco rápido, reinicia webserver y/o scheduler:

```powershell
docker compose -f docker-compose.airflow.yml restart airflow-webserver airflow-scheduler
```

## Salud y logs del sistema

```powershell
# Healthcheck del webserver
curl http://localhost:8080/health

# Logs del scheduler
docker compose -f docker-compose.airflow.yml logs airflow-scheduler --no-color -f

# Logs del Postgres (metadatos)
docker compose -f docker-compose.airflow.yml logs airflow-postgres --no-color -f
```

## Inicialización (se hace automáticamente)
El servicio `airflow-init` ya ejecuta:
- Migración de DB (`airflow db migrate`)
- Creación del usuario admin (`admin/admin`)
- Generación de DAGs desde Kedro (si aplica)

Si necesitas re-inicializar desde cero:

```powershell
# Opción nuclear: borra volúmenes y re-crea la base de datos (pierdes histórico)
docker compose -f docker-compose.airflow.yml down -v

docker compose -f docker-compose.airflow.yml up -d --build
```

## Solución de problemas comunes
- No veo mis DAGs:
  - Revisa import errors:
    ```powershell
    docker exec rainbow-six-airflow-webserver airflow dags list-import-errors
    ```
  - Verifica que cada DAG tiene un `dag_id` único (en este repo ya están separados por pipeline).
  - Asegúrate de que el código del proyecto está montado: `./proyecto-ml:/app`.
- Conflicto de puerto 8080 (otra app ocupando 8080):
  - Detén temporalmente el webserver o cambia el mapeo de puerto en `docker-compose.airflow.yml` (por ejemplo a `8081:8080`).
- Errores de permisos con archivos en Windows:
  - Evita rutas con caracteres especiales o muy largas y revisa antivirus/bloqueo de archivos.
- La UI no refresca:
  - Pulsa el icono de recarga, espera un minuto o reinicia `airflow-webserver`.

## Notas del proyecto
- Este Airflow usa `LocalExecutor` y Postgres como metastore.
- El entorno de Kedro se configura con variables de entorno (`KEDRO_HOME=/app`, `PYTHONPATH=/app/src`).
- Los DAGs de este repo llaman a pipelines de Kedro a través de `KedroSession`.

¿Quieres que también agregue ejemplos para programar ejecuciones (cron) o dejar todo en ejecución manual (@once)?
