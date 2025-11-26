# Arquitectura del Proyecto Rainbow Six ML

## 1. Visión General
Este proyecto implementa una plataforma de MLOps para el procesamiento, entrenamiento y evaluación de modelos de analítica y machine learning sobre datos del ecosistema de Rainbow Six. Integra:
- Orquestación con **Airflow**.
- Estructuración y modularización de pipelines con **Kedro**.
- Versionado de datos y artefactos con **DVC**.
- Seguimiento de experimentos y modelos con **MLflow** (carpeta `mlruns/`).
- Contenedorización con **Docker** y ambientes diferenciados (Jupyter, Kedro runtime, Airflow).
- Monitoreo con **Prometheus** (configuración en `monitoring/prometheus.yml`).

## 2. Componentes Principales
### 2.1 Kedro
Organiza el flujo de datos en capas bajo `proyecto-ml/data/` y catálogos definidos en `conf/base/catalog.yml`. Las capas siguen buenas prácticas:
- `01_raw` → Datos originales (versionados con DVC).
- `02_intermediate`, `03_primary`, `04_feature` → Limpieza, normalización y feature engineering.
- `05_model_input` → Conjuntos para entrenamiento / validación.
- `06_models` → Artefactos de modelos entrenados.
- `07_model_output` → Predicciones / inferencias.
- `08_reporting` → Resultados agregados / métricas para negocio.

### 2.2 Airflow
Los DAGs en `proyecto-ml/dags/` automatizan pipelines de:
- Clasificación (`classification_pipeline_dag.py`).
- Regresión (`regression_pipeline_dag.py`).
- Comparación de modelos (`model_comparison_dag.py`).
- Aprendizaje no supervisado (`unsupervised_learning_dags.py`).
- Procesos Rainbow Six específicos (ej. ingestión diaria / semanal). Logs se almacenan en `proyecto-ml/logs/` segmentados por `dag_id=`.

### 2.3 DVC
Controla versiones de datos y artefactos mediante `dvc.yaml` y almacenamiento remoto en `data/dvc_storage/`. Permite reproducibilidad exacta de transformaciones y asegura trazabilidad en cambios de datasets.

### 2.4 MLflow
La carpeta `mlruns/` evidencia tracking de experimentos (parámetros, métricas, artefactos). Debe integrarse con los nodos finales de Kedro y/o DAGs para registrar:
- Parámetros de entrenamiento (hyperparámetros).
- Métricas de validación.
- Modelos serializados.
- Etiquetado de versiones promovidas a producción.

### 2.5 Docker & Compose
Multiples Dockerfiles:
- `Dockerfile.kedro` → Ejecución pipelines.
- `Dockerfile.jupyter` → Exploración / prototipado.
- `Dockerfile.airflow` → Scheduler + Webserver.
Y archivos `docker-compose.yml` / `docker-compose.airflow.yml` para orquestar servicios locales. Facilita aislamiento y reproducibilidad.


## 3. Flujo de Datos End-to-End
1. Ingestión cruda → almacenada en `01_raw` y versionada por DVC.
2. Transformaciones incrementales (limpieza, enriquecimiento, agregaciones) pasan por capas intermedias.
3. Feature engineering consolida variables en `04_feature` y conjuntos listos en `05_model_input`.
4. Entrenamiento genera modelos persistidos en `06_models` y métricas + predicciones en `07_model_output`.
5. Reporting y comparaciones finalizan en `08_reporting` y se exponen para negocio / dashboards.

## 4. Flujo de Entrenamiento y Orquestación
- Airflow dispara DAGs según frecuencia (diaria / semanal). Cada DAG invoca scripts Kedro o comandos CLI.
- Parametrización centralizada en archivos `parameters_*.yml` dentro de `conf/base/` (ej. clasificación, regresión, comparación, no supervisado).
- Reutilización de nodos: separar claramente nodos de extracción, transformación, modelado, evaluación.

## 5. Gestión de Experimentos
Integrar MLflow en nodos de entrenamiento para registrar automáticamente:
- `run_name` = combinación de modelo + timestamp + dataset.
- Métricas core por tipo de problema (accuracy, precision, recall, F1 para clasificación; RMSE, MAE, R² para regresión; silhouette, Davies-Bouldin para clustering).
- Artifacts: modelo pickled, gráficos (curvas ROC, importancia de features, clusters).



## 6. Mapeo Notebooks ↔ Proposito
| Notebook | Propósito |  
|----------|-----------|
| 01_business_understanding | Contexto de negocio |
| 02_data_understanding | Exploración y profiling |
| 03_data_preparation | Limpieza y transformaciones |
| 04_unsupervised_learning | Clustering / reducción dimensión |
| 05_final_analysis | Síntesis resultados 
| Modelos clasificación (5) | Entrenamiento individual |
| Modelos regresión (5) | Entrenamiento individual |




