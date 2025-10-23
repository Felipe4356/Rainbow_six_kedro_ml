# 🐳 Docker Configuration - Rainbow Six ML Pipeline

Esta carpeta contiene la configuración Docker completa para el proyecto de Machine Learning de Rainbow Six Siege, con servicios optimizados para desarrollo y producción.

---

## 📁 Estructura de Archivos

```
docker/
├── Dockerfile.kedro              # Imagen para Kedro ML Pipeline
├── Dockerfile.jupyter            # Imagen para JupyterLab con extensiones ML
├── README.md                    # Esta documentación
└── .dockerignore               # Archivos excluidos del build

../
├── docker-compose.yml           # Orquestación principal de servicios
├── docker-compose.override.yml  # Configuración para desarrollo local
├── docker-ml.bat               # Script de gestión para Windows
├── .env                        # Variables de entorno
└── scripts/
    └── init-db.sql             # Inicialización de base de datos
```

---

## 🚀 Inicio Rápido

### ✅ Estado Actual del Sistema

**Los siguientes servicios están FUNCIONANDO correctamente:**

- **📊 Kedro Viz**: http://localhost:4141 (Visualización de pipelines)
- **📓 Jupyter Lab**: http://localhost:8888 (Desarrollo interactivo)

### 🎯 Comandos de Gestión Rápida

```bash
# ═══════════════════════════════════════════
# GESTIÓN PRINCIPAL (Usar script Windows)
# ═══════════════════════════════════════════

# Ver estado actual
docker-ml.bat status

# Iniciar entorno de desarrollo completo
docker-ml.bat dev

# Iniciar entorno de producción
docker-ml.bat prod

# Ver logs en tiempo real
docker-ml.bat logs

# Parar todos los servicios
docker-ml.bat stop

# Limpiar sistema completo
docker-ml.bat clean
```

### 🔧 Comandos Docker Compose Directos

```bash
# ═══════════════════════════════════════════
# PERFILES DISPONIBLES
# ═══════════════════════════════════════════

# Desarrollo: Jupyter + Kedro Viz + Redis
docker-compose --profile development up -d

# Producción: Kedro + Scheduler + PostgreSQL + Redis
docker-compose --profile production up -d

# Base de datos: Solo PostgreSQL
docker-compose --profile database up -d

# MLflow: Tracking de experimentos
docker-compose --profile mlflow up -d

# Monitoreo: Prometheus (opcional)
docker-compose --profile monitoring up -d

# Cache: Solo Redis
docker-compose --profile cache up -d
```

---

## 🏗️ Servicios Disponibles

### 🎮 Servicios Principales

| 🔧 Servicio | 🌐 Puerto | 📋 Descripción | 🏷️ Perfil |
|-------------|-----------|----------------|-----------|
| **kedro-viz** | **4141** | **Visualización interactiva de pipelines ML** | `development`, `production` |
| **jupyter-lab** | **8888** | **JupyterLab para desarrollo de modelos** | `development` |
| **kedro-prod** | - | **Ejecución de pipelines en producción** | `production` |
| **kedro-scheduler** | - | **Scheduler automático (cada 6 horas)** | `production` |

### 💾 Servicios de Datos

| 🔧 Servicio | 🌐 Puerto | 📋 Descripción | 🏷️ Perfil |
|-------------|-----------|----------------|-----------|
| **postgres** | **5433** | **Base de datos para métricas y resultados** | `database`, `production` |
| **redis** | **6379** | **Cache para modelos y resultados** | `cache`, `development` |
| **mlflow** | **5000** | **Tracking de experimentos ML** | `mlflow`, `development` |

### 📊 Servicios de Monitoreo

| 🔧 Servicio | 🌐 Puerto | 📋 Descripción | 🏷️ Perfil |
|-------------|-----------|----------------|-----------|
| **prometheus** | **9090** | **Monitoreo de métricas del sistema** | `monitoring` |

---

## 🎯 Casos de Uso Específicos

### 🧪 Desarrollo y Experimentación
```bash
# Iniciar entorno completo de desarrollo
docker-compose --profile development up -d

# Servicios incluidos:
# ✅ Jupyter Lab: http://localhost:8888
# ✅ Kedro Viz: http://localhost:4141  
# ✅ Redis: localhost:6379
# ✅ PostgreSQL: localhost:5433 (dev)
# ✅ MLflow: http://localhost:5000

# Acceder a Jupyter para desarrollo
start http://localhost:8888

# Visualizar pipelines ML
start http://localhost:4141
```

### 🏭 Producción y Deployment
```bash
# Iniciar entorno de producción
docker-compose --profile production up -d

# Servicios incluidos:
# ✅ Kedro Scheduler (ejecuta cada 6 horas)
# ✅ Kedro Viz: http://localhost:4141
# ✅ PostgreSQL: localhost:5433 (prod)
# ✅ Redis: localhost:6379

# Verificar logs del scheduler
docker-compose logs kedro-scheduler -f
```

### 🔬 Solo Experimentación ML
```bash
# Solo MLflow para tracking
docker-compose --profile mlflow up -d

# Acceder a MLflow UI
start http://localhost:5000
```

---

## 💻 Comandos de Ejecución de Pipelines

### 🐳 Dentro de Contenedores Docker

```bash
# ═══════════════════════════════════════════
# ACCESO A CONTENEDORES
# ═══════════════════════════════════════════

# Acceder al contenedor de Kedro
docker exec -it rainbow-six-kedro-viz bash

# Acceder al contenedor de Jupyter
docker exec -it rainbow-six-jupyter bash

# ═══════════════════════════════════════════
# EJECUCIÓN DE PIPELINES ML
# ═══════════════════════════════════════════

# Pipeline completo (todos los modelos)
kedro run

# Pipelines específicos
kedro run --pipeline=rainbow_six           # Preparación de datos
kedro run --pipeline=modelo_clasificacion  # 5 modelos clasificación
kedro run --pipeline=modelo_regresion      # 5 modelos regresión  
kedro run --pipeline=model_comparison      # Comparación final

# Visualización interactiva
kedro viz --host=0.0.0.0 --port=4141

# Jupyter dentro del contenedor
kedro jupyter lab --ip=0.0.0.0 --port=8888
```

### 📊 Comandos de Información

```bash
# Ver información del proyecto
kedro info

# Listar todos los pipelines disponibles
kedro registry list

# Ver datasets configurados
kedro catalog list

# Verificar configuración
kedro config list
```

---

## 🏗️ Arquitectura de Contenedores

### 🐳 Imagen Kedro (`Dockerfile.kedro`)
```dockerfile
# Características principales:
✅ Python 3.9-slim (optimizada)
✅ Usuario no-root (mluser) para seguridad
✅ Kedro + todas las librerías ML
✅ XGBoost, Scikit-learn, Pandas incluidos
✅ Health check automático
✅ Volúmenes persistentes configurados
```

### 📓 Imagen Jupyter (`Dockerfile.jupyter`)
```dockerfile
# Características principales:
✅ JupyterLab 4.0+ con extensiones ML
✅ Plotly, IPywidgets para visualización
✅ Kedro[notebook] integrado
✅ Configuración automática
✅ Acceso sin token (desarrollo)
```

### 🌐 Red y Volúmenes
```yaml
# Red personalizada
rainbow-six-network: Comunicación entre servicios

# Volúmenes persistentes
postgres_data: Base de datos PostgreSQL
redis_data: Cache Redis
prometheus_data: Métricas Prometheus

# Bind mounts
./proyecto-ml/data → /app/data
./proyecto-ml/logs → /app/logs
./proyecto-ml/notebooks → /app/notebooks
```

---

## 🔧 Variables de Entorno

### 📄 Archivo `.env` (configuración principal)
```bash
# ═══════════════════════════════════════════
# CONFIGURACIÓN DE BASE DE DATOS
# ═══════════════════════════════════════════
POSTGRES_DB=rainbow_six_ml
POSTGRES_USER=ml_user
POSTGRES_PASSWORD=rainbow123

# ═══════════════════════════════════════════
# CONFIGURACIÓN DE KEDRO
# ═══════════════════════════════════════════
KEDRO_ENV=production                    # local / production
KEDRO_HOME=/app
KEDRO_CONFIG_FILE=conf/base/parameters.yml

# ═══════════════════════════════════════════
# CONFIGURACIÓN DE MLFLOW
# ═══════════════════════════════════════════
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=rainbow_six_experiments

# ═══════════════════════════════════════════
# CONFIGURACIÓN DE PYTHON
# ═══════════════════════════════════════════
PYTHONPATH=/app/src
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

### 🎛️ Variables por Servicio
```bash
# Kedro Production
KEDRO_ENV=production
PYTHONPATH=/app/src

# Jupyter Development  
KEDRO_ENV=local
JUPYTER_ENABLE_LAB=yes
JUPYTER_TOKEN=""    # Sin autenticación para dev

# PostgreSQL
POSTGRES_DB=rainbow_six_ml
POSTGRES_USER=ml_user
POSTGRES_PASSWORD=rainbow123
```

---

## 🚀 Workflows Automatizados

### 🔄 Scheduler de Producción (kedro-scheduler)

El scheduler ejecuta automáticamente los pipelines cada **6 horas**:

```bash
# Secuencia automática:
1. 🎮 rainbow_six           # Preparación de datos
2. 📊 modelo_clasificacion  # 5 modelos clasificación
3. 📈 modelo_regresion      # 5 modelos regresión
4. ⚖️ model_comparison      # Comparación final

# Log del scheduler:
🚀 Iniciando scheduler de Rainbow Six ML...
⏰ Ejecutando pipelines de ML programado - $(date)
✅ Pipelines de ML completados - $(date)
⏳ Esperando 6 horas para próxima ejecución...
```

### 📊 Base de Datos Automática

Inicialización automática de PostgreSQL con esquemas:

```sql
-- Esquemas creados automáticamente:
ml_models      # Métricas de modelos ML
game_data      # Datos de jugadores Rainbow Six
experiments    # Tracking de experimentos

-- Tablas principales:
player_stats          # Estadísticas de jugadores
model_metrics         # Métricas de modelos ML
experiment_runs       # Historial de experimentos
```

---

## 📊 Monitoreo y Logs

### 🔍 Comandos de Diagnóstico

```bash
# ═══════════════════════════════════════════
# ESTADO DE SERVICIOS
# ═══════════════════════════════════════════

# Ver todos los contenedores Rainbow Six
docker ps --filter "name=rainbow-six" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Estado específico con health checks
docker-compose ps

# ═══════════════════════════════════════════
# LOGS Y DEBUGGING
# ═══════════════════════════════════════════

# Logs de todos los servicios
docker-compose logs -f

# Logs específicos por servicio
docker-compose logs kedro-viz -f
docker-compose logs jupyter-lab -f
docker-compose logs kedro-scheduler -f
docker-compose logs postgres -f

# Logs con timestamps
docker-compose logs --timestamps kedro-scheduler

# Últimas 50 líneas
docker-compose logs --tail=50 kedro-viz
```

### 🏥 Health Checks Configurados

```yaml
# Kedro Services
kedro-prod:
  healthcheck:
    test: ["CMD", "kedro", "info"]
    interval: 30s
    timeout: 10s
    retries: 3

# Kedro Viz
kedro-viz:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:4141"]
    interval: 30s

# PostgreSQL
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ml_user -d rainbow_six_ml"]
    interval: 10s

# Redis
redis:
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 10s
```

---

## 🐛 Troubleshooting

### ❌ Problemas Comunes y Soluciones

#### 1. **Servicios no inician**
```bash
# Diagnóstico
docker-compose ps
docker-compose logs

# Solución: Reconstruir imágenes
docker-compose down
docker-compose build --no-cache
docker-compose --profile development up -d
```

#### 2. **Puerto ocupado**
```bash
# Error: "Port already in use"
# Solución: Cambiar puertos en docker-compose.yml
ports:
  - "4142:4141"  # Cambiar 4141 por 4142
  - "8889:8888"  # Cambiar 8888 por 8889
```

#### 3. **Problemas de permisos**
```bash
# Windows: Verificar compartir drives en Docker Desktop
# Solución: Reiniciar contenedores
docker-compose down
docker-compose --profile development up -d
```

#### 4. **Falta de memoria**
```bash
# Error: "Container killed (OOMKilled)"
# Solución: Aumentar memoria en Docker Desktop
# Configuración → Resources → Advanced → Memory: 8GB+
```

#### 5. **Build fails**
```bash
# Error durante construcción
# Solución: Limpiar sistema Docker
docker system prune -a
docker volume prune
docker-compose build --no-cache
```

### 🔧 Comandos de Recuperación

```bash
# ═══════════════════════════════════════════
# RESET COMPLETO DEL SISTEMA
# ═══════════════════════════════════════════

# 1. Parar todos los servicios
docker-compose down -v

# 2. Limpiar imágenes y volúmenes
docker system prune -a
docker volume prune

# 3. Reconstruir desde cero
docker-compose build --no-cache

# 4. Reiniciar servicios
docker-compose --profile development up -d

# ═══════════════════════════════════════════
# RESET SELECTIVO
# ═══════════════════════════════════════════

# Solo reconstruir un servicio
docker-compose build kedro-viz --no-cache
docker-compose up kedro-viz -d

# Reiniciar servicio específico
docker-compose restart jupyter-lab
```

---

## 🔐 Seguridad y Mejores Prácticas

### ✅ Medidas de Seguridad Implementadas

```bash
🔒 Usuario no-root (mluser) en todos los contenedores
🔒 Puertos específicos expuestos (no rangos amplios)
🔒 Variables de entorno centralizadas en .env
🔒 Volúmenes montados con permisos correctos
🔒 Health checks para monitoreo automático
🔒 Red Docker personalizada (aislamiento)
🔒 Credenciales de base de datos configurables
```

### 🚀 Optimizaciones de Performance

```bash
📈 Multi-stage builds para imágenes optimizadas
📈 Cache de layers Docker para builds rápidos
📈 Usuario dedicado para mejor gestión de permisos
📈 Volúmenes persistentes para datos
📈 Health checks para recuperación automática
📈 Red bridge personalizada para comunicación eficiente
```

---

## 📚 Referencias y Recursos

### 🔗 Enlaces Útiles

- **Kedro Documentation**: https://docs.kedro.org/
- **Docker Compose**: https://docs.docker.com/compose/
- **JupyterLab**: https://jupyterlab.readthedocs.io/
- **MLflow**: https://mlflow.org/docs/
- **PostgreSQL Docker**: https://hub.docker.com/_/postgres

### 📖 Archivos de Configuración

```bash
docker-compose.yml          # Configuración principal
docker-compose.override.yml # Override para desarrollo
.env                        # Variables de entorno
docker-ml.bat               # Script de gestión Windows
scripts/init-db.sql         # Inicialización de BD
```

---

## 🏆 Próximos Pasos Sugeridos

### 🎯 Desarrollo Inmediato

1. **Acceder a los servicios web**:
   ```bash
   start http://localhost:4141  # Kedro Viz
   start http://localhost:8888  # Jupyter Lab
   ```

2. **Ejecutar pipeline ML**:
   ```bash
   docker exec -it rainbow-six-kedro-viz bash
   kedro run --pipeline=modelo_clasificacion
   ```

3. **Explorar datos en Jupyter**:
   ```bash
   # Ya disponible en http://localhost:8888
   # Notebooks están en /app/notebooks/
   ```

### 🚀 Mejoras Futuras

- [ ] **CI/CD Pipeline**: GitHub Actions para deployment
- [ ] **API REST**: Servir modelos como microservicios
- [ ] **Monitoring avanzado**: Grafana + Prometheus
- [ ] **Scaling**: Docker Swarm o Kubernetes
- [ ] **Model Registry**: MLflow Model Registry completo

---

## 👥 Soporte y Contacto

Para problemas técnicos:

1. **Verificar logs**: `docker-compose logs`
2. **Consultar este README**: Sección troubleshooting
3. **GitHub Issues**: Crear issue en el repositorio
4. **Reset del sistema**: Usar comandos de recuperación

---

<div align="center">

## 🎮 **Sistema Docker Rainbow Six ML**
### ✅ **FUNCIONANDO CORRECTAMENTE**

**Kedro Viz**: http://localhost:4141 | **Jupyter Lab**: http://localhost:8888

---

**Desarrollado por**: ML Team Rainbow Six | **Fecha**: Octubre 2025 | **Versión**: 1.0

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)](docker-compose.yml)
[![Kedro](https://img.shields.io/badge/Kedro-Configured-orange?style=for-the-badge&logo=python)](Dockerfile.kedro)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-red?style=for-the-badge&logo=jupyter)](Dockerfile.jupyter)

</div>