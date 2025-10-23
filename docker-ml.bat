@echo off
REM ==========================================
REM Rainbow Six ML - Docker Management Script
REM ==========================================

echo.
echo 🎮 Rainbow Six ML - Docker Management
echo ====================================

if "%1"=="help" goto help
if "%1"=="dev" goto dev
if "%1"=="prod" goto prod
if "%1"=="jupyter" goto jupyter
if "%1"=="viz" goto viz
if "%1"=="logs" goto logs
if "%1"=="stop" goto stop
if "%1"=="clean" goto clean
if "%1"=="build" goto build
if "%1"=="status" goto status

:help
echo.
echo Comandos disponibles:
echo.
echo   dev       - Iniciar entorno de desarrollo (Jupyter + Kedro Viz + Redis)
echo   prod      - Iniciar entorno de producción completo
echo   jupyter   - Solo Jupyter Lab (puerto 8888)
echo   viz       - Solo Kedro Viz (puerto 4141)
echo   logs      - Ver logs de todos los servicios
echo   status    - Ver estado de los contenedores
echo   stop      - Parar todos los servicios
echo   clean     - Limpiar contenedores e imágenes
echo   build     - Reconstruir imágenes
echo.
echo Ejemplos:
echo   docker-ml.bat dev
echo   docker-ml.bat prod
echo   docker-ml.bat logs
echo.
goto end

:dev
echo 🚀 Iniciando entorno de DESARROLLO...
echo.
docker-compose --profile development up -d --build
echo.
echo ✅ Servicios iniciados:
echo    - Jupyter Lab: http://localhost:8888
echo    - Kedro Viz: http://localhost:4141
echo    - Redis: localhost:6379
echo.
echo Para ver logs: docker-ml.bat logs
goto end

:prod
echo 🏭 Iniciando entorno de PRODUCCIÓN...
echo.
docker-compose --profile production up -d --build
echo.
echo ✅ Servicios de producción iniciados
echo    - Kedro Viz: http://localhost:4141
echo    - PostgreSQL: localhost:5433
echo    - Scheduler ejecutándose en background
echo.
goto end

:jupyter
echo 📓 Iniciando solo Jupyter Lab...
echo.
docker-compose up jupyter-lab -d --build
echo.
echo ✅ Jupyter Lab disponible en: http://localhost:8888
goto end

:viz
echo 📊 Iniciando solo Kedro Viz...
echo.
docker-compose up kedro-viz -d --build
echo.
echo ✅ Kedro Viz disponible en: http://localhost:4141
goto end

:logs
echo 📋 Mostrando logs de servicios...
echo.
docker-compose logs -f
goto end

:status
echo 📈 Estado de contenedores:
echo.
docker ps --filter "name=rainbow-six" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.
goto end

:stop
echo 🛑 Parando todos los servicios...
echo.
docker-compose down
echo.
echo ✅ Todos los servicios parados
goto end

:clean
echo 🧹 Limpiando contenedores e imágenes...
echo.
docker-compose down -v --rmi all
docker system prune -f
echo.
echo ✅ Limpieza completada
goto end

:build
echo 🔨 Reconstruyendo imágenes...
echo.
docker-compose build --no-cache
echo.
echo ✅ Imágenes reconstruidas
goto end

:end
echo.
echo Para más ayuda: docker-ml.bat help