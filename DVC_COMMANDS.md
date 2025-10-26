DVC — Comandos útiles y guía rápida (PowerShell)

Este archivo recopila los comandos DVC y PowerShell que se usaron en este repositorio para versionar datos y trabajar con el remote local. Pégalos en PowerShell (Windows) desde la raíz del repo.

1) Ver configuración y remotos

```powershell
dvc remote list
Get-Content .dvc\config
```

2) Estado del pipeline y grafo

```powershell
dvc status
dvc dag
dvc stage list
```

3) Añadir datos al control de DVC

```powershell
# Añadir una carpeta de datos (ej. datos crudos)
dvc add proyecto-ml\data\01_raw
# Añadir carpeta completa si no hay solapamiento con stages
# (si hay solapamiento, usar dvc commit en la etapa correspondiente)
dvc add proyecto-ml\data
```

4) Registrar (commit) outputs de una etapa cuando los archivos ya existen

```powershell
# Forzar registro de outputs de la etapa data_preparation
dvc commit data_preparation -f
# Registrar outputs de otra etapa
 dvc commit model_comparison -f
```

5) Subir objetos al remote

```powershell
dvc push -j 4
```

6) Mapear hash <-> archivo (comprobación)

```powershell
# Ver hashes (md5) y rutas en dvc.lock
Get-Content dvc.lock
# Buscar los blobs en el remote local (ejemplo de ruta)
Get-ChildItem -Path 'C:\Users\droid\Desktop\dvc_data_R2' -Recurse -File | Select-Object FullName
```

7) Comandos Git relacionados (metadatos)

```powershell
# Añadir metadatos DVC y commitear
git add dvc.yaml dvc.lock *.dvc .dvc/config .gitignore
git commit -m "Register pipeline / data with DVC"
git push origin main
```

Notas importantes
- DVC guarda objetos por hash en el remote (p. ej. files/md5/ab/abcdef...). No verás los nombres originales de los CSV/PNG en la carpeta remota; la relación nombre->hash está en `dvc.lock` y en los `.dvc`/`dvc.yaml`.
- No declares al mismo tiempo un directorio como `outs` en `dvc.yaml` y, a la vez, archivos dentro del mismo directorio en otras etapas: DVC considera eso un solapamiento. Elige declarar el directorio entero o los archivos individuales.
- Flujo recomendado para outputs de pipeline:
  1) Ejecuta el pipeline (o coloca los archivos en el workspace).
  2) `dvc commit <stage>` para asociar los archivos al stage.
  3) `dvc push` para subirlos al remote.

Sugerencia: si quieres que inserte esta sección en el `README.md` (en lugar de dejar un archivo separado), dímelo y lo hago (añadiré la sección en README o crearé un enlace a este archivo).