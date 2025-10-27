# DVC en este repo: instalación y guardado local (Windows/PowerShell)

Guía corta para instalar DVC, configurar almacenamiento LOCAL y guardar/recuperar datos y artefactos en este proyecto.

Funciona en Windows con PowerShell (>=5.1). Los comandos están escritos para ejecutarse desde la raíz del repo, salvo que se indique lo contrario.

## 1) Requisitos e instalación

- Git y Python 3.9+ instalados.
- Recomendado: entorno virtual (venv).

```powershell
# Crear y activar venv (opcional pero recomendado)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias del proyecto (incluye DVC)
pip install -r requirements.txt

# Verificar instalación de DVC
dvc --version
```

## 2) Estructura DVC de este repo

- Archivo de pipeline: `proyecto-ml/dvc.yml` (ejecuta etapas Kedro).
- Cache local de DVC: por defecto redirigida a `data/dvc_storage`.
- Remoto local de DVC: definido en `.dvc/config` como `localstore`.

Puedes revisar la configuración con:

```powershell
dvc remote list
Get-Content .dvc\config
```

Ejemplo típico de `.dvc/config` en este repo:

```
[core]
    remote = localstore
[cache]
    dir = ../data/dvc_storage
['remote "localstore"']
    url = C:\Users\TU_USUARIO\Desktop\dvc_data
```

- `cache.dir` establece dónde DVC guarda su caché local (aquí: `data/dvc_storage`).
- `remote url` es la carpeta LOCAL donde se subirán los objetos (tu “almacén” compartible).

## 3) Configurar o cambiar el remoto LOCAL

Si quieres usar otra carpeta local como remoto (por ejemplo `D:\dvc_remote`), puedes modificarlo así:

```powershell
# Ver el remoto actual
dvc remote list

# Cambiar la carpeta del remoto local existente (localstore)
dvc remote modify localstore url D:\dvc_remote

# Confirmar
Get-Content .dvc\config
```

Si no existiera ningún remoto, créalo y márcalo por defecto:

```powershell
dvc remote add -d localstore D:\dvc_remote
```

## 4) Guardar datos “manuales” (fuera del pipeline)

Caso típico: versionar una carpeta de datos crudos.

```powershell
# Añadir al control de DVC (crea un .dvc y mete la ruta en .gitignore)
dvc add proyecto-ml\data\01_raw

# Registrar los metadatos en Git (no se suben los binarios a Git)
git add proyecto-ml\data\01_raw.dvc .gitignore
git commit -m "Track 01_raw con DVC"

# Subir los objetos al remoto LOCAL
dvc push
```

Cuando cambien los archivos de `01_raw`, repite:

```powershell
dvc add proyecto-ml\data\01_raw
git add proyecto-ml\data\01_raw.dvc
git commit -m "Update datos 01_raw"
dvc push
```

Alternativa si esa ruta es output de una etapa y ya existe el stage: usa `dvc commit` en lugar de `dvc add`.

## 5) Guardar outputs de pipeline (con Kedro)

El pipeline está descrito en `proyecto-ml/dvc.yml`. Para reproducir y guardar:

```powershell
# Desde la raíz del repo o entrando en la carpeta del pipeline
Set-Location .\proyecto-ml

# Reproducir todo el pipeline (o una etapa)
dvc repro
# dvc repro model_comparison  # ejemplo de etapa específica

# Registrar el lock file en Git y subir los objetos a remoto
git add .\dvc.lock
git commit -m "Actualiza resultados del pipeline"
dvc push

# (Opcional) volver a la raíz
Set-Location ..
```

Tips:
- Si ejecutas desde la raíz, puedes indicar el archivo con `-f`:
  ```powershell
  dvc repro -f .\proyecto-ml\dvc.yml
  ```
- Los resultados y modelos se escriben bajo `proyecto-ml\data\06_models` y `proyecto-ml\data\08_reporting` según las etapas.

## 6) Recuperar datos en otra máquina o después de un clone

```powershell
# Descargar objetos desde el remoto LOCAL a tu workspace
dvc pull
```

Si el remoto apunta a una ruta que no existe en tu máquina, modifícalo como en la sección 3.

## 7) Buenas prácticas y errores comunes

- No ignores con `.dvcignore` rutas que sean `deps`, `outs` o `metrics` del pipeline. Si DVC dice “Path ... is ignored by .dvcignore”, elimina o ajusta ese patrón.
- Evita solapamientos de salidas: no declares a la vez un directorio como `outs` y ficheros dentro de ese directorio como `outs/metrics` de otra etapa. Declara el directorio entero O los archivos individuales, pero no ambos.
- Si ves “'dvc.yaml' does not exist”, ejecuta desde `proyecto-ml` o usa `-f .\proyecto-ml\dvc.yml`.
- Git sólo guarda metadatos (`.dvc`, `dvc.lock`, `dvc.yml`), los binarios van al remoto de DVC; recuerda correr `dvc push` tras hacer commit.
- En Windows usa rutas con `\` y prefijo `.\u200b` cuando sea relativo.

## 8) Chequeos útiles

```powershell
# Estado general del pipeline y grafo
dvc status
dvc dag

# Listar etapas
dvc stage list

# Ver configuración actual
dvc remote list
Get-Content .dvc\config

# Inspeccionar hashes/artefactos
Get-Content proyecto-ml\dvc.lock
```

## 9) Flujo recomendado resumido

- Datos manuales:
  1) `dvc add <ruta>`
  2) `git add` + `git commit`
  3) `dvc push`

- Pipeline Kedro:
  1) `cd proyecto-ml` y `dvc repro`
  2) `git add proyecto-ml\dvc.lock` + `git commit`
  3) `dvc push`

---

