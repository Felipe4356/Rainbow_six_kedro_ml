# Introducción
Para el desarrollo de este proyecto se escogió el dataset Rainbow Six Siege.

El dataset contiene información detallada de partidas clasificadas del videojuego. La razón principal de la elección del data set fue contar con un conjunto de datos rico en variables tanto numéricas como categóricas, que permitiera aplicar un proceso completo de análisis de datos y preparación para futuros modelos predictivos.

# Estructura del proyecto
Para la realización de este proyecto se desarrollo a través de de 3 fases:

Fase 1: Business Understanding
En esta fase definimos el objetivo principal del proyecto:
Analizar y preparar los datos de Rainbow Six Siege para detectar patrones en el juego.
Identificar comportamientos atípicos y variables clave que impactan el principalmente en el rendimiento.
Establecer un pipeline reproducible con Kedro, que permita la automatización del proceso de análisis.

Fase 2: Data Understanding
En esta fase se realizamos una exploración inicial de los datos:
Revisión de la estructura de los datasets.
Identificación de valores nulos y duplicados.
Clasificación de variables en numéricas y categóricas.
Histogramas de distribución.
Boxplots para detectar outliers.
Mapas de calor para evaluar correlaciones entre variables.
Análisis específico del uso de armas primarias y secundarias.

Fase 3: Data Preparation
En esta fase se realizamos las transformaciones necesarias para preparar los datos antes del modelado:
Combinación de datasets en un único dataset.
Limpieza de datos, eliminando duplicados y valores faltantes.
Tratamiento de outliers, generando un dataset depurado.
Generación de variables derivadas, como el cálculo del K/D ratio (Kills/Deaths).
Normalización y estructuración de los datos para obtener un dataset final listo para futuras fases de modelado.

### Visualizacion de procesamiento de los datos en pipeline con kedro viz

<img src="images/kedro-pipeline.png" alt="kedro viz" width="300" style="border: 2px solid #ddd; border-radius: 6px;"/>

# Intrucciones del proyecto
- Pasos de instalaccion para ejecutar el proyecto:

1. Creacion del entorno virtual
   
    ```bash
    python -m venv venv
    ```
2. Activacion del entorno virtual
   
    en windows
     ```bash
     venv\Scripts\activate
     ```
  
     en macOS/Linux:
  
     ```bash
     source venv/bin/activate
     ```
4. Intalacion de librerias necesarias:
- Al tener etorno ya instalado y activado ejecute  `requirements.txt`
    ```bash
    pip install -r requirements.txt
    ```

3. Verificar que este instalado kedro
   verifique que el kedro este instalado para poder ejecutar
    ```bash
       kedro info
    ```

# Como trabajar en kedro usando Notebook Jupyter
- Sobre utilizar kedro y trabajar con Notebook de Jupyterrevise [la documentacion de kedro](https://docs.kedro.org/en/1.0.0/tutorials/notebooks_tutorial).

  
1. Instalar Jupyter antes utizar
    ```bash
    pip install jupyter
    ```
2. Entrar al entorno del proyecto
    ```bash
    cd proyecto-ml
    ```

3. Despues  instalar  ejecute para poder trabajar en los notebook de forma local en navegador
    ```bash
    kedro jupyter notebook
    ```


# Referencia: 
https://www.kaggle.com/datasets/maxcobra/rainbow-six-siege-s5-ranked-dataset


