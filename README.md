# Introducion
- 

# Estructura del proyecto
-

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

   


