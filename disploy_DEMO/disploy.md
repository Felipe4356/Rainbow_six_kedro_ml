# Como instalar el disploy_prediciones

Por favor, descargue el archivo `disploy_r2.zip` desde la carpeta `disploy_demo` ubicada en el repositorio del proyecto. Una vez descargado, descomprima el archivo y siga los pasos que se detallan a continuación para instalar el módulo Disploy en su entorno de Visual Studio Code.


1. Crea tu entorno virtual
```bash
 python -m venv venv 
 ```

2. Activa el entorno virtual

```bash  
venv\Scripts\activate 
```

3. Realizar un pip install

```bash
pip install -r requirements.txt 
```

4. Ingresar a la carpeta del  poryecto 

```bash 
cd proyecto_R2 
```

5. Entrenar modelo

```bash
 python.exe entrenar_modelo_.py
 ```

6. Ejecuta el simulador de prediciones

```bash 
uvicorn main:app --reload 
```

7. Selecciona la tecla ctr+click a esta direcion para abrir el interfaz del diploy en el navegador

```bash
 http://127.0.0.1:8000
 ```