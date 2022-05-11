# Instrucciones despliegue de back
* Instalar con pip virtual env
    ```bash
    pip install venv
    ```
* Crear un ambiente virtual
    ```bash
    python3 -m virtualenv venv
    ```
* Activar el ambiente virtual
   - Para windows 
        ```bash
        . ./venv/Scripts/activate 
        ```
    - Para Linux
        ```bash
        . ./venv/bin/activate
        ```
* Instalar todos los paquetes del requirements.txt
    ```bash
    pip install -r requirements.txt
    ```
* Activar el servidor de flask
     ```bash
    python3 ./src/app.py
    ```
* Para desactivar el ambiente virtual
    - Para windows 
        ```bash
        . ./venv/Scripts/deactivate 
        ```
    - Para Linux
        ```bash
        . ./venv/bin/deactivate
        ```