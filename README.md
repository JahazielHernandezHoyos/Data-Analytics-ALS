# Proyecto de Procesamiento de Lenguaje de Señas

Este proyecto tiene como objetivo procesar y limpiar un conjunto de datos de lenguaje de señas. El conjunto de datos en bruto se encuentra en formato Parquet y contiene información sobre gestos de lenguaje de señas en 3D.

## Configuración del entorno virtual

1. Asegúrate de tener instalado Python 3.7 o superior en tu sistema.
2. Instala `virtualenv` para crear entornos virtuales de Python:

```bash
pip install virtualenv
```
3. Crea un entorno virtual en el directorio del proyecto:
4. Activa el entorno virtual:

- En Windows:

  ```
  venv\Scripts\activate
  ```

- En macOS y Linux:

  ```
  source venv/bin/activate
  ```
5. Instala las dependencias del proyecto:

  ```bash
  pip install -r requirements.txt
  ```

## Estructura del proyecto

.
├── main.py
├── config.py
├── src
│ ├── data_cleaning.py
│ ├── data_processing.py
│ └── utils.py
├── train_landmark_files
│ └── """landmark files.parquet"""
└── data
├── train.csv
└── npy_data

### Descripción de los archivos

1. `main.py`: Archivo principal que ejecuta todo el flujo de trabajo del proyecto. Se encarga de leer el archivo CSV con información sobre los archivos de datos en bruto, filtrarlos si es necesario, limpiar los datos, dividirlos en conjuntos de entrenamiento y validación y guardarlos en formato `.npy`.

2. `config.py`: Contiene las configuraciones y constantes del proyecto, como las rutas de los directorios y la proporción de división entre los conjuntos de entrenamiento y validación.

3. `src/data_cleaning.py`: Contiene la clase `ParquetDataCleaner`, que se encarga de limpiar los datos en bruto. Realiza tareas como eliminar las filas con valores faltantes y corregir errores en las coordenadas.

4. `src/data_processing.py`: Contiene la clase `ParquetDataProcessor`, que procesa los datos limpios para prepararlos para el entrenamiento y la validación. Divide los datos en conjuntos de entrenamiento y validación, reformatea los datos en una forma adecuada y guarda los datos en archivos `.npy`.

5. `src/utils.py`: Contiene varias funciones de utilidad que se utilizan en el proyecto, como leer y escribir archivos CSV, filtrar un archivo CSV por palabras clave específicas y otros.

## Cómo usar el proyecto

1. Asegúrate de tener el entorno virtual activado (ver sección de configuración del entorno virtual).

2. Ejecuta el script principal `main.py`:

  ```bash
  python main.py
  ```

Esto leerá el archivo `train.csv`, procesará los datos en bruto, los dividirá en conjuntos de entrenamiento y validación, y los guardará en archivos `.npy` en el directorio `data/npy_data`.

3. Los datos procesados estarán disponibles en el directorio `data/npy_data` para su uso posterior en entrenamiento y validación de modelos de aprendizaje automático.

Si deseas cambiar la configuración del proyecto, como las rutas de los directorios o la proporción de división entre los conjuntos de entrenamiento y validación, modifica las constantes en config.py.

Si necesitas realizar cambios en el proceso de limpieza o procesamiento de datos, modifica las clases ParquetDataCleaner y ParquetDataProcessor en los archivos src/data_cleaning.py y src/data_processing.py, respectivamente.