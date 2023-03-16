"""Codigo para visualizar un archvo .parquet al azar"""
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# Ruta de los archivos .parquet
file_path = "/train_landmark_files/2044"

# Lista de archivos .parquet
files = os.listdir(file_path)

# Seleccionar un archivo .parquet al azar
random_file = random.choice(files)

# Ruta del archivo .parquet seleccionado
random_file_path = os.path.join(file_path, random_file)

# Leer el archivo .parquet
data = pd.read_parquet(random_file_path)

# Visualizar los datos
plt.scatter(data["x"], data["y"])
plt.show()

print(
    f"El archivo {random_file} tiene {len(data)} puntos y se encuentra en la carpeta {file_path}",
    "Ultimos 30 datos" + str(data.tail(30)),
    "Primeros 30 datos" + str(data.head(30)),
    "Datos aleatorios" + str(data.sample(30)),
)