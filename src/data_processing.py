"""Módulo para el procesamiento y almacenamiento de datos."""

import numpy as np
import pandas as pd


class DataProcessor:
    """Clase base para el procesamiento de datos."""

    def split_data(self, data, train_ratio):
        """Método para dividir los datos en conjuntos de entrenamiento y validación."""
        pass

    def save_npy_file(self, data, file_path):
        """Guarda los datos en un archivo .npy.

        Args:
            data (array): Datos a guardar.
            file_path (str): Ruta del archivo de salida.
        """
        np.save(file_path, data)


class ParquetDataProcessor(DataProcessor):
    """Clase para el procesamiento de datos en formato parquet."""

    def reshape_data(self, data):
        """Convierte el DataFrame en una matriz NumPy con la forma (n_puntos, n_frames, n_descriptores).

        Args:
            data (DataFrame): Datos en formato de DataFrame.

        Returns:
            ndarray: Datos en formato de matriz NumPy.
        """
        n_frames = data["frame"].max()
        n_points = data["landmark_index"].nunique()
        n_descriptors = len(data.columns) - 2

        reshaped_data = np.empty(
            (n_points, int(n_frames) + 1, n_descriptors), dtype=np.float16
        )

        for _, row in (
            data.groupby(["landmark_index", "frame"], as_index=False).mean().iterrows()
        ):
            point, frame = int(row["landmark_index"]), int(row["frame"])
            reshaped_data[point - 1, frame - 1] = row.iloc[2:].values.flatten()

        return reshaped_data

    def split_data(self, data, train_ratio):
        """Divide los datos en conjuntos de entrenamiento y validación.

        Args:
            data (DataFrame): Datos a dividir.
            train_ratio (float): Proporción de datos para el conjunto de entrenamiento.

        Returns:
            tuple: Datos de entrenamiento y validación.
        """
        reshaped_data = self.reshape_data(data)
        data = data.sample(frac=1).reset_index(drop=True)
        train_size = int(len(reshaped_data) * train_ratio)
        train_data = reshaped_data[:train_size]
        val_data = reshaped_data[train_size:]
        return train_data, val_data
