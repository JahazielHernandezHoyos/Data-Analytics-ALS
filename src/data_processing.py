"""Módulo para el procesamiento y almacenamiento de datos."""

import numpy as np
import pandas as pd


class DataProcessor:
    """Clase base para el procesamiento de datos."""

    def save_npy_file(self, data, file_path):
        """Guarda los datos en un archivo .npy.

        Args:
            data (array): Datos a guardar.
            file_path (str): Ruta del archivo de salida.
        """
        np.save(file_path, data)


    def reshape_data(self, data):
        """Convierte el DataFrame en una matriz NumPy con la forma (n_frames, n_descriptores).

        Args:
            data (DataFrame): Datos en formato de DataFrame.

        Returns:
            ndarray: Datos en formato de matriz NumPy.
        """
        n_frames = data["frame"].max()

        # Cambiamos la forma del resultado a (n_frames, n_descriptores)
        data_descriptor = np.zeros((int(n_frames), 8))

        for frame in range(1, int(n_frames) + 1):
            frame_data = data[data["frame"] == frame]
            d1 = np.mean(frame_data["x"])
            d2 = np.mean(frame_data["y"])
            d3 = np.max(frame_data["x"])
            d4 = np.max(frame_data["y"])
            d5 = np.min(frame_data["x"])
            d6 = np.min(frame_data["y"])
            d7 = np.sum(np.square(frame_data["x"]))
            d8 = np.sum(np.square(frame_data["y"]))
            data_descriptor[frame - 1, :] = [d1, d2, d3, d4, d5, d6, d7, d8]

        return data_descriptor

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
