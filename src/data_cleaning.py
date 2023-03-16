"""Módulo para la limpieza y preprocesamiento de datos."""

import pandas as pd


class DataCleaner:
    """Clase base para la limpieza de datos."""

    def clean(self, data):
        """Método para limpiar los datos."""
        pass


class ParquetDataCleaner(DataCleaner):
    """Clase para la limpieza de datos en formato parquet."""

    def clean(self, file_path):
        """Limpia y preprocesa los datos en un archivo parquet.

        Args:
            file_path (str): Ruta del archivo parquet.

        Returns:
            DataFrame: Datos limpios y preprocesados.
        """
        data = pd.read_parquet(file_path)
        cleaned_data = data[['frame', 'landmark_index', 'x', 'y', 'z']]
        return cleaned_data
