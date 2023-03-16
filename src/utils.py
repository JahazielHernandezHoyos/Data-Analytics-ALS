"""Módulo de utilidades."""

import pandas as pd


def read_csv(csv_path):
    """Lee un archivo CSV y devuelve un DataFrame.

    Args:
        csv_path (str): Ruta del archivo CSV.

    Returns:
        DataFrame: Datos leídos del archivo CSV.
    """
    df = pd.read_csv(csv_path)
    return df


def filter_csv_by_sign(csv_file: str, output_file: str, sign: str) -> None:
    """
    Filtra un archivo CSV por la columna 'sign' y guarda el resultado en un nuevo archivo CSV.

    Args:
        csv_file: La ruta al archivo CSV original.
        output_file: La ruta al archivo CSV de salida.
        sign: La palabra clave para filtrar en la columna 'sign'.
    """
    data = pd.read_csv(csv_file)
    filtered_data = data[data["sign"] == sign]
    filtered_data.to_csv(output_file, index=False)

def save_dict_to_csv(dictionary, output_file):
    """Guarda un diccionario en un archivo CSV.

    Args:
        dictionary (dict): Diccionario a guardar.
        output_file (str): Ruta del archivo CSV de salida.
    """
    df = pd.DataFrame.from_dict(dictionary, orient="index")
    df.to_csv(output_file)
