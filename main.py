"""Script principal para ejecutar el flujo de trabajo."""

from src.data_cleaning import ParquetDataCleaner
from src.data_processing import ParquetDataProcessor
from src import utils
import config
import os
import numpy as np


def main():
    """Ejecuta el flujo de trabajo principal."""
    data_info = utils.read_csv(config.CSV_FILE)

    # Filtrar el archivo train.csv por la palabra clave 'bird'
    filtered_csv_path = config.DATA_PATH + "filtered_train.csv"
    utils.filter_csv_by_sign(config.CSV_FILE, filtered_csv_path, "time")

    # Leer el CSV filtrado esto se comenta si se quiere procesar todas las palabras
    data_info = utils.read_csv(filtered_csv_path)

    # Crear las carpetas necesarias si no existen
    os.makedirs(config.CLEANED_DATA_PATH, exist_ok=True)
    os.makedirs(config.NPY_DATA_PATH, exist_ok=True)

    data_cleaner = ParquetDataCleaner()
    data_processor = ParquetDataProcessor()

    train_subjects_data = {}
    val_subjects_data = {}
    min_and_max = {}
    for index, row in data_info.iterrows():
        parquet_path = config.RAW_DATA_PATH + row["path"]
        cleaned_data = data_cleaner.clean(parquet_path)

        # Agregar esta línea para imprimir el número máximo de frames en cleaned_data
        print(
            "Max frames in cleaned_data:",
            cleaned_data["frame"].max(),
            "for participant_id:",
            row["participant_id"],
            "and sequence_id:",
            row["sequence_id"],
        )

        # Visualizar como esta compuesta cleaned_data
        print(
              "cleaned_data:",
              cleaned_data,
              cleaned_data.shape,
              cleaned_data.columns,
              cleaned_data["frame"].max(),
              cleaned_data["frame"].min(),
              cleaned_data["frame"].unique(),
              cleaned_data["frame"].value_counts())
        # Dividir y guardar los datos en archivos .npy
        train_data, val_data = data_processor.split_data(
            cleaned_data, config.TRAIN_RATIO
        )
        data_processor.save_npy_file(
            train_data,
            f"{config.NPY_DATA_PATH}{row['participant_id']}_{row['sequence_id']}_train.npy",
        )
        data_processor.save_npy_file(
            val_data,
            f"{config.NPY_DATA_PATH}{row['participant_id']}_{row['sequence_id']}_val.npy",
        )

        # Almacenar la información en los diccionarios
        participant_id = row["participant_id"]
        if participant_id not in train_subjects_data:
            train_subjects_data[participant_id] = {"n_points": 0, "n_frames": set()}

        if participant_id not in val_subjects_data:
            val_subjects_data[participant_id] = {"n_points": 0, "n_frames": set()}

        train_subjects_data[participant_id]["n_points"] += len(train_data)
        train_subjects_data[participant_id]["n_frames"].update(
            np.unique(train_data[:, 1])
        )

        val_subjects_data[participant_id]["n_points"] += len(val_data)
        val_subjects_data[participant_id]["n_frames"].update(np.unique(val_data[:, 1]))

    utils.save_dict_to_csv(train_subjects_data, config.TRAIN_SUBJECTS_DATA_PATH)
    utils.save_dict_to_csv(val_subjects_data, config.VAL_SUBJECTS_DATA_PATH)


if __name__ == "__main__":
    main()
