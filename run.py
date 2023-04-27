# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from fastdtw import fastdtw
import pywt
from tqdm.auto import tqdm
import multiprocessing as mp

# %%
"""Config"""
discart_points_face = True
landmarks = 75 # 75 sin puntos de la cara y con puntos de la cara son 543
data = pd.read_csv("train.csv") # Cargar el csv
# word = "time"
words = data["sign"].unique()
number_words = 33 # Número de palabras a generar
# filtered_data = data[data["sign"] == word] # Filtrar por palabra
csv_path = "filter_for_word.csv" # Guardar el csv filtrado
# filtered_data.to_csv(csv_path, index=False) # Guardar el csv filtrado
num_frames = 60 # Número de fotogramas para las secuencias interpoladas

# %%
#leer un .parquet aleatorio
# path_one_file = filtered_data.iloc[6]["path"]
# df = pd.read_parquet(path_one_file)

# %%
def process_path_data(path_number, filtered_data):
    """
    Procesa los datos de una ruta específica (los .parquet), extrae y reformatea la información relevante de los puntos (x, y).

    Args:
        path_number (int): El índice de la ruta en el DataFrame filtered_data.
        filtered_data (pandas.DataFrame): Un DataFrame que contiene información sobre las rutas de los archivos de datos.

    Returns:
        numpy.ndarray: Un array de la secuencia reformateada con la forma (num_samples, 543, 2), donde num_samples es el
                       número de fotogramas únicos en los datos.
    """
    path_file = filtered_data.iloc[path_number]["path"]
    data = pd.read_parquet(path_file, columns=["frame", "x", "y", "type"])
    if discart_points_face:
        data = data[data["type"] != "face"]
    # Reemplazar NaN con 0
    cleaned_data = np.nan_to_num(data[["x", "y"]].to_numpy())
    num_samples = data["frame"].nunique()
    seq = cleaned_data.reshape(num_samples, landmarks, 2)
    return seq

# %%
def interpolate_sequence(seq, num_frames):
    """
    Interpola una secuencia única a una longitud específica en la dimensión del tiempo.

    Args:
        seq (numpy.ndarray): La secuencia de entrada con la forma (T, landmarks, 2), donde T es el número de fotogramas.
        num_frames (int): El número de fotogramas para la secuencia interpolada.

    Returns:
        numpy.ndarray: La secuencia interpolada con la forma (num_frames, landmarks, 2).
    """
    seq_interp = np.zeros((num_frames, landmarks, 2))
    seq = np.nan_to_num(seq)

    def interp_func(x):
        f = interp1d(np.arange(seq.shape[0]), x, kind='linear', bounds_error=False)
        return f(np.linspace(0, seq.shape[0] - 1, num_frames))

    seq_interp[:,:,0] = np.apply_along_axis(interp_func, 0, seq[:,:,0])
    seq_interp[:,:,1] = np.apply_along_axis(interp_func, 0, seq[:,:,1])

    return seq_interp

# %%
def interpolate_sequences_list(sequences, num_frames):
    """
    Interpola una lista de secuencias a una longitud específica en la dimensión del tiempo.

    Args:
        sequences (list): Una lista de secuencias, cada una con la forma (T, 543, 2), donde T es el número de fotogramas.
        num_frames (int): El número de fotogramas para las secuencias interpoladas.

    Returns:
        numpy.ndarray: Un array que contiene las secuencias interpoladas con la forma (num_frames, 543, 2, num_sequences),
                       donde num_sequences es el número de secuencias en la lista de entrada.
    """
    num_sequences = len(sequences)
    interpolated_sequences = np.zeros((num_frames, landmarks, 2, num_sequences))

    for idx, seq in enumerate(sequences):
        seq_interp = interpolate_sequence(seq, num_frames)
        interpolated_sequences[:, :, :, idx] = seq_interp

    return interpolated_sequences


# %%

def add_descriptors(sequence):
    """
    Args: for example: (60, 75, 2) where 60 is the number of frames, 75 is the number of landmarks and 2 is the x and y coordinates
    Returns: (60, 75, 1) 1 is for the resume X and Y coordinates in one value mean of the wavelet coefficients
    """
    descriptors = np.mean(pywt.dwt(sequence, 'db1', axis=-1)[0], axis=-1, keepdims=True)
    return descriptors


# %%
def export_file_train(filtered_data, num_frames, word_name):
    """unic word train data select in the config"""
    sequences = [] # Lista de secuencias
    for i in range(filtered_data.shape[0]):
        seq = process_path_data(i, filtered_data)
        sequences.append(seq)

    interpolated_sequences = interpolate_sequences_list(sequences, num_frames)
    interpolated_sequences = interpolated_sequences.transpose((3,0,1,2)) #reorganizacion

    # Añade los descriptores de derivada para cada secuencia interpolada
    interpolated_sequences_with_descriptors = []
    for seq in interpolated_sequences:
        seq_with_descriptors = add_descriptors(seq)
        interpolated_sequences_with_descriptors.append(seq_with_descriptors)

    # Convierte la lista de secuencias interpoladas con descriptores en un array de NumPy
    interpolated_sequences_with_descriptors = np.array(interpolated_sequences_with_descriptors)

    # print(interpolated_sequences_with_descriptors.shape) # (num_sequences, num_frames, landmarks, n-descriptors)

    # """ export interpolated_sequences to npy"""
    if discart_points_face is True:
        np.save(f"interpolated_sequences_{word_name}_75points_with_descriptors.npy", interpolated_sequences_with_descriptors)
        plt.plot(interpolated_sequences_with_descriptors[1, :, 0], interpolated_sequences_with_descriptors[1, :, 1], 'o')
    else:
        np.save(f"interpolated_sequences_{word_name}_543points_with_descriptors.npy", interpolated_sequences_with_descriptors)
        plt.plot(interpolated_sequences_with_descriptors[1, :, 0], interpolated_sequences_with_descriptors[1, :, 1], 'o')

# %%
"""main"""
# se seleccionan de forma aleatoria y de un tamaño de "number_words" las palabras que se van a utilizar de forma tal de que no se repitan
# words_select = np.random.choice(words, size=number_words, replace=False)
# for word in words_select:
#     filtered_data = data[data["sign"] == word] # Filtrar por palabra
#     print("palabra", word, "tiene", filtered_data.shape[0], "muestras")
#     export_file_train(filtered_data, num_frames)

def process_word(word):
    filtered_data = data[data["sign"] == word]
    # print("palabra", word, "tiene", filtered_data.shape[0], "muestras")
    export_file_train(filtered_data, num_frames, word)

if __name__ == "__main__":
    words_select = np.random.choice(words, size=number_words, replace=False)
    num_cores = 6 # número de núcleos a utilizar
    pool = mp.Pool(num_cores)
    with tqdm(total=len(words_select)) as pbar:
        for _ in pool.imap_unordered(process_word, words_select):
            pbar.update()
    pool.close()
    pool.join()