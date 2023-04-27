import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras import layers, models 

val=[]
time  = np.load('interpolated_sequences_time_75points_with_descriptors.npy')
puzzle= np.load('interpolated_sequences_puzzle_75points_with_descriptors.npy'  )

time=time[:,:,0:200,:]
puzzle=puzzle[:,:,0:200,:]

# %%
time= np.where(np.isnan(time), 0, time)
puzzle= np.where(np.isnan(puzzle), 0, puzzle)
# %%
for i in range(0,len(time)):
#     time[i,:,:,:] = np.nan_to_num(time[i,:,:,:], nan=0)
      time[i,:,:,:]= np.where(np.isnan(time), 0, time)

for i in range(0,len(puzzle)):
     puzzle[i,:,:,:]= np.where(np.isnan(puzzle), 0, puzzle)
# %% Matriz de correlación
for sec in range(0,20):
    M1=time[sec,:,:,0]
    M2=puzzle[sec,:,:,0]
    # Transponemos los datos para que los puntos sean las filas y los fotogramas sean las columnas
    M1 = np.transpose(M1, (1, 0))
    M2 = np.transpose(M2, (1, 0))
    # Unimos las dos matrices a lo largo de la dimensión 0
    matrices = np.concatenate((M1, M2), axis=1)
    # Calculamos la matriz de correlación utilizando numpy.corrcoef()
    corr_matrix = np.corrcoef(matrices)
    #corr_matrix = np.where(np.isnan(corr_matrix), 0, corr_matrix)
    plt.figure()
    plt.imshow(corr_matrix,cmap='jet')

