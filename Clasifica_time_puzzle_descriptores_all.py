import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras import layers, models

val = []
time = np.load("interpolated_sequences_time_543points_with_descriptors.npy")
puzzle = np.load("interpolated_sequences_puzzle_543points_with_descriptors.npy")

time = time[:, :, 468:489, 0:2]
puzzle = puzzle[:, :, 468:489, 0:2]
# %%
time = np.transpose(time, (3, 0, 1, 2)).reshape(
    (time.shape[0], 30, time.shape[2], time.shape[3])
)
puzzle = np.transpose(puzzle, (3, 0, 1, 2)).reshape(
    (puzzle.shape[0], 30, puzzle.shape[2], puzzle.shape[3])
)

# %%
# for i in range(0,10):
#     plt.figure()
#     plt.plot(time[i,:,300,1])
#     plt.plot(time[i,:,300,3])


# %%
# for i in range(0,9):
#     plt.figure()
#     plt.imshow(time[:,10:20,0,i],cmap='jet')
#     plt.figure()
#     plt.imshow(puzzle[:,10:20,0,i],cmap='jet')

# %%
# sujeto=1
# for i in range(0,8):
#     plt.figure()
#     plt.plot(time[sujeto,:,i],'.')
#     plt.plot(puzzle[sujeto,:,i],'.')
# %% separar trainy test
a, b, c, d = time.shape
l_train = int(0.5 * a)
time_train = time[0:l_train, :, :, :]
time_test = time[l_train:-1, :, :, :]
a, b, c, d = puzzle.shape
l_train = int(0.5 * a)
puzzle_train = puzzle[0:l_train, :, :, :]
puzzle_test = puzzle[l_train:-1, :, :, :]
# %% concatenar clases
X_train = np.concatenate((time_train, puzzle_train), axis=0)
X_test = np.concatenate((time_test, puzzle_test), axis=0)
# Reemplazar valores nan
# X_train = np.nan_to_num(X_train, nan=0.5)
# X_test = np.nan_to_num(X_test, nan=0.5)

# %%
y_train = np.concatenate(
    (np.zeros(len(time_train)), np.ones(len(puzzle_train))), axis=0
)
y_test = np.concatenate((np.zeros(len(time_test)), np.ones(len(puzzle_test))), axis=0)

y_train_oh = keras.utils.to_categorical(y_train)
y_test_oh = keras.utils.to_categorical(y_test)
a, b, c, d = X_train.shape
print(a, b, c, d)

# %%
# plt.figure()
# plt.imshow(X_train[0,:,5:6,0],cmap='jet')
# plt.figure()
# plt.imshow(X_train[1,:,5:6,0],cmap='jet')


# %%
model = models.Sequential()

model.add(
    layers.Conv2D(
        256,
        (7, 1),
        activation="relu",
        padding="same",
        strides=(1, 1),
        input_shape=(b, c, d),
    )
)
# model.add(layers.MaxPooling2D((2, 2)))

model.add(
    layers.Conv2D(128, (10, 1), activation="relu", padding="valid", strides=(1, 1))
)
# model.add(layers.MaxPooling2D((2, 2)))

model.add(
    layers.Conv2D(128, (5, 1), activation="relu", padding="valid", strides=(1, 1))
)
# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(2, activation="softmax"))
model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.SGD(lr=0.1),
    metrics=["accuracy"],
)
history = model.fit(
    x=X_train,
    y=y_train_oh,
    batch_size=128,
    epochs=600,
    validation_data=(X_test, y_test_oh),
)

max_val_acc = np.max(history.history["val_acc"])

val = np.append(val, max_val_acc)
plt.figure()
plt.plot(val, "--*")
plt.xlabel("punto")
plt.ylabel("val accuracy")

plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("epochs", fontsize=15)
plt.legend(fontsize=20)
plt.figure()
plt.plot(history.history["acc"], label="train accuracy")
plt.plot(history.history["val_acc"], label="val accuracy")
plt.xlabel("epochs", fontsize=15)
plt.legend(fontsize=20)
plt.show()
