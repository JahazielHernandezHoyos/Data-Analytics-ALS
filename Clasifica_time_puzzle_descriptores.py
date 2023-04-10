import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras import layers, models

val=[]
puzzle_train = np.load('asl-signs/puzzle_train.npy')
time_train   = np.load('asl-signs/time_train.npy'  )
puzzle_test  = np.load('asl-signs/puzzle_test.npy' )
time_test    = np.load('asl-signs/time_test.npy'   )

# %%

punto=300
plt.figure()
plt.plot(puzzle_train[0,:,0])
# %%

def extract_descriptors(data_train):
    data_train_descriptor=np.zeros((543,8))
    for i in range(0,543):
        d1=np.mean(data_train[i,:,0])
        d2=np.mean(data_train[i,:1])
        d3=np.max(data_train[i,:,0])
        d4=np.max(data_train[i,:,1])
        d5=np.min(data_train[i,:,0])
        d6=np.min(data_train[i,:,1])
        d7 = np.sum(np.square(data_train[i,:,0]))
        d8 = np.sum(np.square(data_train[i,:,1]))
        data_train_descriptor[i,:]=[d1,d2,d3,d4,d5,d6,d7,d8]
    return data_train_descriptor

puzzle_train_with_descriptors = extract_descriptors(puzzle_train)
time_train_with_descriptors = extract_descriptors(time_train)
puzzle_test_with_descriptors = extract_descriptors(puzzle_test)
time_test_with_descriptors = extract_descriptors(time_test)


# %%
plt.figure()
plt.imshow(puzzle_train_with_descriptors[0:20,:])


# %%


X_train_0=np.expand_dims(puzzle_train_with_descriptors,axis=0)
X_train_1=np.expand_dims(time_train_with_descriptors,  axis=0)
X_test_0 =np.expand_dims(puzzle_test_with_descriptors, axis=0)
X_test_1 =np.expand_dims(time_test_with_descriptors,   axis=0)

X_train_0=np.expand_dims(X_train_0,axis=3)
X_train_1=np.expand_dims(X_train_1,  axis=3)
X_test_0 =np.expand_dims(X_test_0, axis=3)
X_test_1 =np.expand_dims(X_test_1,   axis=3)

# %%

X_train = np.concatenate((X_train_0, X_train_1),  axis=0)
X_test  = np.concatenate((X_test_0  , X_test_1 ), axis=0)

X_train = np.nan_to_num(X_train, nan=0)
X_test = np.nan_to_num(X_test, nan=0)

# plt.figure()
# plt.imshow(X_train_0[0,:,:,0])
# plt.figure()
# plt.imshow(X_train_1[0,:,:,0])
y_train=np.concatenate((np.zeros(len(X_train_0)),np.ones(len(X_train_1)) ), axis=0)
y_test=np.concatenate((np.zeros(len(X_test_0)),np.ones(len(X_test_1)) ), axis=0)

y_train_oh = keras.utils.to_categorical(y_train)
y_test_oh = keras.utils.to_categorical(y_test)
a,b,c,d=X_train.shape
print(a,b,c,d)

# %%
n_channel=1
model = models.Sequential()

model.add(layers.Conv2D(256, (2, 2), activation='sigmoid', padding="same", strides=(1, 1),input_shape=(b,c,d)))
#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (1, 2), activation='sigmoid',padding="same",strides=(1, 1)))
#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (1, 2), activation='sigmoid',padding="valid",strides=(1, 1)))
#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(keras.layers.Dense(128, activation='sigmoid'))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.001), metrics=['accuracy'])
history = model.fit(x=X_train, y=y_train_oh, batch_size=64, epochs=1000, validation_data=(X_test, y_test_oh))

max_val_acc = np.max(history.history['val_acc'])

val=np.append(val,max_val_acc)
plt.figure()
plt.plot(val,'--*')
plt.xlabel('punto')
plt.ylabel('val accuracy')

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epochs', fontsize=15)
plt.legend(fontsize=20)
plt.figure()
plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='val accuracy')
plt.xlabel('epochs', fontsize=15)
plt.legend(fontsize=20)
plt.show()