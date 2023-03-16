"""
Visualizar un archivo .npy con matplotlib
"""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import numpy as np
import os


PATH = "data/npy_data"

# select a random file
file = np.random.choice(os.listdir(PATH))
PATH = os.path.join(PATH, file)

data = np.load(PATH)

# normalizar los datos
data = data / data.max()

# Visualizar los datos
plt.plot(data[:, 0], data[:, 1], "o")
plt.show()

# Visualizar los datos con plotly
# fig = go.Figure(data=go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers"))
# fig.show()

print(data[:50])
print("Forma del archivo .npy:", data.shape)
print("Tipo de datos:", data.dtype)
print("Primeros 50 datos:\n", data[:50])
print("Últimos 50 datos:\n", data[-50:])
print("Datos aleatorios:\n", data[np.random.randint(0, data.shape[0], 50)])

