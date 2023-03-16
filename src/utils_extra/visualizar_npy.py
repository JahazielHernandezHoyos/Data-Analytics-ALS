"""
Visualizar un archivo .npy con matplotlib
"""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import numpy as np

PATH = "data/npy_data/2044_131370200_train.npy"

data = np.load(PATH)

# Visualizar los datos
plt.plot(data[:, 0], data[:, 1], "o")
plt.show()

# Visualizar los datos con plotly
fig = go.Figure(data=go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers"))
fig.show()

# ver los primeros 50 datos
print(data[:50])
print("Forma del archivo .npy:", data.shape)