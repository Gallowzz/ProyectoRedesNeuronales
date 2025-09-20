import numpy as np
import DnnLib

#Crear datos de entrada
x = np.array([[0.5, -0.2, 0.1]])

# Crear capa densa: 3 entradas, 2 salidas, con activación ReLU
layer = DnnLib.DenseLayer(3, 2, DnnLib.ActivationType.RELU)

# Modificar manualmente los pesos y bias
layer.weights = np.array([[0.1, 0.2, 0.3],
[0.4, 0.5, 0.6]])
layer.bias = np.array([0.01, -0.02])

# Forward con activación
y = layer.forward(x)
print("Salida con activación:", y)

# Forward lineal (sin activación)
y_lin = layer.forward_linear(x)
print("Salida lineal:", y_lin)

# Aplicar activaciones directamente
print("Sigmoid:", DnnLib.sigmoid(np.array([0.0, 2.0, -1.0])))