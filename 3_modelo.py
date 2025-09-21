import numpy as np
import DnnLib
import json

# Cargar Data de Entrenamiento
data = np.load("mnist_train.npz")
inputs = data["images"].reshape(-1, 784) / 255
targets = data["labels"]

# Abrir Archivo JSON
with open("mnist_mlp_pretty.json","r") as ah:
    datos = json.load(ah)

# Inicializar Capas
layer0 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
layer1 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

# Datos Capa 0
layer0.weights = np.array(datos["layers"][0]["W"]).T
layer0.bias = np.array(datos["layers"][0]["b"]).T

# Datos Capa 1
layer1.weights = np.array(datos["layers"][1]["W"]).T
layer1.bias = np.array(datos["layers"][1]["b"]).T

# Forward Pass
output0 = layer0.forward(inputs)
output1 = layer1.forward(output0)

# Precision
predictions = np.argmax(output1, axis=1)
accuracy = np.mean(predictions == targets)
print("Precision: ",accuracy)