import numpy as np
import DnnLib
import matplotlib.pyplot as plt
import json

# Cargar Data de Entrenamiento
data = np.load("mnist_train.npz")

# Abrir Archivo JSON
with open("mnist_mlp_pretty.json","r") as ah:
    datos = json.load(ah)

# Inicializar Capas
layer0 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
layer1 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

# Datos Capa 0:
layer0.weights = np.array(datos["layers"][0]["W"])
layer0.bias = np.array(datos["layers"][0]["b"])

output0 = layer0.forward(data["images"])

# Datos Capa 1:
layer1.weights = np.array(datos["layers"][1]["W"])
layer1.bias = np.array(datos["layers"][1]["b"])

output1 = layer1.forward(output0)

# Salidas
print("Salida Capa 0: "+output0)
print("Salida Capa 1: "+output1)