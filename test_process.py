import numpy as np
import DnnLib

# Proceso general para probar el modelo
# UNA FUNCION DE AYUDA, NO EJECUTAR SOLO

def test_model(data, params):
    # Inicializar Data
    inputs = data["images"].reshape(-1, 784) / 255
    targets = data["labels"]
    
   # Inicializar Capas
    layers = [
        DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]
    # Datos por capa
    for idx in range(len(layers)):
        layers[idx].weights = np.array(params["layers"][idx]["W"]).T
        layers[idx].bias = np.array(params["layers"][idx]["b"]).T
    
    # Forward Pass
    activation = inputs
    for layer in layers:
        activation = layer.forward(activation)
    output = activation
    
    # Precision
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions == targets)

    return accuracy