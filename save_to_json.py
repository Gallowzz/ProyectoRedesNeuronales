import json
import numpy as np

# Guardar Parametros en JSON
# Funciona para ambos modelos con valores harcoded
# Ya que ambos tienen la misma cantidad de clases y entradas

def save_params(layers, name):
    params = {
        "input_shape": [28,28],
        "preprocess": {"scale": 255.0},
        "layers":[
            {"units": 128, "activation": "RELU", "W":layers[0].weights.T.tolist(), "b":layers[0].bias.T.tolist()},
            {"units": 10, "activation": "SOFTMAX", "W":layers[1].weights.T.tolist(), "b":layers[1].bias.T.tolist()}
        ]
    }
    with open(name, "w") as ah:
        json.dump(params, ah, indent=4)