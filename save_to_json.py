import json
import numpy as np

# Guardar Parametros en JSON
# Funciona para ambos modelos con valores harcoded
# Ya que ambos tienen la misma cantidad de clases y entradas

def save_params(layers, name):
    layers_json = []
    for idx, layer in enumerate(layers):
        layer_info = {
            "units": layer.bias.shape[0],
            "activation": layer.activation_type.name,
            "W": layers[idx].weights.T.tolist(),
            "b": layers[idx].bias.T.tolist()
        }
        layers_json.append(layer_info)
    
    params = {
        "input_shape": [28,28],
        "preprocess": {"scale": 255.0},
        "layers": layers_json
    }
    with open(name, "w") as ah:
        json.dump(params, ah, indent=4)