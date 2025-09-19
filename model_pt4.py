import numpy as np
import DnnLib
import json
import argparse

# Crear Argumentos para el parser
parser = argparse.ArgumentParser(description="Train MNIST MLP Model.")
parser.add_argument('--mode', type=str, default="Test", help="Test or Train the Model")
parser.add_argument('--epochs', type=int, default=10, help="Epoch Amount")
parser.add_argument('--batch-size', type=int, default=64, help="Batch Size")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning Rate for Optimizer")
parser.add_argument('--optimizer', type=str, default="Adam", help="Optimizer to train with(SGD, SGD+Momentum, Adam, or RMSprop)")
args = parser.parse_args()

# Guardar Parametros en JSON
def save_params(layers):
    params = {
        "layers":[
            {"units": 128, "activation": "RELU", "W":layers[0].weights.T.tolist(), "b":layers[0].bias.T.tolist()},
            {"units": 10, "activation": "SOFTMAX", "W":layers[1].weights.T.tolist(), "b":layers[1].bias.T.tolist()}
        ]
    }
    name = "new_nmist_model.json"
    with open(name, "w") as ah:
        json.dump(params, ah, indent=4)

# Entrenamiento
def train():
    # Cargar Data de Entrenamiento
    data = np.load("mnist_train.npz")
    inputs = data["images"].reshape(-1, 784) / 255
    targets = data["labels"]
    
    # Convertir a One-Hot
    y = np.zeros((60000, 10), dtype=np.float32)
    y[np.arange(60000), targets] = 1.0
    
    # Inicializar Optimizador
    learning_rate = args.learning_rate
    opt_name = args.optimizer
    if opt_name == "SGD":
        optimizer = DnnLib.SGD(learning_rate)
    elif opt_name == "SGD+Momentum":
        optimizer = DnnLib.SGD(learning_rate, 0.9)
    elif opt_name == "Adam":
        optimizer = DnnLib.Adam(learning_rate)
    elif opt_name == "RMSProp":
        optimizer = DnnLib.RMSProp(learning_rate)
    else:
        print("No se reconoce el Optimizador, utilizadon ADAM")
        opt_name = "Adam"
        optimizer = DnnLib.Adam(learning_rate)

    print(f"\n--- Entrenando con {opt_name} ---")

    # Inicializar Red
    layers = [
        DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]
    optimizer.reset()
    
    n_samples = inputs.shape[0]
    
    for epoch in range(args.epochs):
        # Mezclar Data
        indexes = np.random.permutation(n_samples)
        X_shuffled = inputs[indexes]
        y_shuffled = y[indexes]
            
        epoch_loss = 0.0
        n_batches = 0
        batch_size = args.batch_size
            
        # Generar Sub-Batches
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
    
            # Forward Pass
            activation = X_batch
            for layer in layers:
                activation = layer.forward(activation)
            output = activation
    
            # Perdida
            loss = DnnLib.cross_entropy(output, y_batch)
                
            # Backward pass
            grad = DnnLib.cross_entropy_gradient(output, y_batch)
            for layer in reversed(layers):
                grad = layer.backward(grad)
            optimizer.update(layer)
    
            epoch_loss += loss
            n_batches += 1
    
        # Precision
        avg_loss = epoch_loss / n_batches
        predicted_classes = np.argmax(output, axis=1)
        target_classes = np.argmax(y_batch, axis=1)
        accuracy = np.mean(predicted_classes == target_classes)
        print(f"Epoca {epoch+1}, Perdida Promedio: {avg_loss:.4f}, Precision: {accuracy:.4f}")

    # Guardar Parametros
    save_params(layers)
    print("Modelo Guardado Exitosamente")
        
# Prueba
def test():
    data = np.load("mnist_test.npz")
    inputs = data["images"].reshape(-1, 784) / 255
    targets = data["labels"]
    
    # Abrir Archivo JSON
    with open("new_nmist_model.json","r") as ah:
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

# Detectar Modo
if args.mode == "Test":
    test()
elif args.mode == "Train": 
    train()
else:
    "No hay Modo"