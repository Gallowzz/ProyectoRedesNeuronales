# Librerias
import numpy as np
import DnnLib
import json
import argparse

# Funciones de Ayuda
from save_to_json import save_params
from initialize_optimizers import init_opts
from test_process import test_model

# Crear Argumentos para el parser
parser = argparse.ArgumentParser(description="Train MNIST MLP Model.")
parser.add_argument('--mode', type=str, default="Test", help="Test or Train the Model")
parser.add_argument('--epochs', type=int, default=10, help="Epoch Amount")
parser.add_argument('--batch-size', type=int, default=64, help="Batch Size")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning Rate for Optimizer")
parser.add_argument('--optimizer', type=str, default="Adam", help="Optimizer to train with(SGD, SGD+Momentum, Adam, or RMSprop)")
args = parser.parse_args()

# Estructura de la Red para el modelo
structure = [
    DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
    DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
]

# Entrenamiento
def train():
    # Cargar Data de Entrenamiento
    data = np.load("./datafiles/mnist_train.npz")
    inputs = data["images"].reshape(-1, 784) / 255
    targets = data["labels"]
    
    # Convertir a One-Hot
    y = np.zeros((60000, 10), dtype=np.float32)
    y[np.arange(60000), targets] = 1.0
    
    # Inicializar Optimizador
    learning_rate = args.learning_rate
    opt_name = args.optimizer
    optimizer = init_opts(opt_name, learning_rate)

    print(f"\n--- Entrenando con {optimizer[0]} ---")

    # Inicializar Red
    layers = structure
    optimizer[1].reset()
    n_samples = inputs.shape[0]
    
    for epoch in range(args.epochs):
        # Mezclar Data
        indexes = np.random.permutation(n_samples)
        X_shuffled = inputs[indexes]
        y_shuffled = y[indexes]
            
        epoch_loss = 0.0
        n_batches = 0
        batch_size = args.batch_size
        # Para predicciones
        correct = 0
        total = 0
            
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
                optimizer[1].update(layer)

            # Prediccion
            predicted_classes = np.argmax(output, axis=1)
            target_classes = np.argmax(y_batch, axis=1)
            
            # Metricas
            epoch_loss += loss
            n_batches += 1
            correct += np.sum(predicted_classes == target_classes)
            total += len(y_batch)
    
        # Precision
        avg_loss = epoch_loss / n_batches
        accuracy = correct/total
        print(f"Epoca {epoch+1}, Perdida Data: {avg_loss:.4f}, Precision: {accuracy:.4f}")

    # Guardar Parametros
    save_params(layers, "new_mnist_model.json")
    print("Modelo Guardado Exitosamente")
        
# Prueba
def test():
    # Cargar Data
    data = np.load("./datafiles/mnist_test.npz")
    
    # Abrir Archivo JSON
    with open("new_mnist_model.json","r") as ah:
        params = json.load(ah)
    
    accuracy = test_model(data, params, structure)
    
    print("Precision: ", accuracy)

# Detectar Modo
if args.mode.lower() == "test":
    test()
elif args.mode.lower() == "train": 
    train()
else:
    "No hay Modo"