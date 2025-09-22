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
parser = argparse.ArgumentParser(description="Train Fashion MNIST MLP Model.")
parser.add_argument('--mode', type=str, default="Test", help="Test or Train the Model")
parser.add_argument('--epochs', type=int, default=10, help="Epoch Amount")
parser.add_argument('--batch-size', type=int, default=64, help="Batch Size")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning Rate for Optimizer")
parser.add_argument('--optimizer', type=str, default="Adam", help="Optimizer to train with(SGD, SGD+Momentum, Adam, or RMSprop)")
args = parser.parse_args()

# Estructura de la Red
structure = [
    DnnLib.DenseLayer(784, 352, DnnLib.ActivationType.RELU),
    DnnLib.Dropout(0.2),
    DnnLib.DenseLayer(352, 156, DnnLib.ActivationType.RELU),
    DnnLib.Dropout(0.5),
    DnnLib.DenseLayer(156, 72, DnnLib.ActivationType.RELU),
    DnnLib.Dropout(0.1),
    DnnLib.DenseLayer(72, 10, DnnLib.ActivationType.SOFTMAX)
]

# Funcion aparte para el forward para simplificar la implementacion del dropout
def forward_pass (layers, X, training):
    activation = X
    for layer in layers:
        if hasattr(layer, 'training'):
            layer.training = training
        activation = layer.forward(activation)
    return activation

# Entrenamiento
def train():
    # Cargar Data de Entrenamiento
    data = np.load("./datafiles/fashion_mnist_train.npz")
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

    # Agregar Regularizacion
    for layer in layers:
        if not hasattr(layer, 'training'):
            layer.set_regularizer(DnnLib.RegularizerType.L2, 0.001)

    n_samples = inputs.shape[0]
    
    for epoch in range(args.epochs):
        # Mezclar Data
        indexes = np.random.permutation(n_samples)
        X_shuffled = inputs[indexes]
        y_shuffled = y[indexes]
            
        total_loss = 0.0
        data_loss = 0.0
        reg_loss = 0.0
        val_loss = 0.0
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
            output = forward_pass(layers, X_batch, True)
            
            # Perdida
            epoch_data_loss = DnnLib.cross_entropy(output, y_batch)
            epoch_reg_loss = 0.0
            for layer in layers:
                if not hasattr(layer, 'training'):
                    epoch_reg_loss += layer.compute_regularization_loss() / (n_batches + 1)
            loss = epoch_data_loss + epoch_reg_loss
            
            # Backward pass
            grad = DnnLib.cross_entropy_gradient(output, y_batch)
            for layer in reversed(layers):
                grad = layer.backward(grad)
                if not hasattr(layer, 'training'):
                    optimizer[1].update(layer)

            # Prediccion
            predicted_classes = np.argmax(output, axis=1)
            target_classes = np.argmax(y_batch, axis=1)

            # Validacion con Dropout Desactivado
            val_output = forward_pass(layers, X_batch, False)
            epoch_val_loss = DnnLib.cross_entropy(val_output, y_batch)
            
            # Metricas
            data_loss += epoch_data_loss
            reg_loss += epoch_reg_loss
            total_loss += loss
            val_loss += epoch_val_loss
            n_batches += 1
            correct += np.sum(predicted_classes == target_classes)
            total += len(y_batch)
    
        # Precision
        avg_data_loss = data_loss / n_batches
        avg_reg_loss = reg_loss / n_batches
        avg_loss = total_loss / n_batches
        avg_val_loss = val_loss / n_batches
        accuracy = correct/total

        # Resultados por Epoca
        print(f"Epoca {epoch+1}:")
        print(f"- Perdida Data: {avg_data_loss:.4f}, Perdida Regularizacion: {avg_reg_loss:.4f}, ")
        print(f"- Perdida Total: {avg_loss:.4f}, Perdida Valoracion: {avg_val_loss:.4f}")
        print(f"- Precision: {accuracy:.4f}")

    # Guardar Parametros
    dense_layers = []
    for layer in layers:
        if not hasattr(layer, 'training'):
            dense_layers.append(layer)
    save_params(dense_layers, "new_fashion_mnist_model.json")
    print("Modelo Guardado Exitosamente")
        
# Prueba
def test():
    data = np.load("./datafiles/fashion_mnist_test.npz")
    
    # Abrir Archivo JSON
    with open("new_fashion_mnist_model.json","r") as ah:
        params = json.load(ah)

    layers = []
    for layer in structure:
        if not hasattr(layer, 'training'):
            layers.append(layer)
    
    accuracy = test_model(data, params, layers)
    
    print("Precision: ",accuracy)

# Detectar Modo
if args.mode.lower() == "test":
    test()
elif args.mode.lower() == "train": 
    train()
else:
    "No hay Modo"