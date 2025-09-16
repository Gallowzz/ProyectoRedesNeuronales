import numpy as np
import DnnLib

# Cargar Data de Entrenamiento
data = np.load("mnist_train.npz")
inputs = data["images"].reshape(-1, 784) / 255
targets = data["labels"]

# Convertir a One-Hot
y = np.zeros((60000, 10), dtype=np.float64)
y[np.arange(60000), targets] = 1.0

# Generar Capas
layers = [
    DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
    DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX),
]

# Inicializar Optimizadores
optimizers = [
    ("SGD", DnnLib.SGD(0.001)),
    ("SGD+Momentum", DnnLib.SGD(0.001, 0.9)),
    ("Adam", DnnLib.Adam(0.001)),
    ("RMSprop", DnnLib.RMSprop(0.001))
]

# Entrenamiento con distintos optimizadores
for opt_name, optimizer in optimizers:
    print(f"\n--- Entrenando con {opt_name} ---")

    layers = [
        DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX),
    ]
    optimizer.reset()

    for epoch in range(20):
        # Forward Pass
        activation = inputs
        for layer in layers:
            activation = layer.forward(activation)
        output = activation

        # Perdida
        loss = DnnLib.cross_entropy(output, y)

        # Backward Pass
        grad = DnnLib.cross_entropy_gradient(output, y)
        for layer in reversed(layers):
            grad = layer.backward(grad)
            optimizer.update(layer)

        if epoch % 5 == 0:
            predicted_classes = np.argmax(output, axis=1)
            accuracy = np.mean(predicted_classes == y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Final
    print(f" Perdida final con {opt_name}: {loss:.6f}")
