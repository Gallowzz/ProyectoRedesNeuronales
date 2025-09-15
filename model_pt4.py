import numpy as np
import DnnLib

# Generar Data
np.random.seed(42)
n_samples = 2500
n_features = 8

data = np.random.randn(n_samples, n_features).astype(np.float64)

# Generar Clases
n_classes = 5

y_labels = np.zeros(n_samples, dtype=int)
y_labels[:500] = 0
y_labels[500:1000] = 1
y_labels[1000:1500] = 2
y_labels[1500:2000] = 3
y_labels[2000:] = 4

y = np.zeros((n_samples, n_classes), dtype=np.float64)
y[np.arange(n_samples), y_labels] = 1.0

# Generar Capas
layers = [
    DnnLib.DenseLayer(8, 15, DnnLib.ActivationType.RELU),
    DnnLib.DenseLayer(15, 18, DnnLib.ActivationType.RELU),
    DnnLib.DenseLayer(18, 12, DnnLib.ActivationType.RELU),
    DnnLib.DenseLayer(12, 5, DnnLib.ActivationType.SOFTMAX),
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
        DnnLib.DenseLayer(8, 15, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(15, 18, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(18, 12, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(12, 5, DnnLib.ActivationType.SOFTMAX),
    ]
    optimizer.reset()

    for epoch in range(20):
        # Forward Pass
        activation = data
        for layer in layers:
            activation = layer.forward(activation)
        output = activation

        # Perdida
        loss = DnnLib.mse(output, y)

        # Backward Pass
        grad = DnnLib.mse_gradient(output, y)
        for layer in reversed(layers):
            grad = layer.backward(grad)
            optimizer.update(layer)

        if epoch % 5 == 0:
            ss_res = np.sum((y - output) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            print(f" Epoca {epoch}, Perdida: {loss:.6f}, R²: {r2_score:.4f}")

    # Final
    print(f" Perdida final con {opt_name}: {loss:.6f}")