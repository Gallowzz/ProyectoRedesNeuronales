import numpy as np
import DnnLib

# Cargar Data de Entrenamiento
data = np.load("mnist_train.npz")
inputs = data["images"].reshape(-1, 784) / 255
targets = data["labels"]

# Convertir a One-Hot
y = np.zeros((60000, 10), dtype=np.float32)
y[np.arange(60000), targets] = 1.0

# Inicializar Optimizadores
optimizers = [
    ("SGD", DnnLib.SGD(0.001)),
    ("SGD+Momentum", DnnLib.SGD(0.001, 0.9)),
    ("Adam", DnnLib.Adam(0.001)),
    ("RMSprop", DnnLib.RMSprop(0.001))
]

for opt_name, optimizer in optimizers:
    print(f"\n--- Entrenando con {opt_name} ---")

    layers = [
        DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU),
        DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    ]
    optimizer.reset()

    n_samples = inputs.shape[0]

    for epoch in range(5):
        # Mezclar Data
        indexes = np.random.permutation(n_samples)
        X_shuffled = inputs[indexes]
        y_shuffled = y[indexes]
        
        epoch_loss = 0.0
        n_batches = 0
        batch_size = 64
        
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