import numpy as np
import matplotlib.pyplot as plt

data = np.load("mnist_train.npz")
images = data["images"]
labels = data["labels"]

print("Shape imágenes:", images.shape)
print("Shape etiquetas:", labels.shape)

plt.figure(figsize=(6,6))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")
plt.show()