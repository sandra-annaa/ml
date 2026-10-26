# KNN Image Classification using Fashion MNIST

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2. Preprocess the images
# Convert 28x28 images into 784 features
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Use a subset to reduce execution time
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

# Fashion MNIST class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 3. Experiment with different K values
k_values = [3, 5, 7, 9]
results = []

for k in k_values:
    # Create KNN classifier
    model = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Predict test labels
    start_pred = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_pred

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "K": k,
        "Accuracy": accuracy,
        "Training Time (s)": train_time,
        "Prediction Time (s)": prediction_time
    })

    print(f"K = {k}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    print("-" * 30)

# 4. Display comparison table
results_df = pd.DataFrame(results)
print("\nComparison of K values:")
print(results_df.round(4).to_string(index=False))

# 5. Plot accuracy for different K values
plt.figure(figsize=(8, 5))
plt.plot(
    results_df["K"],
    results_df["Accuracy"],
    marker="o"
)
plt.title("KNN Accuracy for Different K Values")
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# 6. Plot prediction time for different K values
plt.figure(figsize=(8, 5))
plt.plot(
    results_df["K"],
    results_df["Prediction Time (s)"],
    marker="o"
)
plt.title("KNN Prediction Time for Different K Values")
plt.xlabel("Value of K")
plt.ylabel("Prediction Time (seconds)")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# 7. Display sample test images and predictions using the best K
best_k = int(results_df.loc[results_df["Accuracy"].idxmax(), "K"])
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)

sample_images = X_test[:10]
sample_predictions = best_model.predict(sample_images)

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap="gray")
    plt.title(
        f"Pred: {class_names[sample_predictions[i]]}\n"
        f"Actual: {class_names[y_test[i]]}",
        fontsize=8
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

print(f"\nBest K value: {best_k}")