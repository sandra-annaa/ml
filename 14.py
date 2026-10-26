
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data.astype("float32")
y = mnist.target.astype("int64")

print("Dataset shape:", X.shape)

# 2. Preprocess the data
# Normalize pixel values from 0-255 to 0-1
X = X / 255.0

# Use a stratified subset to reduce training time
X_train, _, y_train, _ = train_test_split(
    X, y,
    train_size=10000,
    random_state=42,
    stratify=y
)

X_test, _, y_test, _ = train_test_split(
    X, y,
    train_size=2000,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Define neural networks with different activations
activations = {
    "Sigmoid": "logistic",
    "ReLU": "relu",
    "Tanh": "tanh"
}

results = {}

# 4. Train and evaluate each model
for name, activation in activations.items():
    print("\nTraining model with", name, "activation...")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation=activation,
        solver="adam",
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Predict test labels
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        "Accuracy": accuracy,
        "Training Time": training_time,
        "Iterations": model.n_iter_,
        "Loss": model.loss_,
        "Loss Curve": model.loss_curve_,
        "Model": model
    }

    print("Accuracy:", round(accuracy, 4))
    print("Training time:", round(training_time, 2), "seconds")
    print("Iterations:", model.n_iter_)
    print("Final training loss:", round(model.loss_, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# 5. Compare model performance
print("\nPERFORMANCE COMPARISON")
print("-" * 65)
print("{:<12} {:<12} {:<16} {:<12} {:<12}".format(
    "Activation", "Accuracy", "Time (seconds)",
    "Iterations", "Final Loss"
))

for name, result in results.items():
    print("{:<12} {:<12.4f} {:<16.2f} {:<12} {:<12.4f}".format(
        name,
        result["Accuracy"],
        result["Training Time"],
        result["Iterations"],
        result["Loss"]
    ))

# 6. Plot classification accuracy
plt.figure(figsize=(8, 5))
plt.bar(
    results.keys(),
    [r["Accuracy"] for r in results.values()]
)
plt.title("MNIST Accuracy for Different Activation Functions")
plt.xlabel("Activation Function")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 7. Plot training time
plt.figure(figsize=(8, 5))
plt.bar(
    results.keys(),
    [r["Training Time"] for r in results.values()]
)
plt.title("Training Time Comparison")
plt.xlabel("Activation Function")
plt.ylabel("Training Time (seconds)")
plt.tight_layout()
plt.show()

# 8. Plot convergence using training loss
plt.figure(figsize=(8, 5))

for name, result in results.items():
    plt.plot(result["Loss Curve"], label=name)

plt.title("Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.legend()
plt.tight_layout()
plt.show()