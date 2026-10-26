
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Original training data:", X_train.shape)
print("Original testing data:", X_test.shape)

# 2. Preprocess the images
# Flatten each 28 x 28 image into 784 features
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 3. Select subsets to reduce computation time
X_train, _, y_train, _ = train_test_split(
    X_train, y_train,
    train_size=8000,
    random_state=42,
    stratify=y_train
)

X_test, _, y_test, _ = train_test_split(
    X_test, y_test,
    train_size=2000,
    random_state=42,
    stratify=y_test
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 4. Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Define SVM classifiers with different kernels
models = {
    "Linear": SVC(kernel="linear", C=1.0),
    "Polynomial": SVC(kernel="poly", degree=3, C=1.0, gamma="scale"),
    "RBF": SVC(kernel="rbf", C=1.0, gamma="scale")
}

results = {}
accuracies = []
training_times = []

# 6. Train and evaluate each classifier
for name, model in models.items():
    print("\nTraining", name, "SVM...")

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        "Accuracy": accuracy,
        "Training Time": train_time,
        "Prediction Time": predict_time
    }

    accuracies.append(accuracy)
    training_times.append(train_time)

    print("Accuracy:", round(accuracy, 4))
    print("Training Time:", round(train_time, 2), "seconds")
    print("Prediction Time:", round(predict_time, 2), "seconds")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# 7. Compare performance
print("\nPERFORMANCE COMPARISON")
print("-" * 65)
print("{:<15} {:<15} {:<20} {:<15}".format(
    "Kernel", "Accuracy", "Training Time(s)", "Prediction(s)"
))

for name, result in results.items():
    print("{:<15} {:<15.4f} {:<20.2f} {:<15.2f}".format(
        name,
        result["Accuracy"],
        result["Training Time"],
        result["Prediction Time"]
    ))

# 8. Visualize accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), accuracies)
plt.title("SVM Kernel Performance on Fashion MNIST")
plt.xlabel("Kernel Type")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 9. Visualize training time comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), training_times)
plt.title("Training Time Comparison")
plt.xlabel("Kernel Type")
plt.ylabel("Training Time (seconds)")
plt.tight_layout()
plt.show()
