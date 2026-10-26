
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Wine Quality dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

print("Dataset:")
print(df.head())
print("Dataset shape:", df.shape)

# 2. Preprocess the data
df = df.dropna()

X = df.drop("quality", axis=1)
y = df["quality"]

print("\nQuality distribution:")
print(y.value_counts().sort_index())

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Define MLP architectures
architectures = {
    "1 hidden layer (32 neurons)": (32,),
    "1 hidden layer (64 neurons)": (64,),
    "2 hidden layers (64, 32)": (64, 32),
    "2 hidden layers (128, 64)": (128, 64),
    "3 hidden layers (128, 64, 32)": (128, 64, 32)
}

results = {}

# 6. Train and evaluate each MLP
for name, layers in architectures.items():
    print("\nTraining:", name)

    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        "Accuracy": accuracy,
        "Training Time": train_time,
        "Model": model
    }

    print("Accuracy:", round(accuracy, 4))
    print("Training time:", round(train_time, 2), "seconds")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# 7. Compare the architectures
print("\nARCHITECTURE COMPARISON")
print("-" * 65)
print("{:<35} {:<15} {:<15}".format(
    "Architecture", "Accuracy", "Time (seconds)"
))

for name, result in results.items():
    print("{:<35} {:<15.4f} {:<15.2f}".format(
        name,
        result["Accuracy"],
        result["Training Time"]
    ))

# 8. Visualize accuracy comparison
plt.figure(figsize=(10, 5))
plt.bar(
    results.keys(),
    [r["Accuracy"] for r in results.values()]
)
plt.title("MLP Architecture Accuracy Comparison")
plt.xlabel("Network Architecture")
plt.ylabel("Accuracy")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# 9. Visualize training time
plt.figure(figsize=(10, 5))
plt.bar(
    results.keys(),
    [r["Training Time"] for r in results.values()]
)
plt.title("MLP Training Time Comparison")
plt.xlabel("Network Architecture")
plt.ylabel("Training Time (seconds)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()