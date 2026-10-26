import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data[:, [0, 2]]  # Sepal length and petal length
y = (iris.target == 0).astype(int)  # Setosa vs Non-Setosa

print("Dataset shape:", X.shape)
print("Setosa: 1, Non-Setosa: 0")

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Implement Linear SVM
model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Non-Setosa", "Setosa"],
    zero_division=0
))

# 7. Visualize the decision boundary and margins
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)

# Plot the training data
plt.scatter(
    X_train[y_train == 0, 0], X_train[y_train == 0, 1],
    color="blue", label="Non-Setosa"
)
plt.scatter(
    X_train[y_train == 1, 0], X_train[y_train == 1, 1],
    color="red", label="Setosa"
)

# Draw the decision boundary and margins
w = model.coef_[0]
b = model.intercept_[0]

x_line = np.linspace(x_min, x_max, 300)
y_boundary = -(w[0] * x_line + b) / w[1]
y_margin_positive = -(w[0] * x_line + b - 1) / w[1]
y_margin_negative = -(w[0] * x_line + b + 1) / w[1]

plt.plot(x_line, y_boundary, "k-", label="Decision Boundary")
plt.plot(x_line, y_margin_positive, "k--", label="Margins")
plt.plot(x_line, y_margin_negative, "k--")

# Highlight support vectors
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=150, facecolors="none", edgecolors="black",
    label="Support Vectors"
)

plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Petal Length")
plt.title("Linear SVM: Setosa vs Non-Setosa")
plt.legend()
plt.tight_layout()
plt.show()