
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Select one feature and target
X = df[["AveRooms"]].values
y = df["MedHouseVal"].values

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize X using training data only
x_mean = X_train.mean()
x_std = X_train.std()

X_train_scaled = (X_train - x_mean) / x_std
X_test_scaled = (X_test - x_mean) / x_std

# 5. Gradient Descent
m = 0.0
b = y_train.mean()
learning_rate = 0.01
epochs = 1000
n = len(X_train_scaled)

for i in range(epochs):
    y_pred = m * X_train_scaled[:, 0] + b

    dm = (2 / n) * np.sum(
        X_train_scaled[:, 0] * (y_pred - y_train)
    )
    db = (2 / n) * np.sum(y_pred - y_train)

    m = m - learning_rate * dm
    b = b - learning_rate * db

m_gd = m
b_gd = b

# Predict using Gradient Descent
y_pred_gd = m_gd * X_test_scaled[:, 0] + b_gd

# 6. Normal Equation
X_train_design = np.c_[
    np.ones(len(X_train_scaled)), X_train_scaled[:, 0]
]
X_test_design = np.c_[
    np.ones(len(X_test_scaled)), X_test_scaled[:, 0]
]

theta = np.linalg.pinv(X_train_design) @ y_train

b_ne = theta[0]
m_ne = theta[1]

# Predict using Normal Equation
y_pred_ne = X_test_design @ theta

# 7. Evaluate both models
print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, y_pred_gd))
print("R2 Score:", r2_score(y_test, y_pred_gd))

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, y_pred_ne))
print("R2 Score:", r2_score(y_test, y_pred_ne))

# 8. Plot data points and fitted lines
plt.scatter(X_test[:, 0], y_test, alpha=0.3, label="Actual data")

order = np.argsort(X_test[:, 0])
plt.plot(
    X_test[order, 0],
    y_pred_gd[order],
    color="red",
    label="Gradient Descent"
)
plt.plot(
    X_test[order, 0],
    y_pred_ne[order],
    color="green",
    linestyle="--",
    label="Normal Equation"
)

plt.xlabel("Average Rooms per Dwelling")
plt.ylabel("Median House Value (in $100,000s)")
plt.title("California Housing: Simple Linear Regression")
plt.legend()
plt.show()
