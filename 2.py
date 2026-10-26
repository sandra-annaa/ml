
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Auto MPG dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

df = pd.read_csv(
    url,
    sep=r"\s+",
    names=columns,
    na_values="?",
    comment="\t",
    skipinitialspace=True
)

# 2. Preprocess the dataset
df = df[["displacement", "mpg"]].dropna()

X = df[["displacement"]].values
y = df["mpg"].values

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train and evaluate Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

# 5. Polynomial Regression for degrees 2 and 3
models = {}

for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    models[degree] = (poly, model)

    print("\nPolynomial Regression - Degree", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

# 6. Visualize the fitted models
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.scatter(X, y, alpha=0.4, label="Actual data")
plt.plot(
    X_plot,
    linear_model.predict(X_plot),
    label="Linear Regression"
)

for degree, (poly, model) in models.items():
    X_plot_poly = poly.transform(X_plot)
    plt.plot(
        X_plot,
        model.predict(X_plot_poly),
        label=f"Polynomial Degree {degree}"
    )

plt.xlabel("Engine Displacement")
plt.ylabel("Miles per Gallon (MPG)")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()
