
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# 1. Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Target classes:", data.target_names)

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. MLE: Logistic Regression with negligible regularization
mle_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=10000
    ))
])

mle_model.fit(X_train, y_train)

# 4. MAP with L1 regularization (Laplace prior)
l1_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l1", C=1.0, solver="liblinear", max_iter=10000
    ))
])

l1_model.fit(X_train, y_train)

# 5. MAP with L2 regularization (Gaussian prior)
l2_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=10000
    ))
])

l2_model.fit(X_train, y_train)

# 6. Evaluate the models
models = {
    "MLE (approx.)": mle_model,
    "MAP with L1": l1_model,
    "MAP with L2": l2_model
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

print("\nPerformance Comparison:")
print(pd.DataFrame(results).round(4).to_string(index=False))

# 7. Compare parameter estimates (coefficients)
feature_names = data.feature_names

coef_table = pd.DataFrame({
    "Feature": feature_names,
    "MLE": mle_model.named_steps["model"].coef_[0],
    "MAP_L1": l1_model.named_steps["model"].coef_[0],
    "MAP_L2": l2_model.named_steps["model"].coef_[0]
})

print("\nParameter Estimates (first 10 features):")
print(coef_table.head(10).round(4).to_string(index=False))

# 8. Count non-zero coefficients
print("\nNumber of non-zero coefficients:")
for name, model in models.items():
    coefficients = model.named_steps["model"].coef_[0]
    print(name, ":", np.sum(np.abs(coefficients) > 1e-6))
