# Logistic Regression: Diabetes Prediction
# Compare performance with and without feature scaling

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the Pima Indians Diabetes dataset
# Dataset URL (CSV): 768 rows, 8 input features, 1 target column
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

data = pd.read_csv(url, names=columns)

# 2. Preprocess: replace impossible zero values in selected medical measurements
# with missing values, then fill them with each column's median.
zero_columns = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
data[zero_columns] = data[zero_columns].replace(0, float("nan"))
data[zero_columns] = data[zero_columns].fillna(data[zero_columns].median())

# Split input features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 3. Split once so both models are evaluated on the same test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model WITHOUT feature scaling
model_unscaled = LogisticRegression(max_iter=1000)
model_unscaled.fit(X_train, y_train)
pred_unscaled = model_unscaled.predict(X_test)

# 5. Scale features: fit scaler only on training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model WITH feature scaling
model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)
pred_scaled = model_scaled.predict(X_test_scaled)

# 6. Evaluate both models
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0)
    }

results = pd.DataFrame({
    "Without Scaling": get_metrics(y_test, pred_unscaled),
    "With Scaling": get_metrics(y_test, pred_scaled)
})

print("Performance comparison:")
print(results.round(3))

# 7. Plot metric comparison
results.plot(kind="bar", figsize=(9, 5))
plt.title("Logistic Regression: With vs Without Feature Scaling")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.tight_layout()
plt.show()