
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (download Online Retail.xlsx first)
df = pd.read_excel("Online Retail.xlsx")

# Preprocess data
df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["Total"] = df["Quantity"] * df["UnitPrice"]

# Create customer-level features
customer = df.groupby("CustomerID").agg(
    TotalSpend=("Total", "sum"),
    TotalQuantity=("Quantity", "sum"),
    Transactions=("InvoiceNo", "nunique")
).reset_index()

# Create target customer segments
customer["Segment"] = pd.qcut(
    customer["TotalSpend"], q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

X = customer[["TotalQuantity", "Transactions"]]
y = customer["Segment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ID3-style tree: entropy criterion
model = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=42
)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Visualize decision tree
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns,
          class_names=model.classes_.astype(str),
          filled=True, rounded=True)
plt.show()

# Feature importance
print(pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False))
