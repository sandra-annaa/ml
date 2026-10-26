
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load the Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

print("Dataset shape:", X.shape)
print("Number of images:", len(X))
print("Number of features:", X.shape[1])

# 2. Preprocess the data
# Pixel values range from 0 to 16, so scale them to 0 to 1
X_scaled = X / 16.0

# 3. Try different numbers of clusters
k_values = range(2, 16)

inertia_values = []
silhouette_values = []

for k in k_values:
    # Create K-means model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Fit the model and assign each image to a cluster
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate inertia
    inertia_values.append(kmeans.inertia_)

    # Calculate silhouette score
    score = silhouette_score(
        X_scaled,
        cluster_labels,
        sample_size=min(2000, len(X_scaled)),
        random_state=42
    )

    silhouette_values.append(score)

    print(
        f"K = {k:2d} | "
        f"Inertia = {kmeans.inertia_:.2f} | "
        f"Silhouette Score = {score:.4f}"
    )

# 4. Display results in a table
results = pd.DataFrame({
    "Number of Clusters (K)": list(k_values),
    "Inertia": inertia_values,
    "Silhouette Score": silhouette_values
})

print("\nClustering Results:")
print(results.round(4).to_string(index=False))

# 5. Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertia_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(list(k_values))
plt.grid(True)
plt.show()

# 6. Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K Values")
plt.xticks(list(k_values))
plt.grid(True)
plt.show()

# 7. Find the K with the highest silhouette score
best_index = np.argmax(silhouette_values)
best_k = list(k_values)[best_index]

print("\nBest K based on silhouette score:", best_k)
print("Highest silhouette score:", round(silhouette_values[best_index], 4))
