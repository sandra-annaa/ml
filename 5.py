
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# 1. Load the 20 Newsgroups training dataset
data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

print("Number of documents:", len(data.data))

# 2. Convert text documents into word-count vectors
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=1000
)

X = vectorizer.fit_transform(data.data)
vocabulary = vectorizer.get_feature_names_out()

# Total count of each word across all documents
counts = np.asarray(X.sum(axis=0)).ravel()

V = len(vocabulary)
N = counts.sum()

print("Vocabulary size:", V)
print("Total word occurrences:", N)

# 3. MLE: estimate word probabilities from observed counts
p_mle = counts / N

# 4. MAP estimation with different symmetric Dirichlet priors
# alpha > 1 gives an interior posterior mode
alphas = [1.1, 1.5, 2.0]

estimates = {
    "MLE": p_mle
}

for alpha in alphas:
    p_map = (counts + alpha - 1) / (N + V * (alpha - 1))
    estimates[f"MAP alpha={alpha}"] = p_map

# 5. Compare the estimated distributions
rows = []

for name, probabilities in estimates.items():
    entropy = -np.sum(
        probabilities[probabilities > 0] *
        np.log(probabilities[probabilities > 0])
    )

    rows.append({
        "Method": name,
        "Probability sum": probabilities.sum(),
        "Entropy": entropy,
        "Zero probabilities": np.sum(probabilities == 0),
        "Maximum word probability": probabilities.max()
    })

print("\nComparison of estimators:")
print(pd.DataFrame(rows).round(6).to_string(index=False))

# 6. Display the 10 most probable words for each method
for name, probabilities in estimates.items():
    top_indices = np.argsort(probabilities)[-10:][::-1]

    top_words = pd.DataFrame({
        "Word": vocabulary[top_indices],
        "Probability": probabilities[top_indices]
    })

    print(f"\nTop 10 words using {name}:")
    print(top_words.round(6).to_string(index=False))

# 7. Compare probabilities for selected words
selected_words = ["computer", "government", "space", "game", "religion"]
word_to_index = {
    word: i for i, word in enumerate(vocabulary)
}

print("\nSelected word probabilities:")
for word in selected_words:
    if word in word_to_index:
        i = word_to_index[word]
        print(f"\nWord: {word}")
        for name, probabilities in estimates.items():
            print(f"{name}: {probabilities[i]:.8f}")
    else:
        print(f"\nWord '{word}' is not in the selected vocabulary.")
