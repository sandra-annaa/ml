# Naive Bayes Text Classification using 20 Newsgroups
# Compare Multinomial Naive Bayes and Bernoulli Naive Bayes

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Load the dataset
train_data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test_data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

X_train_text = train_data.data
y_train = train_data.target

X_test_text = test_data.data
y_test = test_data.target

print("Training documents:", len(X_train_text))
print("Testing documents:", len(X_test_text))
print("Number of categories:", len(train_data.target_names))

# 2. Preprocess text and convert it into numerical features
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=20000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Number of word features:", X_train.shape[1])

# 3. Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred_mnb = mnb.predict(X_test)

# 4. Train Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
pred_bnb = bnb.predict(X_test)

# 5. Evaluate both classifiers
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
    }

mnb_metrics = evaluate_model(y_test, pred_mnb)
bnb_metrics = evaluate_model(y_test, pred_bnb)

# 6. Display comparison table
results = pd.DataFrame(
    [mnb_metrics, bnb_metrics],
    index=["Multinomial Naive Bayes", "Bernoulli Naive Bayes"]
)

print("\nPerformance Comparison:")
print(results.round(4))

# 7. Print detailed classification reports
print("\nMultinomial Naive Bayes Report:")
print(classification_report(
    y_test,
    pred_mnb,
    labels=range(len(train_data.target_names)),
    target_names=train_data.target_names,
    zero_division=0
))

print("\nBernoulli Naive Bayes Report:")
print(classification_report(
    y_test,
    pred_bnb,
    labels=range(len(train_data.target_names)),
    target_names=train_data.target_names,
    zero_division=0
))

# 8. Plot performance comparison
results.plot(kind="bar", figsize=(10, 6))

plt.title("Multinomial NB vs Bernoulli NB")
plt.ylabel("Score")
plt.xlabel("Classifier")
plt.ylim(0, 1)
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()
