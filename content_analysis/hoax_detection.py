import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
df = pd.read_csv("../datasets/fetched_reddit_content_large.csv")  # Should have 'content' and '2_way_label' columns

# Preprocessing
X = df["content"]
y = df["2_way_label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a classifier - SGD with hinge loss (linear SVM)
model = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print top words for each class based on model coefficients
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_n = 20

# Top N positive weights (associated with label 1 = hoax)
top_hoax_indices = np.argsort(coefs)[-top_n:]
print("\nTop words predicting HOAX:")
for i in reversed(top_hoax_indices):
    print(f"{feature_names[i]}: {coefs[i]:.4f}")

# Top N negative weights (associated with label 0 = real)
top_real_indices = np.argsort(coefs)[:top_n]
print("\nTop words predicting REAL:")
for i in top_real_indices:
    print(f"{feature_names[i]}: {coefs[i]:.4f}")
