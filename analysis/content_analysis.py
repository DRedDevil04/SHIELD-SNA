import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import io
import base64

def analyse_content(df):
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

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Hoax Detection")

    # Convert matplotlib plot to base64 image for web usage
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Print top words for each class
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_n = 20

    top_hoax_indices = np.argsort(coefs)[-top_n:]
    top_hoax_words = [(feature_names[i], coefs[i]) for i in reversed(top_hoax_indices)]

    top_real_indices = np.argsort(coefs)[:top_n]
    top_real_words = [(feature_names[i], coefs[i]) for i in top_real_indices]

    # Save top hoax words to file
    os.makedirs("./shared_data", exist_ok=True)
    with open("./shared_data/top_hoax_keywords.txt", "w") as f:
        for word, count in top_hoax_words:
            f.write(f"{word}\n")

    # Return results
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix_base64": image_base64,
        "top_hoax_words": top_hoax_words,
        "top_real_words": top_real_words
    }
