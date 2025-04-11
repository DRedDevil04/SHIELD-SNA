import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import os
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_top_n_words(corpus, n=50):
    all_words = ' '.join(corpus).split()
    filtered_words = [word for word in all_words if word.lower() not in stop_words]
    return Counter(filtered_words).most_common(n)

def run_content_analysis():
    # Load CSV
    df = pd.read_csv("../datasets/fetched_reddit_content_large.csv")

    # Combine title + content
    df['text'] = df['clean_title'].fillna('') + " " + df['content'].fillna('')

    # Filter out rows with missing labels
    df = df[df['2_way_label'].notna()]

    # Clean text
    df['clean_text'] = df['text'].apply(preprocess)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['2_way_label'], test_size=0.2, random_state=42)

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        min_df=3,
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train classifier
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_tfidf, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Hoax Detection")
    plt.show()

    # Top hoax keywords
    hoax_text = df[df['2_way_label'] == 1]['clean_text']
    top_hoax_words = get_top_n_words(hoax_text)
    print("Top hoax keywords:", top_hoax_words[:50])

    # Save top hoax words to a shared file
    os.makedirs("../shared_data", exist_ok=True)
    with open("../shared_data/top_hoax_keywords.txt", "w") as f:
        for word, count in top_hoax_words:
            f.write(f"{word}\n")

    # WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(hoax_text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud - Hoax Posts")
    plt.show()

    # Save model and vectorizer
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/hoax_model.pkl")
    joblib.dump(vectorizer, "../models/vectorizer.pkl")
    print("✅ Model and vectorizer saved to ../models/")

# Only run when this file is executed directly
if __name__ == "__main__":
    run_content_analysis()
