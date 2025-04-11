# analysis/hoax_classifier.py

import os
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Load model and vectorizer ---
BASE_DIR = os.path.dirname(__file__)
try:
    model_path = os.path.join(BASE_DIR, "../models/hoax_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "../models/vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError as e:
    print(f"❌ Model files missing: {e}")
    model, vectorizer = None, None

def predict_hoax(title, content):
    if model is None or vectorizer is None:
        return "Model not loaded"
    text = preprocess(title + " " + content)
    X = vectorizer.transform([text])
    return model.predict(X)[0]
