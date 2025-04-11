import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import os

def run_sentiment_analysis(show_plot=False):
    # Download VADER lexicon (only once)
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    # Load dataset
    df = pd.read_csv("../datasets/fetched_reddit_content_large.csv")
    df['clean_title'] = df['clean_title'].fillna("")

    # --- Sentiment Scoring ---
    df['vader_sentiment_raw'] = df['clean_title'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['vader_sentiment'] = df['vader_sentiment_raw'].apply(lambda x: round((x + 1) * 5, 2))
    df['textblob_polarity'] = df['clean_title'].apply(lambda x: TextBlob(x).sentiment.polarity)

    def label_sentiment(score):
        if score >= 7:
            return "positive"
        elif score <= 3:
            return "negative"
        else:
            return "neutral"

    df['sentiment_category'] = df['vader_sentiment'].apply(label_sentiment)

    # --- Threat Keyword Detection ---
    threat_keywords_path = "../shared_data/top_hoax_keywords.txt"
    if os.path.exists(threat_keywords_path):
        with open(threat_keywords_path, "r") as f:
            threat_keywords = [line.strip().lower() for line in f if line.strip()]
    else:
        print("⚠️ Warning: Threat keyword file not found.")
        threat_keywords = []

    def detect_threat_keywords(text, keywords):
        text = text.lower()
        return int(any(kw in text for kw in keywords))

    df['threat_flag'] = df['clean_title'].apply(lambda x: detect_threat_keywords(x, threat_keywords))

    # Count hoax posts with threat keywords
    hoax_threats = df[(df['2_way_label'] == 1) & (df['threat_flag'] == 1)]
    print(f"⚠️ Threat-flagged hoax posts: {len(hoax_threats)}")

    # --- Plotting ---
    if show_plot:
        sns.histplot(data=df, x='vader_sentiment', hue='2_way_label', bins=30, kde=True)
        plt.title("Sentiment Distribution (0–10): Hoax vs. Non-Hoax Posts")
        plt.xlabel("VADER Sentiment Score (0 to 10)")
        plt.ylabel("Post Count")
        plt.tight_layout()
        plt.show()

    # --- Save Results ---
    os.makedirs("../shared_data", exist_ok=True)
    df.to_csv("../shared_data/sentiment_analysis.csv", index=False)
    hoax_threats.to_csv("../shared_data/hoax_threats.csv", index=False)

    print("✅ Sentiment and threat analysis saved to ../shared_data/")

# Run independently
if __name__ == "__main__":
    run_sentiment_analysis(show_plot=True)
