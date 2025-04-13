import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import os
import io
import base64

def analyse_sentiment(df=None, show_plot=False):
    # Download VADER lexicon if not already present
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()

    # Load dataset if not provided
    if df is None:
        df = pd.read_csv("../datasets/fetched_reddit_content_large.csv")

    df['clean_title'] = df['clean_title'].fillna("")

    # Sentiment scoring
    df['vader_sentiment_raw'] = df['clean_title'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['vader_sentiment'] = df['vader_sentiment_raw'].apply(lambda x: round((x + 1) * 5, 2))
    df['textblob_polarity'] = df['clean_title'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Label sentiment
    def label_sentiment(score):
        if score >= 7:
            return "positive"
        elif score <= 3:
            return "negative"
        else:
            return "neutral"

    df['sentiment_category'] = df['vader_sentiment'].apply(label_sentiment)

    # Threat keyword detection
    threat_keywords_path = "./shared_data/top_hoax_keywords.txt"
    threat_keywords = []
    if os.path.exists(threat_keywords_path):
        with open(threat_keywords_path, "r") as f:
            threat_keywords = [line.strip().lower() for line in f if line.strip()]
    else:
        print("⚠️ Warning: Threat keyword file not found.")

    def detect_threat_keywords(text, keywords):
        text = text.lower()
        return int(any(kw in text for kw in keywords))

    df['threat_flag'] = df['clean_title'].apply(lambda x: detect_threat_keywords(x, threat_keywords))

    # Filter hoax + flagged
    hoax_threats = df[(df['2_way_label'] == 1) & (df['threat_flag'] == 1)]
    print(f"⚠️ Threat-flagged hoax posts: {len(hoax_threats)}")

    # Optional: create plot
    base64_plot = None
    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x='vader_sentiment', hue='2_way_label', bins=30, kde=True, ax=ax)
        ax.set_title("Sentiment Distribution (0–10): Hoax vs. Non-Hoax Posts")
        ax.set_xlabel("VADER Sentiment Score (0 to 10)")
        ax.set_ylabel("Post Count")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_plot = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

    # Save results
    os.makedirs("./shared_data", exist_ok=True)
    df.to_csv("./shared_data/sentiment_analysis.csv", index=False)
    hoax_threats.to_csv("./shared_data/hoax_threats.csv", index=False)

    print("✅ Sentiment and threat analysis saved to ./shared_data/")

    return {
        "full_df": df,
        "hoax_threats": hoax_threats,
        "base64_plot": base64_plot,
        "threat_keywords": threat_keywords,
        "threat_hoax_count": len(hoax_threats)
    }

# Run directly
if __name__ == "__main__":
    analyse_sentiment(show_plot=True)
