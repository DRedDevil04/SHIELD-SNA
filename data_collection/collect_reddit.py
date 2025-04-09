# collect_reddit.py
import praw
import pandas as pd
from textblob import TextBlob
import re
from datetime import datetime
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load credentials from .env
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_AGENT = os.getenv("REDDIT_AGENT")

KEYWORDS = ["hoax call", "bomb threat", "fake alert", "false alarm", "school hoax"]
REDDIT_LIMIT = 100

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^A-Za-z0-9#@ ]+", '', text)
    return text.lower().strip()

def get_sentiment(text):
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity

def collect_reddit():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_AGENT
    )
    posts = []
    query = " OR ".join(KEYWORDS)

    for submission in reddit.subreddit("news+worldnews+conspiracy").search(query, limit=REDDIT_LIMIT):
        raw_text = submission.title + "\n" + submission.selftext
        clean = clean_text(raw_text)
        posts.append({
            "platform": "Reddit",
            "post_id": submission.id,
            "user_id": submission.author.name if submission.author else "deleted",
            "timestamp": datetime.fromtimestamp(submission.created_utc, timezone.utc),
            "text": raw_text,
            "cleaned_text": clean,
            "sentiment_score": get_sentiment(clean),
            "url": submission.url
        })
    return posts

if __name__ == "__main__":
    print("[*] Collecting Reddit posts...")
    reddit_data = collect_reddit()
    df = pd.DataFrame(reddit_data)
    df.to_csv("reddit_hoax_data.csv", index=False)
    print("[✓] Saved to 'reddit_hoax_data.csv'")
