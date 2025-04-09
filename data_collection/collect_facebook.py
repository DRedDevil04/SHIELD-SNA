# collect_facebook.py
from facebook_scraper import get_posts
import pandas as pd
from textblob import TextBlob
import re

FACEBOOK_PAGES = ["cnn", "ndtv", "newyorktimes"]
KEYWORDS = ["hoax call", "bomb threat", "fake alert", "false alarm", "school hoax"]

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^A-Za-z0-9#@ ]+", '', text)
    return text.lower().strip()

def get_sentiment(text):
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity

def collect_facebook():
    data = []
    for page in FACEBOOK_PAGES:
        try:
            found = False
            for post in get_posts(page, pages=5):
                found = True
                text = post['text'] or ""
                if any(k.lower() in text.lower() for k in KEYWORDS):
                    clean = clean_text(text)
                    data.append({
                        "platform": "Facebook",
                        "post_id": post['post_id'],
                        "user_id": page,
                        "timestamp": post['time'],
                        "text": text,
                        "cleaned_text": clean,
                        "sentiment_score": get_sentiment(clean),
                        "url": post.get("post_url")
                    })
            if not found:
                print(f"[!] No posts collected from page: {page}")
        except Exception as e:
            print(f"[!] Error collecting from Facebook page '{page}': {e}")
    return data

if __name__ == "__main__":
    print("[*] Collecting Facebook posts...")
    facebook_data = collect_facebook()
    df = pd.DataFrame(facebook_data)
    df.to_csv("facebook_hoax_data.csv", index=False)
    print("[✓] Saved to 'facebook_hoax_data.csv'")
