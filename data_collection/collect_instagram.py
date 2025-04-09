import instaloader
from instaloader import Profile
import pandas as pd
from textblob import TextBlob
import re
import os
import time
import random
from dotenv import load_dotenv

# Load credentials
load_dotenv()

INSTAGRAM_PROFILES = ["cnn", "ndtv", "newyorktimes"]
KEYWORDS = ["hoaxcall", "bomb threat", "school hoax"]
MAX_INSTAGRAM_POSTS = 50

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^A-Za-z0-9#@ ]+", '', text)
    return text.lower().strip()

def get_sentiment(text):
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity

def keyword_match(text, keywords):
    return any(keyword.lower() in text.lower() for keyword in keywords)

def collect_instagram():
    L = instaloader.Instaloader()
    results = []

    username = os.getenv("INSTA_USERNAME")
    password = os.getenv("INSTA_PASSWORD")

    if not username or not password:
        print("[!] Instagram credentials not found in .env file.")
        return []

    try:
        L.load_session_from_file(username)
        print("[✓] Session loaded.")
    except FileNotFoundError:
        print("[!] No saved session found. Logging in...")
        try:
            L.login(username, password)
            L.save_session_to_file()
        except Exception as e:
            print(f"[!] Login failed: {e}")
            return []

    for profile_name in INSTAGRAM_PROFILES:
        try:
            print(f"[*] Collecting from @{profile_name}...")
            profile = Profile.from_username(L.context, profile_name)
            count = 0

            for post in profile.get_posts():
                caption = post.caption or ""
                clean = clean_text(caption)

                if keyword_match(clean, KEYWORDS):
                    results.append({
                        "platform": "Instagram",
                        "profile": profile_name,
                        "post_id": post.shortcode,
                        "user_id": post.owner_username,
                        "timestamp": post.date_utc,
                        "text": caption,
                        "cleaned_text": clean,
                        "sentiment_score": get_sentiment(clean),
                        "url": f"https://instagram.com/p/{post.shortcode}"
                    })
                    count += 1

                if count >= MAX_INSTAGRAM_POSTS:
                    break

            time.sleep(random.randint(10, 20))  # Random delay between profiles

        except Exception as e:
            print(f"[!] Error collecting posts from @{profile_name}: {e}")
            time.sleep(random.randint(15, 30))  # Longer delay after error

    return results

if __name__ == "__main__":
    print("[*] Collecting Instagram posts...")
    instagram_data = collect_instagram()

    if instagram_data:
        df = pd.DataFrame(instagram_data)
        df.to_csv("instagram_hoax_data.csv", index=False)
        print("[✓] Saved to 'instagram_hoax_data.csv'")
    else:
        print("[!] No Instagram data collected. CSV not created.")
