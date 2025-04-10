from data_collection.reddit_collector import fetch_reddit_data
from content_analysis.text_preprocessing import preprocess
from content_analysis.hoax_detection_model import is_hoax_related
from sentiment_analysis.sentiment_module import get_sentiment, threat_level

if __name__ == "__main__":
    data = fetch_reddit_data("news", "hoax call", size=100)
    for post in data:
        if is_hoax_related(post['title']):
            tokens = preprocess(post['title'])
            sentiment = get_sentiment(post['title'])
            print(f"{post['title']} => Sentiment: {sentiment}, Threat: {threat_level(sentiment)}")

