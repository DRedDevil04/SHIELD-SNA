import requests
import json

def fetch_reddit_data(subreddit, query, size=100):
    url = f"https://api.pushshift.io/reddit/search/submission/?q={query}&subreddit={subreddit}&size={size}"
    response = requests.get(url)
    data = response.json()
    return data['data']

# Example usage:
if __name__ == "__main__":
    results = fetch_reddit_data("news", "hoax call", size=50)
    for post in results:
        print(post['title'])

