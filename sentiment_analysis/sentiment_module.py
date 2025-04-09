from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 to 1

# Classify into threat levels
def threat_level(score):
    if score < -0.5:
        return "High Threat"
    elif score < 0:
        return "Medium Threat"
    else:
        return "Low Threat"

