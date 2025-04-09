from datetime import datetime

def correlate(posts, call_events, time_window=3600):
    correlated = []
    for post in posts:
        post_time = datetime.fromtimestamp(post['created_utc'])
        for event in call_events:
            event_time = datetime.strptime(event['time'], "%Y-%m-%d %H:%M:%S")
            if abs((post_time - event_time).total_seconds()) <= time_window:
                correlated.append((post, event))
    return correlated

