import pandas as pd
import matplotlib.pyplot as plt
import os

def run_temporal_analysis(time_window_days=1, show_plot=False):
    # Load Reddit data
    df = pd.read_csv("../datasets/fetched_reddit_content_large.csv")
    #df = pd.read_csv("../datasets/all_train.tsv", sep='\t')

    # Convert timestamp to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    df = df.dropna(subset=['created_utc'])  # remove rows with invalid timestamps

    # Load hoax call event data
    events_path = "../datasets/all_train.tsv"
    if not os.path.exists(events_path):
        print("❌ Hoax call events file not found.")
        return

    events = pd.read_csv(events_path, sep='\t')  # Updated to include sep='\t'
    events['timestamp'] = pd.to_datetime(events['timestamp'])

    # Define time window
    time_window = pd.Timedelta(days=time_window_days)

    # Correlate post volume with event window
    temporal_correlations = []

    for _, event in events.iterrows():
        start_time = event['timestamp'] - time_window
        end_time = event['timestamp'] + time_window
        mask = (df['created_utc'] >= start_time) & (df['created_utc'] <= end_time)
        relevant_posts = df[mask]
        hoax_posts = relevant_posts[relevant_posts['2_way_label'] == 1]

        temporal_correlations.append({
            'event_id': event['event_id'],
            'event_time': event['timestamp'],
            'total_posts': len(relevant_posts),
            'hoax_posts': len(hoax_posts),
        })

    temporal_df = pd.DataFrame(temporal_correlations)

    # Save to shared location
    os.makedirs("../shared_data", exist_ok=True)
    temporal_df.to_csv("../shared_data/temporal_correlation.csv", index=False)
    print("✅ Temporal correlation data saved to ../shared_data/temporal_correlation.csv")

    # Plotting (optional)
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.bar(temporal_df['event_time'], temporal_df['total_posts'], width=0.5, label="Total Posts")
        plt.bar(temporal_df['event_time'], temporal_df['hoax_posts'], width=0.3, label="Hoax Posts", color='red')
        plt.xlabel("Event Time")
        plt.ylabel("Number of Posts (±{} day)".format(time_window_days))
        plt.title("Post Volume Around Hoax Call Events")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Run independently
if __name__ == "__main__":
    run_temporal_analysis(time_window_days=1, show_plot=True)
