import pandas as pd
import matplotlib.pyplot as plt

def plot_hoax_trends_by_subreddit(file_path, top_n=5):
    # Load data safely
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    df = df.dropna(subset=['created_utc'])

    # Filter hoax posts
    hoax_df = df[df['2_way_label'] == 1]

    # Identify top N subreddits by hoax post count
    top_subs = hoax_df['subreddit'].value_counts().head(top_n).index

    # Convert timestamp to month
    hoax_df['month'] = hoax_df['created_utc'].dt.to_period('M').astype(str)

    # Filter to only top N subreddits
    hoax_df = hoax_df[hoax_df['subreddit'].isin(top_subs)]

    # Group by subreddit and month
    grouped = hoax_df.groupby(['subreddit', 'month']).size().unstack(fill_value=0)

    # Plot
    plt.figure(figsize=(14, 7))
    for subreddit in grouped.index:
        plt.plot(grouped.columns, grouped.loc[subreddit], label=subreddit, marker='o')

    plt.title("Hoax Post Trends Over Time by Top Subreddits")
    plt.xlabel("Month")
    plt.ylabel("Number of Hoax Posts")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Subreddit")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_hoax_trends_by_subreddit("/home/nitu/Programs/sem6/SNA/SHIELD-SNA/datasets/fetched_reddit_content_large.csv")
