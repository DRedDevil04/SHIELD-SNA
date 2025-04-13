import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def analyse_temporal(df, top_n=5, return_plot=True):
    # Ensure timestamp conversion
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    df = df.dropna(subset=['created_utc'])

    # Filter hoax posts
    hoax_df = df[df['2_way_label'] == 1]

    # Identify top N subreddits by hoax post count
    top_subs = hoax_df['subreddit'].value_counts().head(top_n).index.tolist()

    # Convert timestamp to month
    hoax_df['month'] = hoax_df['created_utc'].dt.to_period('M').astype(str)

    # Filter to only top N subreddits
    hoax_df = hoax_df[hoax_df['subreddit'].isin(top_subs)]

    # Group by subreddit and month
    grouped = hoax_df.groupby(['subreddit', 'month']).size().unstack(fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    for subreddit in grouped.index:
        ax.plot(grouped.columns, grouped.loc[subreddit], label=subreddit, marker='o')

    ax.set_title("Hoax Post Trends Over Time by Top Subreddits")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Hoax Posts")
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Subreddit")
    plt.tight_layout()

    base64_plot = None
    if return_plot:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_plot = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

    return {
        "base64_plot": base64_plot,
        "top_subreddits": top_subs,
        "trend_data": grouped
    }

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("../datasets/fetched_network_data.csv", on_bad_lines="skip")
    result = analyse_temporal(df)
    print("Top subreddits:", result["top_subreddits"])
