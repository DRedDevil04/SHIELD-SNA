import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import community as community_louvain

# Initialize app
app = dash.Dash(__name__)
server = app.server

# Download VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Load dataset
df = pd.read_csv("datasets/fetched_reddit_content_large.csv")
df['clean_title'] = df['clean_title'].fillna("")

# --- Sentiment Analysis ---
df['vader_sentiment_raw'] = df['clean_title'].apply(lambda x: sid.polarity_scores(x)['compound'])
df['vader_sentiment'] = df['vader_sentiment_raw'].apply(lambda x: round((x + 1) * 5, 2))

def label_sentiment(score):
    if score >= 7:
        return "positive"
    elif score <= 3:
        return "negative"
    else:
        return "neutral"

df['sentiment_category'] = df['vader_sentiment'].apply(label_sentiment)

# --- Network Graph Construction ---
df = df.dropna(subset=['author', 'id'])
post_author_map = df.set_index('id')['author'].to_dict()

G = nx.DiGraph()
for _, row in df.iterrows():
    commenter = row['author']
    parent_post = row.get('linked_submission_id')
    if pd.notnull(parent_post):
        parent_author = post_author_map.get(parent_post)
        if parent_author and parent_author != commenter:
            G.add_edge(commenter, parent_author)

partition = community_louvain.best_partition(G.to_undirected())
deg_cent = nx.degree_centrality(G)
btw_cent = nx.betweenness_centrality(G)
nx.set_node_attributes(G, partition, 'community')
nx.set_node_attributes(G, deg_cent, 'deg_centrality')
nx.set_node_attributes(G, btw_cent, 'btw_centrality')

top_users = sorted(deg_cent, key=deg_cent.get, reverse=True)[:50]
H = G.subgraph(top_users)
pos = nx.spring_layout(H, seed=42)

# Graph visual
edge_x, edge_y = [], []
for edge in H.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
node_x, node_y, node_color, node_text = [], [], [], []

for node in H.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_color.append(partition[node])
    node_text.append(f"{node}<br>Degree: {deg_cent[node]:.2f}<br>Betweenness: {btw_cent[node]:.2f}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    textposition="top center",
    text=[node for node in H.nodes()],
    hovertext=node_text,
    marker=dict(
        color=node_color,
        size=[deg_cent[node]*100 for node in H.nodes()],
        colorscale='Viridis',
        showscale=True,
        line_width=2
    )
)

network_fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
    title='Reddit Author Interaction Network',
    showlegend=False,
    hovermode='closest',
    margin=dict(t=30, l=10, b=10, r=10)
))

# --- Layout ---
app.layout = html.Div([
    html.H1("Hoax Campaign Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label="Sentiment Analysis", children=[
            html.Div([
                html.H3("Sentiment Distribution by Subreddit"),
                dcc.Dropdown(
                    id='subreddit-dropdown',
                    options=[{'label': sub, 'value': sub} for sub in df['subreddit'].unique()],
                    value=df['subreddit'].unique()[0],
                    placeholder="Select a subreddit"
                ),
                dcc.Graph(id='sentiment-histogram'),
                html.Div(id='summary-box', style={'marginTop': 20})
            ])
        ]),
        dcc.Tab(label="Author Network", children=[
            html.Br(),
            dcc.Graph(figure=network_fig),
            html.Div([
                html.H4("📦 Network Stats"),
                html.P(f"Nodes: {G.number_of_nodes()}"),
                html.P(f"Edges: {G.number_of_edges()}"),
                html.P(f"Communities Detected: {len(set(partition.values()))}")
            ], style={'padding': '10px'}),
            html.Div([
                html.H4("🔥 Top 5 Central Users"),
                html.Ul([
                    html.Li(f"{user} — Degree: {deg_cent[user]:.3f}, Betweenness: {btw_cent[user]:.3f}")
                    for user in top_users[:5]
                ])
            ], style={'padding': '10px'})
        ]),
        dcc.Tab(label="Content Analysis", children=[
            html.Br(),
            html.H3("Top Words by Hoax Label"),
            dcc.Dropdown(
                id='label-dropdown',
                options=[
                    {'label': label, 'value': label}
                    for label in df['2_way_label'].dropna().unique()
                ],
                value='hoax',
                placeholder="Select hoax label"
            ),
            dcc.Graph(id='content-bar')
        ])
    ])
])

# --- Callbacks ---
@app.callback(
    Output('sentiment-histogram', 'figure'),
    Output('summary-box', 'children'),
    Input('subreddit-dropdown', 'value')
)
def update_graph(selected_subreddit):
    filtered_df = df[df['subreddit'] == selected_subreddit]

    fig = px.histogram(
        filtered_df,
        x='vader_sentiment',
        color='2_way_label',
        barmode='overlay',
        nbins=30,
        title=f"Sentiment Distribution in r/{selected_subreddit}",
        labels={'vader_sentiment': 'VADER Sentiment Score (0 to 10)', '2_way_label': 'Hoax Label'}
    )

    sentiment_counts = filtered_df['sentiment_category'].value_counts().to_dict()
    summary = [html.P(f"{k.title()} posts: {v}") for k, v in sentiment_counts.items()]
    return fig, summary

@app.callback(
    Output('content-bar', 'figure'),
    Input('label-dropdown', 'value')
)
def update_content_analysis(selected_label):
    subset = df[df['2_way_label'] == selected_label]
    titles = subset['clean_title'].fillna("").tolist()

    # Check if there's any non-empty title left after filtering
    if not any(titles) or len(subset) == 0:
        return px.bar(title="No content available for this label.")

    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    try:
        X = vectorizer.fit_transform(titles)
    except ValueError:
        return px.bar(title="No significant words found (only stopwords).")

    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    words, freqs = zip(*sorted_words)

    label_display = str(selected_label).title()
    fig = px.bar(x=words, y=freqs, labels={'x': 'Keyword', 'y': 'Frequency'}, title=f"Top Words in {label_display} Posts")
    return fig

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)
