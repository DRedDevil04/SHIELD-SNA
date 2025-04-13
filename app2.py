import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate
import base64
import io
import os

# Import module functions
from analysis import sentiment_analysis, content_analysis, network_analysis, temporal_analysis

# Initialize app
app = dash.Dash(__name__)
server = app.server

# --- Layout ---
app.layout = html.Div([
    html.H1("Hoax Campaign Dashboard", style={'textAlign': 'center'}),

    # CSV Upload
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV'),
        multiple=False
    ),
    html.Div(id='file-content'),

    dcc.Tabs([
        dcc.Tab(label="Sentiment Analysis", children=[
            html.Div([
                html.H3("Sentiment Distribution by Subreddit"),
                dcc.Dropdown(id='subreddit-dropdown'),
                dcc.Graph(id='sentiment-histogram'),
                html.Div(id='summary-box', style={'marginTop': 20})
            ])
        ]),
        dcc.Tab(label="Author Network", children=[
            html.Br(),
            dcc.Graph(id='network-graph'),
        ]),
        dcc.Tab(label="Content Analysis", children=[
            html.Br(),
            dcc.Graph(id='content-bar')
        ]),
        dcc.Tab(label="Temporal Analysis", children=[
            html.Br(),
            dcc.Graph(id='temporal-analysis')
        ])
    ])
])

# --- CSV Upload Callback ---
@app.callback(
    Output('file-content', 'children'),
    Output('subreddit-dropdown', 'options'),
    Output('subreddit-dropdown', 'value'),
    Output('sentiment-histogram', 'figure'),
    Output('summary-box', 'children'),
    Output('network-graph', 'figure'),
    Output('content-bar', 'figure'),
    Output('temporal-analysis', 'figure'),
    Input('upload-data', 'contents')
)
def upload_file(contents):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Call the modules for processing
    df = sentiment_analysis(df)  # Updated df with sentiment columns
    network_fig = network_analysis(df)
    content_fig = content_analysis(df, 'hoax')  # Default to 'hoax' label
    temporal_fig = temporal_analysis(df)

    # Update dropdown options dynamically based on subreddits in the dataset
    subreddit_options = [{'label': sub, 'value': sub} for sub in df['subreddit'].unique()]
    default_subreddit = df['subreddit'].unique()[0]

    # Sentiment Histogram update
    fig, summary = update_sentiment_graph(df, default_subreddit)

    return "File uploaded successfully!", subreddit_options, default_subreddit, fig, summary, network_fig, content_fig, temporal_fig


def update_sentiment_graph(df, selected_subreddit):
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


def network_analysis(df):
    # Assuming your existing network analysis code
    G = network_analysis(df)  # function to create graph
    
    # Create a Plotly figure for network visualization
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='black')))
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='blue')))
    
    fig.update_layout(title="Reddit Author Network")
    return fig


def sentiment_analysis(df):
    # Perform the sentiment analysis steps as you already have
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob
    import nltk

    nltk.download('vader_lexicon')

    sid = SentimentIntensityAnalyzer()
    df['clean_title'] = df['clean_title'].fillna("")

    # Perform sentiment analysis
    df['vader_sentiment_raw'] = df['clean_title'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['vader_sentiment'] = df['vader_sentiment_raw'].apply(lambda x: round((x + 1) * 5, 2))
    df['textblob_polarity'] = df['clean_title'].apply(lambda x: TextBlob(x).sentiment.polarity)

    def label_sentiment(score):
        if score >= 7:
            return "positive"
        elif score <= 3:
            return "negative"
        else:
            return "neutral"

    df['sentiment_category'] = df['vader_sentiment'].apply(label_sentiment)
    return df


def content_analysis(df, label):
    # Placeholder content analysis (replace with your actual analysis)
    df_filtered = df[df['2_way_label'] == 1] if label == 'hoax' else df
    content_fig = go.Figure(data=[go.Bar(
        x=df_filtered['subreddit'].value_counts().index,
        y=df_filtered['subreddit'].value_counts().values,
        name='Hoax Posts'
    )])

    content_fig.update_layout(title=f"Content Analysis for {label} Posts")
    return content_fig


def temporal_analysis(df):
    # Plot temporal trends for hoax posts
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    df = df.dropna(subset=['created_utc'])

    # Filter hoax posts
    hoax_df = df[df['2_way_label'] == 1]

    # Identify top N subreddits by hoax post count
    top_subs = hoax_df['subreddit'].value_counts().head(5).index

    # Convert timestamp to month
    hoax_df['month'] = hoax_df['created_utc'].dt.to_period('M').astype(str)

    # Filter to only top N subreddits
    hoax_df = hoax_df[hoax_df['subreddit'].isin(top_subs)]

    # Group by subreddit and month
    grouped = hoax_df.groupby(['subreddit', 'month']).size().unstack(fill_value=0)

    # Plot
    fig = go.Figure()
    for subreddit in grouped.index:
        fig.add_trace(go.Scatter(x=grouped.columns, y=grouped.loc[subreddit], mode='lines+markers', name=subreddit))

    fig.update_layout(title="Hoax Post Trends Over Time by Top Subreddits")
    return fig


# Run app
if __name__ == '__main__':
    app.run(debug=True)
