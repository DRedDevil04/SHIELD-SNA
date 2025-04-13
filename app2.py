import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import base64
import os
import plotly.graph_objects as go

# Import analysis modules
from analysis.content_analysis import analyse_content
from analysis.network_analysis import analyse_network
from analysis.sentiment_analysis import analyse_sentiment
from analysis.temporal_analysis import analyse_temporal

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Load default dataset once
default_path = './datasets/fetched_reddit_content_large.csv'
default_graph_path = './datasets/fetched_network_data.csv'

df = pd.read_csv(default_path)
df_graph = pd.read_csv(default_graph_path)

sentiment = analyse_sentiment(df)
content = analyse_content(df)
network_fig = analyse_network(df_graph)
temporal = analyse_temporal(df_graph)

subreddit_options = [{'label': sub, 'value': sub} for sub in df['subreddit'].dropna().unique()]
default_subreddit = df['subreddit'].dropna().unique()[0]

# Layout
app.layout = html.Div([
    html.H1("Hoax Campaign Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs([
        dcc.Tab(label="Sentiment Analysis", children=[
            html.Br(),
            html.H3("Sentiment Distribution by Subreddit"),
            dcc.Dropdown(id='subreddit-dropdown', options=subreddit_options, value=default_subreddit),
            dcc.Graph(id='sentiment-histogram'),
            html.Div(id='summary-box', style={'marginTop': 20}),
            html.Div(id='threat-info', children=[
                html.P(f"Threat keywords used: {', '.join(sentiment['threat_keywords'])}"),
                html.P(f"Threat-flagged hoax posts: {sentiment['threat_hoax_count']}")
            ], style={'marginTop': 20})
        ]),
        dcc.Tab(label="Author Network", children=[
            html.Br(),
            dcc.Graph(figure=network_fig)
        ]),
        dcc.Tab(label="Content Analysis", children=[
            html.Br(),
            html.H4("Classification Report"),
            html.Pre(children=content["classification_report"]),
            html.Img(src=f"data:image/png;base64,{content['confusion_matrix_base64']}", style={'width': '500px', 'marginBottom': '20px'}),
            html.H4("Top Hoax Words"),
            html.Ul([html.Li(word) for word in content["top_hoax_words"]]),
            html.H4("Top Real Words"),
            html.Ul([html.Li(word) for word in content["top_real_words"]])
        ]),
        dcc.Tab(label="Temporal Analysis", children=[
            html.Br(),
            html.Img(src=f"data:image/png;base64,{temporal['base64_plot']}", style={'width': '90%', 'marginBottom': '20px'}),
            html.H4("Top Subreddits (Hoax)"),
            html.Ul([html.Li(sub) for sub in temporal['top_subreddits']])
        ])
    ])
])

# Callback to update sentiment plot
@app.callback(
    Output('sentiment-histogram', 'figure'),
    Output('summary-box', 'children'),
    Input('subreddit-dropdown', 'value')
)
def update_sentiment_tab(selected_subreddit):
    df_filtered = sentiment["full_df"]
    filtered_df = df_filtered[df_filtered['subreddit'] == selected_subreddit]

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

# Run app
if __name__ == '__main__':
    app.run(debug=False)
