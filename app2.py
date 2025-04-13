import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from dash.exceptions import PreventUpdate
import base64
import io
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

# --- Layout ---
app.layout = html.Div([
    html.H1("Hoax Campaign Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Choose dataset source:"),
        dcc.RadioItems(
            id='dataset-choice',
            options=[
                {'label': 'Use default dataset', 'value': 'default'},
                {'label': 'Upload your own CSV', 'value': 'upload'}
            ],
            value='default',
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        )
    ]),

    html.Div(id='upload-container'),
    html.Div(id='file-content'),

    dcc.Tabs([
        dcc.Tab(label="Sentiment Analysis", children=[
            html.Div([
                html.H3("Sentiment Distribution by Subreddit"),
                dcc.Dropdown(id='subreddit-dropdown'),
                dcc.Graph(id='sentiment-histogram'),
                html.Div(id='summary-box', style={'marginTop': 20}),
                html.Div(id='threat-info', style={'marginTop': 20})
            ])
        ]),
        dcc.Tab(label="Author Network", children=[
            html.Br(),
            dcc.Graph(id='network-graph'),
        ]),
        dcc.Tab(label="Content Analysis", children=[
            html.Br(),
            html.H4("Classification Report"),
            html.Pre(id='classification-report'),
            html.Img(id='confusion-matrix-img', style={'width': '500px', 'marginBottom': '20px'}),
            html.H4("Top Hoax Words"),
            html.Ul(id='top-hoax-words'),
            html.H4("Top Real Words"),
            html.Ul(id='top-real-words')
        ]),
        dcc.Tab(label="Temporal Analysis", children=[
            html.Br(),
            html.Img(id='temporal-plot-img', style={'width': '90%', 'marginBottom': '20px'}),
            html.H4("Top Subreddits (Hoax)"),
            html.Ul(id='top-subreddits')
        ])
    ])
])

# Show upload button conditionally
@app.callback(
    Output('upload-container', 'children'),
    Input('dataset-choice', 'value')
)
def toggle_upload_area(choice):
    if choice == 'upload':
        return dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV'),
            multiple=False
        )
    return html.Div()

# Main callback
@app.callback(
    Output('file-content', 'children'),
    Output('subreddit-dropdown', 'options'),
    Output('subreddit-dropdown', 'value'),
    Output('sentiment-histogram', 'figure'),
    Output('summary-box', 'children'),
    Output('threat-info', 'children'),
    Output('network-graph', 'figure'),
    Output('classification-report', 'children'),
    Output('confusion-matrix-img', 'src'),
    Output('top-hoax-words', 'children'),
    Output('top-real-words', 'children'),
    Output('temporal-plot-img', 'src'),
    Output('top-subreddits', 'children'),
    Input('dataset-choice', 'value'),
    Input('upload-data', 'contents')
)
def load_data(choice, contents):
    if choice == 'upload':
        if contents is None:
            raise PreventUpdate
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        status_msg = "Custom file uploaded successfully!"
    else:
        default_path = './datasets/fetched_reddit_content_large.csv'
        if not os.path.exists(default_path):
            return "Default dataset not found.", [], None, go.Figure(), "", "", go.Figure(), "", None, [], [], None, []
        df = pd.read_csv(default_path)
        status_msg = "Using default dataset."

    # Run modules
    sentiment = analyse_sentiment(df)
    content = analyse_content(df)
    # network_fig = analyse_network(df)
    # temporal = analyse_temporal(df)

    # Subreddit dropdown
    subreddit_options = [{'label': sub, 'value': sub} for sub in df['subreddit'].dropna().unique()]
    default_subreddit = df['subreddit'].dropna().unique()[0]

    # Sentiment histogram and summary
    fig, summary = update_sentiment_graph(sentiment["full_df"], default_subreddit)

    # Threat info
    threat_info = html.Div([
        html.P(f"Threat keywords used: {', '.join(sentiment['threat_keywords'])}"),
        html.P(f"Threat-flagged hoax posts: {sentiment['threat_hoax_count']}")
    ])

    # Content analysis
    top_hoax_words_list = [html.Li(word) for word in content["top_hoax_words"]]
    top_real_words_list = [html.Li(word) for word in content["top_real_words"]]

    # Temporal analysis
    temporal_plot_src = f"data:image/png;base64,{temporal['base64_plot']}"
    top_subreddits_list = [html.Li(sub) for sub in temporal['top_subreddits']]

    confusion_img_src = f"data:image/png;base64,{content['confusion_matrix_base64']}"

    return (
        status_msg,
        subreddit_options,
        default_subreddit,
        fig,
        summary,
        threat_info,
        network_fig,
        content["classification_report"],
        confusion_img_src,
        top_hoax_words_list,
        top_real_words_list,
        temporal_plot_src,
        top_subreddits_list
    )

# Update sentiment graph
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

# Run app
if __name__ == '__main__':
    app.run(debug=True)
