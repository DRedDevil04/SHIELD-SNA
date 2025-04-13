import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

def build_network_graph(df_stage):
    df_stage = df_stage.dropna(subset=['author', 'id'])
    post_author_map = df_stage.set_index('id')['author'].to_dict()

    G = nx.DiGraph()
    for _, row in df_stage.iterrows():
        commenter = row['author']
        parent_post_id = row.get('linked_submission_id', None)

        if pd.notnull(parent_post_id):
            post_author = post_author_map.get(parent_post_id)
            if post_author and post_author != commenter:
                G.add_edge(commenter, post_author)

    return G

def draw_plotly_graph(G, title):
    # Community detection using modularity-based communities
    communities = list(greedy_modularity_communities(G))
    community_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_map[node] = idx
    
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text = [], [], []
    degree_centrality = nx.degree_centrality(G)

    # Assign colors to nodes based on their community
    node_colors = [community_map.get(node, 0) for node in G.nodes()]
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Degree Centrality: {degree_centrality.get(node, 0):.4f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=10,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Community Group")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            title_x=0.5,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )

    return fig

def analyse_network_temporal(df):
    df = df.dropna(subset=['created_utc', 'author', 'id'])
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

    # Sort and split into 4 equal stages
    df_sorted = df.sort_values('created_utc')
    stage_dfs = np.array_split(df_sorted, 4)

    figures = []
    for i, stage_df in enumerate(stage_dfs, 1):
        G = build_network_graph(stage_df)
        fig = draw_plotly_graph(G, title=f"Stage {i}: Network Evolution")
        figures.append(fig)

    return figures
