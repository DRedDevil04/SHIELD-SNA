import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import os
import plotly.graph_objs as go

def analyse_network(df):
    df = df.dropna(subset=['author', 'id'])

    # Map post ID to author
    post_author_map = df.set_index('id')['author'].to_dict()

    # Create directed graph of commenter → post author
    G = nx.DiGraph()
    for _, row in df.iterrows():
        commenter = row['author']
        parent_post_id = row.get('linked_submission_id', None)

        if pd.notnull(parent_post_id):
            post_author = post_author_map.get(parent_post_id)
            if post_author is None:
                continue 
            if post_author and post_author != commenter:
                G.add_edge(commenter, post_author, relation='commented_on')

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # --- 🔍 COMMUNITY DETECTION ---
    partition = community_louvain.best_partition(G_undirected)
    nx.set_node_attributes(G, partition, "community")

    # --- 📊 CENTRALITY MEASURES ---
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    nx.set_node_attributes(G, degree_centrality, "degree_centrality")
    nx.set_node_attributes(G, betweenness_centrality, "betweenness_centrality")

    # Save data
    os.makedirs("./shared_data", exist_ok=True)

    nx.write_gml(G, "./shared_data/network.gml")

    centrality_df = pd.DataFrame({
        "user": list(degree_centrality.keys()),
        "degree_centrality": list(degree_centrality.values()),
        "betweenness_centrality": list(betweenness_centrality.values()),
        "community": [partition.get(u) for u in degree_centrality.keys()]
    })
    centrality_df.to_csv("./shared_data/centrality_scores.csv", index=False)

    community_df = pd.DataFrame(partition.items(), columns=["user", "community"])
    community_df.to_csv("./shared_data/communities.csv", index=False)

    # --- 🌐 Network Graph (simplified for Dash) ---
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_color = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(partition.get(node))
        node_text.append(f"{node}<br>Degree: {degree_centrality.get(node):.4f}<br>Betweenness: {betweenness_centrality.get(node):.4f}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Community',  # The colorbar title text
                    side='right'  # Corrected 'titleside' to 'side'
                ),
                xanchor='left'
            )
        )
    )


    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text='Reddit Author Network with Community Detection',  # Title text
                        font=dict(size=16)  # Font size for the title
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                ))



    return fig
