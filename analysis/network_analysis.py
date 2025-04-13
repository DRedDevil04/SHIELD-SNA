import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import os

def run_network_analysis():
    # Load dataset
    df = pd.read_csv("/home/nitu/Programs/sem6/SNA/SHIELD-SNA/datasets/fetched_network_data.csv", sep=',')
    print(df['subreddit'].value_counts())
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
            if post_author and post_author != commenter:
                G.add_edge(commenter, post_author, relation='commented_on')

    print(f"Graph: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # --- 🔍 COMMUNITY DETECTION ---
    partition = community_louvain.best_partition(G_undirected)
    nx.set_node_attributes(G, partition, "community")
    num_communities = len(set(partition.values()))
    print(f"🧩 Detected {num_communities} communities")

    # --- 📊 CENTRALITY MEASURES ---
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    nx.set_node_attributes(G, degree_centrality, "degree_centrality")
    nx.set_node_attributes(G, betweenness_centrality, "betweenness_centrality")

    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    print("\n🔝 Top 5 by Degree Centrality:")
    for user, score in top_degree:
        print(f"{user}: {score:.4f}")

    print("\n🔝 Top 5 by Betweenness Centrality:")
    for user, score in top_betweenness:
        print(f"{user}: {score:.4f}")

    # --- Save outputs ---
    os.makedirs("../shared_data", exist_ok=True)

    # Save graph
    nx.write_gml(G, "../shared_data/network.gml")

    # Save centrality scores
    centrality_df = pd.DataFrame({
        "user": list(degree_centrality.keys()),
        "degree_centrality": list(degree_centrality.values()),
        "betweenness_centrality": list(betweenness_centrality.values()),
        "community": [partition.get(u) for u in degree_centrality.keys()]
    })
    centrality_df.to_csv("../shared_data/centrality_scores.csv", index=False)

    # Save community list separately
    community_df = pd.DataFrame(partition.items(), columns=["user", "community"])
    community_df.to_csv("../shared_data/communities.csv", index=False)

    print("✅ Network graph and analysis saved to ../shared_data/")

    # --- 🌐 Full Graph Community Visualization ---
    print("📊 Generating full community graph...")

    pos = nx.spring_layout(G, seed=42)  # Consistent layout across runs
    node_colors = [partition.get(node) for node in G.nodes()]

    plt.figure(figsize=(18, 14))
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           cmap=plt.cm.Set3,
                           node_size=60,
                           alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='red')

    plt.title(" Full Reddit Community Graph by Louvain Detection", fontsize=15)
    plt.axis("on")
    plt.tight_layout()

    plt.savefig("../shared_data/full_community_graph.png", dpi=300)
    plt.show()

# Run if called directly
if __name__ == "__main__":
    run_network_analysis()
