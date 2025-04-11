import networkx as nx
import pandas as pd

#Load the dataset

df = pd.read_csv("/home/nitu/Programs/sem6/SNA/SHIELD-SNA/datasets/fetched_reddit_content.csv")  # Replace with actual filename
print(df.head())
# Drop missing authors
df_network = df.dropna(subset=['author', 'linked_submission_id'])

# Group authors by submission
submission_groups = df_network.groupby('linked_submission_id')['author'].apply(list)

# Build the graph
G = nx.Graph()

for authors in submission_groups:
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            G.add_edge(authors[i], authors[j])

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Centrality
degree_centrality = nx.degree_centrality(G)
top_users = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top central users:", top_users)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.1)
nx.draw(G, pos, with_labels=False, node_size=30, edge_color='gray')
plt.title("User Interaction Network")
plt.show()
