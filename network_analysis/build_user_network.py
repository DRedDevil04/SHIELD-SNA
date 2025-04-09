import networkx as nx

def build_network(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

# Example usage:
edges = [('user1', 'user2'), ('user2', 'user3')]
G = build_network(edges)
nx.draw(G, with_labels=True)

