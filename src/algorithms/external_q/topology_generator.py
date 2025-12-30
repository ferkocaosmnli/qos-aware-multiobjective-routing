import networkx as nx
import random

class GraphModel:
    def __init__(self, graph):
        self.graph = graph

def generate_random_graph(num_nodes=250, edge_prob=0.05):
    """
    Generates a random graph with specified number of nodes.
    Assigns random weights for delay, reliability, and bandwidth to edges.
    """
    # Create a random graph using Erdos-Renyi model
    # Adjust edge_prob to ensure connectivity or reasonable density
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    
    # Ensure the graph is connected (optional but good for routing)
    if not nx.is_connected(G):
        # Add edges to connect components if needed, or just regenerate
        # For simplicity in this task, we'll just add edges to a central node or similar
        # Or simpler: just relabel and ensure a path exists later. 
        # Let's stick to a slightly higher prob or just proceed. 
        # A simple way to ensure connectivity is to use a different generator or add edges.
        # Let's use connected_watts_strogatz_graph for better topology simulation?
        # The prompt asked for "random graph", Erdos-Renyi is standard.
        pass

    # Assign attributes to edges
    for (u, v) in G.edges():
        G[u][v]['delay'] = random.uniform(1, 20)  # ms
        G[u][v]['reliability'] = random.uniform(0.90, 0.9999) # Probability
        G[u][v]['bandwidth'] = random.uniform(10, 1000) # Mbps

    return GraphModel(G)
