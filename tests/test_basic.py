import random

from network.topology_generator import generate_random_graph
from network.validators import is_reachable


def test_simple_connectivity():
    g = generate_random_graph(num_nodes=20, p=0.2, seed=123)
    nodes = list(g.nodes.keys())
    s, d = nodes[0], nodes[-1]
    assert is_reachable(g, s, d)
