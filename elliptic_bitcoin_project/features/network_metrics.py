import numpy as np
import networkx as nx
import torch

def topological_features(edge_index: torch.Tensor, n: int) -> np.ndarray:
    """PageRank + clustering coefficient per node. Off-path is never called."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_index.t().tolist())
    pr = nx.pagerank(G, alpha=0.85, max_iter=200) if G.number_of_edges() else {i: 1.0/n for i in range(n)}
    cl = nx.clustering(G.to_undirected())
    feats = np.array([[pr.get(i, 0.0), cl.get(i, 0.0)] for i in range(n)], dtype=np.float32)
    assert feats.shape == (n, 2), f"Topology feat shape {feats.shape} != ({n}, 2)"
    return feats
