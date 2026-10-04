"""
Dataset loading utilities: fetches Citeseer via PyTorch Geometric,
builds the dense adjacency matrix used by the modularity loss, and
builds the NetworkX topology graph used for Louvain partitioning.
"""

import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

from src import config
from src.utils import build_adj_dense


def load_dataset():
    """
    Loads the Citeseer dataset and returns everything the training and
    evaluation pipeline needs: the PyG data object, the number of node
    features, the dense adjacency matrix, and the NetworkX
    topology graph.
    """
    dataset = Planetoid(root=config.DATA_ROOT, name=config.DATASET_NAME)
    data = dataset[0].to(config.DEVICE)
    data.edge_index = to_undirected(data.edge_index)

    num_nodes = data.num_nodes
    num_features = dataset.num_node_features
    num_classes = int(dataset.num_classes)

    print(f"Dataset loaded: num_nodes={num_nodes}, num_features={num_features}, num_classes={num_classes}")

    adj_dense = build_adj_dense(data.edge_index, num_nodes, config.DEVICE)

    topology_graph = nx.Graph()
    topology_graph.add_nodes_from(range(num_nodes))
    topology_graph.add_edges_from(data.edge_index.T.detach().cpu().numpy().tolist())

    return data, topology_graph, adj_dense, num_features
