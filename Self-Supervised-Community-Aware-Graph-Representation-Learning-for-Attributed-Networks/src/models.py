"""
Model definitions: a residual graph-attention encoder that produces
node embeddings and soft cluster assignments, and a bilinear
discriminator used for the Deep Graph Infomax pretraining objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ResidualGATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
        out_dim = out_channels * (heads if concat else 1)
        self.res_lin = nn.Linear(in_channels, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        res = self.res_lin(x)
        return self.act(self.bn(h + res))


class EnhancedGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim, n_clusters, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.blocks = nn.ModuleList()

        self.blocks.append(
            ResidualGATBlock(num_features, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
        )

        for _ in range(num_layers - 2):
            self.blocks.append(
                ResidualGATBlock(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
            )

        last_in = hidden_dim * num_heads
        self.blocks.append(
            ResidualGATBlock(last_in, embedding_dim, heads=1, concat=False, dropout=dropout)
        )

        self.classifier = nn.Linear(embedding_dim, n_clusters)
        self.cluster_linear = nn.Linear(embedding_dim, n_clusters)

    def forward(self, x, edge_index):
        h = x
        for block in self.blocks:
            h = block(h, edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)

        embeddings = h
        logits = self.classifier(embeddings)
        cluster_logits = self.cluster_linear(embeddings)
        q = F.softmax(cluster_logits, dim=1)
        return logits, embeddings, q


class DGIDiscriminator(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1)

    def forward(self, h, s):
        s_exp = s.unsqueeze(0).repeat(h.size(0), 1)
        logits = self.bilinear(h, s_exp).squeeze(1)
        return logits
