"""
Loss terms used during fine-tuning: a differentiable modularity
objective computed from the soft cluster-assignment matrix, the
DEC-style self-training target distribution, and an entropy
regularizer that discourages degenerate (overly uniform) assignments.
"""

import torch
import torch.nn as nn


class ModularityLoss(nn.Module):
    def __init__(self, adj_dense):
        super().__init__()
        self.register_buffer("A", adj_dense)
        k = torch.sum(self.A, dim=1)
        m = torch.sum(k) / 2.0
        self.register_buffer("k", k)
        self.register_buffer("m", m)
        kkT = torch.outer(self.k, self.k) / (2.0 * m + 1e-12)
        B = self.A - kkT
        self.register_buffer("B", B)

    def forward(self, P):
        PB = torch.matmul(self.B, P)
        M = torch.matmul(P.T, PB)
        Q = torch.trace(M) / (2.0 * self.m + 1e-12)
        return -Q, Q


def target_distribution(q):
    weight = (q ** 2) / (q.sum(dim=0, keepdim=True) + 1e-12)
    p = weight / (weight.sum(dim=1, keepdim=True) + 1e-12)
    return p


def entropy_regularizer(q):
    eps = 1e-12
    return -((q * torch.log(q + eps)).sum(dim=1)).mean()
