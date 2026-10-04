"""
Generic helper functions used across the pipeline: pretty-printing,
adjacency construction, parameter norms, and feature corruption for
the DGI contrastive objective.
"""

import math

import numpy as np
import torch


def print_box(title: str, lines):
    lines = [str(x) for x in lines]
    content_lengths = [len(line) for line in lines]
    width = max([len(title)] + content_lengths) + 4
    top = "+" + "-" * (width - 2) + "+"
    print(top)
    print("| " + title.ljust(width - 4) + " |")
    print(top)
    for line in lines:
        print("| " + line.ljust(width - 4) + " |")
    print(top)


def build_adj_dense(edge_index, num_nodes, device):
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    A[src, dst] = 1.0
    A = torch.maximum(A, A.T)
    return A


def compute_param_norm(model):
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().norm().item() ** 2)
    return float(math.sqrt(total))


def corrupt_features(x):
    idx = torch.randperm(x.size(0), device=x.device)
    return x[idx]


def flatten_metrics(out, prefix, metrics_dict):
    for k, v in metrics_dict.items():
        key = f"{prefix}_{k}"
        if isinstance(v, (np.floating, np.integer)):
            out[key] = float(v)
        elif isinstance(v, (float, int)):
            out[key] = v
        else:
            out[key] = v
