"""
Evaluation utilities: builds an embedding-similarity graph, runs
Louvain community detection on it, and computes internal
community-quality metrics (modularity, coverage, conductance) plus
external metrics (NMI, ARI, F1, precision, recall, accuracy) via a
majority-vote mapping from communities to ground-truth labels.
"""

from collections import Counter

import networkx as nx
import numpy as np
import torch
from community import community_louvain
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.utils import flatten_metrics


def build_similarity_graph_from_embeddings(emb_np, edge_index_np):
    emb_norm = normalize(emb_np)
    G = nx.Graph()
    G.add_nodes_from(range(emb_np.shape[0]))
    src = edge_index_np[0]
    dst = edge_index_np[1]
    for i, j in zip(src, dst):
        w = float(np.dot(emb_norm[int(i)], emb_norm[int(j)]))
        if w < 0:
            w = 0.0
        ii, jj = int(i), int(j)
        if G.has_edge(ii, jj):
            if G[ii][jj]["weight"] < w:
                G[ii][jj]["weight"] = w
        else:
            G.add_edge(ii, jj, weight=w)
    return G


def compute_coverage_and_conductance(partition, G):
    total_edge_weight = 0.0
    intra_weight = 0.0

    for u, v, edata in G.edges(data=True):
        w = edata.get("weight", 1.0)
        total_edge_weight += w
        if partition.get(u) == partition.get(v):
            intra_weight += w

    coverage = intra_weight / (total_edge_weight + 1e-12)

    deg = dict(G.degree(weight="weight"))
    total_volume = sum(deg.values())

    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    conductances = []
    volumes = []

    for cid, nodes in communities.items():
        S = set(nodes)
        cut_w = 0.0
        vol_S = 0.0
        for u in S:
            vol_S += deg.get(u, 0.0)
            for v, edata in G[u].items():
                w = edata.get("weight", 1.0)
                if v not in S:
                    cut_w += w
        denom = min(vol_S, total_volume - vol_S)
        if denom <= 0:
            phi = 1.0
        else:
            phi = cut_w / denom
            phi = max(0.0, min(1.0, phi))
        conductances.append(phi)
        volumes.append(vol_S)

    if len(conductances) == 0:
        conductance_avg = np.nan
        conductance_weighted = np.nan
    else:
        conductance_avg = float(np.mean(conductances))
        vols = np.array(volumes)
        if np.sum(vols) <= 0:
            conductance_weighted = conductance_avg
        else:
            conductance_weighted = float(np.sum(np.array(conductances) * vols) / (np.sum(vols) + 1e-12))

    return coverage, conductance_avg, conductance_weighted


def majority_vote_preds(partition, labels_np):
    if labels_np is None:
        return None

    valid = labels_np >= 0
    if valid.sum() == 0:
        return None

    comm_to_labels = {}
    for i in range(len(labels_np)):
        cid = partition.get(i, None)
        if cid is None:
            continue
        if valid[i]:
            comm_to_labels.setdefault(cid, []).append(int(labels_np[i]))

    global_mode = Counter(labels_np[valid]).most_common(1)[0][0]
    comm_to_major = {}

    for cid in set(partition.values()):
        lbls = comm_to_labels.get(cid, [])
        if len(lbls) > 0:
            comm_to_major[cid] = Counter(lbls).most_common(1)[0][0]
        else:
            comm_to_major[cid] = int(global_mode)

    preds = np.zeros_like(labels_np, dtype=int)
    for i in range(len(labels_np)):
        cid = partition.get(i, None)
        if cid is None:
            preds[i] = int(global_mode)
        else:
            preds[i] = comm_to_major.get(cid, int(global_mode))
    return preds


def partition_metrics(partition, G):
    mm = {}
    try:
        mm["Modularity"] = community_louvain.modularity(partition, G)
    except Exception:
        mm["Modularity"] = np.nan
    cov, cond_avg, cond_w = compute_coverage_and_conductance(partition, G)
    mm["Coverage"] = cov
    mm["Conductance_avg"] = cond_avg
    mm["Conductance_weighted"] = cond_w
    return mm


def evaluate_embeddings_and_partitions(model, data, compute_label_metrics=True):
    """
    Runs Louvain on the embedding-similarity graph and reports its
    internal (modularity, coverage, conductance) and, if labels are
    available, external (NMI, ARI, F1, precision, recall, accuracy)
    metrics.
    """
    model.eval()
    with torch.no_grad():
        embeddings, q = model(data.x, data.edge_index)

    emb_np = embeddings.detach().cpu().numpy()
    edge_index_np = data.edge_index.detach().cpu().numpy()

    G_sim = build_similarity_graph_from_embeddings(emb_np, edge_index_np)

    if G_sim.number_of_edges() > 0:
        part_louvain_sim = community_louvain.best_partition(G_sim, weight="weight")
    else:
        part_louvain_sim = {i: 0 for i in range(data.num_nodes)}

    metrics = {
        "Louvain_similarity": partition_metrics(part_louvain_sim, G_sim),
        "embeddings": emb_np,
        "G_similarity": G_sim,
        "partitions": {
            "Louvain_similarity": part_louvain_sim,
        },
    }

    if compute_label_metrics and hasattr(data, "y"):
        labels_np = data.y.detach().cpu().numpy()

        def external_for_part(part):
            preds = majority_vote_preds(part, labels_np)
            if preds is None:
                return None
            out = {}
            try:
                out["NMI"] = normalized_mutual_info_score(labels_np, preds)
            except Exception:
                out["NMI"] = np.nan
            try:
                out["ARI"] = adjusted_rand_score(labels_np, preds)
            except Exception:
                out["ARI"] = np.nan
            try:
                out["F1_macro"] = f1_score(labels_np, preds, average="macro", zero_division=0)
            except Exception:
                out["F1_macro"] = np.nan
            try:
                out["Precision_macro"] = precision_score(labels_np, preds, average="macro", zero_division=0)
            except Exception:
                out["Precision_macro"] = np.nan
            try:
                out["Recall_macro"] = recall_score(labels_np, preds, average="macro", zero_division=0)
            except Exception:
                out["Recall_macro"] = np.nan
            try:
                out["Accuracy"] = accuracy_score(labels_np, preds)
            except Exception:
                out["Accuracy"] = np.nan
            return out

        metrics["external"] = {
            "Louvain_similarity": external_for_part(part_louvain_sim),
        }

    return metrics


def flatten_evaluation_to_row(eval_metrics, row):
    flatten_metrics(row, "Louvain_similarity", eval_metrics["Louvain_similarity"])

    if "external" in eval_metrics and eval_metrics["external"].get("Louvain_similarity") is not None:
        flatten_metrics(row, "Louvain_similarity_external", eval_metrics["external"]["Louvain_similarity"])

    return row
