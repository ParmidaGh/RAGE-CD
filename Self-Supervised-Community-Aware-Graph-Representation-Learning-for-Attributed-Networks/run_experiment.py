"""
Self-Supervised Graph Representation Learning with Community-Aware
Optimization for Attributed Networks Community Detection

Driver script: runs the full pretrain -> fine-tune -> evaluate pipeline
on Citeseer for `config.NUM_RUNS` independent runs, then prints the
per-run Louvain-on-similarity-graph metrics and the final results.
"""

import gc
import time

import pandas as pd
import torch

from src import config
from src.data import load_dataset
from src.evaluate import evaluate_embeddings_and_partitions, flatten_evaluation_to_row
from src.models import EnhancedGNN
from src.train import finetune_unsupervised, pretrain_dgi
from src.utils import compute_param_norm, print_box

print("Using device:", config.DEVICE)


def run_single_experiment(run_id, data, topology_graph, adj_dense, num_features, num_classes):
    print(f"\n========== RUN {run_id + 1}/{config.NUM_RUNS} ==========")

    model = EnhancedGNN(
        num_features=num_features,
        hidden_dim=config.HIDDEN_DIM,
        embedding_dim=config.EMBEDDING_DIM,
        n_clusters=num_classes,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    t0 = time.time()
    model, pretrain_hist = pretrain_dgi(
        model,
        data,
        config.DEVICE,
        epochs=config.DGI_PRETRAIN_EPOCHS,
        lr=config.LR_PRETRAIN,
        weight_decay=config.WEIGHT_DECAY,
        verbose=True,
    )
    t_pre = time.time() - t0
    print(f"Pretrain finished in {t_pre:.2f}s")

    t1 = time.time()
    model, finetune_hist = finetune_unsupervised(
        model,
        data,
        topology_graph,
        adj_dense,
        config.DEVICE,
        w_dgi=config.W_DGI,
        w_kl=config.W_KL,
        w_mod=config.W_MOD,
        w_ent=config.W_ENT,
        lr=config.LR_FINETUNE,
        weight_decay=config.WEIGHT_DECAY,
        epochs=config.FINETUNE_EPOCHS,
        patience=config.PATIENCE,
        verbose=True,
        use_dgi_term=(config.W_DGI > 0),
    )
    t_fine = time.time() - t1
    print(f"Finetune finished in {t_fine:.2f}s")

    eval_metrics = evaluate_embeddings_and_partitions(
        model,
        data,
        compute_label_metrics=True,
    )

    row = {
        "run_id": run_id + 1,
        "w_dgi": config.W_DGI,
        "w_kl": config.W_KL,
        "w_mod": config.W_MOD,
        "w_ent": config.W_ENT,
        "pretrain_time_s": t_pre,
        "finetune_time_s": t_fine,
        "param_norm": compute_param_norm(model),
    }
    row = flatten_evaluation_to_row(eval_metrics, row)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row, eval_metrics


def main():
    data, topology_graph, adj_dense, num_features, num_classes = load_dataset()

    all_rows = []
    all_eval_metrics = []

    for run_id in range(config.NUM_RUNS):
        row, eval_metrics = run_single_experiment(
            run_id=run_id,
            data=data,
            topology_graph=topology_graph,
            adj_dense=adj_dense,
            num_features=num_features,
            num_classes=num_classes,
        )
        all_rows.append(row)
        all_eval_metrics.append(eval_metrics)

    df = pd.DataFrame(all_rows)

    metric_cols = [c for c in df.columns if c.startswith("Louvain_similarity")]
    display_df = df[metric_cols].rename(
        columns=lambda c: c.replace("Louvain_similarity_external_", "").replace("Louvain_similarity_", "")
    )

    print("\n================== PER-RUN RESULTS ==================")
    print(display_df.to_string(index=False))

    print("\n")
    max_series = display_df.max(numeric_only=True)
    max_lines = [f"{name}: {val:.6f}" for name, val in max_series.items()]
    print_box("Final Results", max_lines)

    print("\n================== DONE ==================")

    return df, all_eval_metrics


if __name__ == "__main__":
    main()
