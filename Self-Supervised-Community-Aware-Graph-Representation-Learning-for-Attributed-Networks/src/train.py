"""
Training routines for the pipeline: unsupervised DGI pretraining of
the encoder, followed by fine-tuning with a combination of a
soft-assignment KL term, a differentiable modularity objective, and an
entropy regularizer. Model selection during fine-tuning uses Louvain
modularity on the topology graph as a label-free validation signal.
"""

import copy
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from community import community_louvain

from src.losses import ModularityLoss, entropy_regularizer, target_distribution
from src.models import DGIDiscriminator
from src.utils import compute_param_norm, corrupt_features


def pretrain_dgi(encoder_model, data, device, epochs=100, lr=0.001, weight_decay=5e-4, verbose=True):
    encoder_model = encoder_model.to(device)
    with torch.no_grad():
        emb_test, _ = encoder_model(data.x, data.edge_index)
        embedding_dim = emb_test.shape[1]

    dgi_disc = DGIDiscriminator(embedding_dim).to(device)
    optimizer = optim.AdamW(
        list(encoder_model.parameters()) + list(dgi_disc.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    bce = nn.BCEWithLogitsLoss()

    history = {"epoch": [], "loss": [], "param_norm": []}

    for epoch in range(1, epochs + 1):
        encoder_model.train()
        optimizer.zero_grad()

        h_pos, _ = encoder_model(data.x, data.edge_index)

        x_corrupt = corrupt_features(data.x)
        x_tmp = x_corrupt
        for block in encoder_model.blocks:
            x_tmp = block(x_tmp, data.edge_index)
            x_tmp = F.dropout(x_tmp, p=encoder_model.dropout, training=encoder_model.training)
        h_neg = x_tmp

        s = torch.sigmoid(h_pos.mean(dim=0))
        pos_logits = dgi_disc(h_pos, s)
        neg_logits = dgi_disc(h_neg, s)

        loss = bce(pos_logits, torch.ones_like(pos_logits)) + bce(neg_logits, torch.zeros_like(neg_logits))
        loss.backward()
        optimizer.step()

        pn = compute_param_norm(encoder_model)
        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))
        history["param_norm"].append(float(pn))

        if verbose:
            print(f"(DGI) Epoch {epoch:03d} | loss: {loss.item():.4f} | param_norm: {pn:.4f}")

    return encoder_model, history


def finetune_unsupervised(
    model,
    data,
    topology_graph,
    adj_dense,
    device,
    w_dgi=0.1,
    w_kl=0.5,
    w_mod=1.0,
    w_ent=0.01,
    lr=0.005,
    weight_decay=5e-4,
    epochs=200,
    patience=60,
    verbose=True,
    use_dgi_term=True,
):
    model = model.to(device)
    mod_fn = ModularityLoss(adj_dense.to(device)).to(device)

    model.eval()
    with torch.no_grad():
        emb_test, _ = model(data.x, data.edge_index)
        embedding_dim = emb_test.shape[1]

    dgi_disc = DGIDiscriminator(embedding_dim).to(device) if use_dgi_term else None
    opt_params = list(model.parameters()) + (list(dgi_disc.parameters()) if dgi_disc is not None else [])
    optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_state = None
    best_score = -float("inf")
    noimp = 0

    history = {
        "epoch": [],
        "total_loss": [],
        "kl": [],
        "mod": [],
        "dgi": [],
        "ent": [],
        "val_mod": [],
        "param_norm": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        embeddings, q = model(data.x, data.edge_index)

        with torch.no_grad():
            p = target_distribution(q.detach())

        eps = 1e-12
        kl_loss = F.kl_div(torch.log(q.clamp(min=eps)), p, reduction="batchmean")
        mod_loss, mod_val = mod_fn(q)

        if use_dgi_term and dgi_disc is not None:
            x_corrupt = corrupt_features(data.x)
            x_tmp = x_corrupt
            for block in model.blocks:
                x_tmp = block(x_tmp, data.edge_index)
                x_tmp = F.dropout(x_tmp, p=model.dropout, training=model.training)
            h_neg = x_tmp

            s = torch.sigmoid(embeddings.mean(dim=0))
            pos_logits = dgi_disc(embeddings, s)
            neg_logits = dgi_disc(h_neg, s)
            dgi_loss = bce(pos_logits, torch.ones_like(pos_logits)) + bce(neg_logits, torch.zeros_like(neg_logits))
        else:
            dgi_loss = torch.tensor(0.0, device=device)

        ent = entropy_regularizer(q)
        total_loss = w_dgi * dgi_loss + w_kl * kl_loss + w_mod * mod_loss + w_ent * ent

        total_loss.backward()
        optimizer.step()

        # Validation surrogate: modularity of the hard assignment on the topology graph.
        model.eval()
        with torch.no_grad():
            emb_val, q_val = model(data.x, data.edge_index)
            hard_assign = q_val.argmax(dim=1).cpu().numpy()
            part = {i: int(hard_assign[i]) for i in range(data.num_nodes)}
            try:
                val_mod = community_louvain.modularity(part, topology_graph)
            except Exception:
                val_mod = -1.0

        pn = compute_param_norm(model)

        history["epoch"].append(epoch)
        history["total_loss"].append(float(total_loss.item()))
        history["kl"].append(float(kl_loss.item()))
        history["mod"].append(float(mod_val))
        history["dgi"].append(float(dgi_loss.item() if isinstance(dgi_loss, torch.Tensor) else dgi_loss))
        history["ent"].append(float(ent.item() if isinstance(ent, torch.Tensor) else ent))
        history["val_mod"].append(float(val_mod))
        history["param_norm"].append(float(pn))

        score = float(val_mod)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            noimp = 0
        else:
            noimp += 1
            if noimp >= patience:
                if verbose:
                    print(f"Finetune early stop at epoch {epoch} | best score {best_score:.4f}")
                break

        if verbose:
            dgi_val = float(dgi_loss.item() if isinstance(dgi_loss, torch.Tensor) else dgi_loss)
            print(
                f"Finetune epoch {epoch:03d} | total_loss: {total_loss.item():.4f} | "
                f"kl: {kl_loss.item():.4f} | mod: {mod_val:.4f} | dgi: {dgi_val:.4f} | "
                f"val_mod: {val_mod:.4f} | param_norm: {pn:.4f}"
            )

        if epoch % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
