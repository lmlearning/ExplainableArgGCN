#!/usr/bin/env python
"""
Evaluate Refined-AFGCN and six PyG baselines on:
  • Accuracy, MCC
  • Spearman-ρ and Kendall-τ alignment with Categoriser ranking.
"""

import argparse
import os
import sys
import math
import numpy as np
import importlib.util
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import matthews_corrcoef

import torch
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.data import DataLoader, Data

# — our dataset + explainable model —
from train_refined_afgcn import AFGraphDataset
from train_consolidated import RefinedAFGCN

# — load the baseline definitions from the user’s file —
spec = importlib.util.spec_from_file_location("pyg_baselines", "./pyg_train.py")
pyg_baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pyg_baselines)  # type: ignore


# ╭────────────────────────  helper: 128-d feature fabrication  ────────────────────────╮
def make_inputs(x, dim=128):
    """Pad / crop node feature matrix to exactly `dim` columns."""
    if x.size(1) == dim:
        return x
    if x.size(1) > dim:
        return x[:, :dim]
    pad = torch.zeros(x.size(0), dim - x.size(1), device=x.device)
    return torch.cat([x, pad], 1)


# ╰──────────────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────  Gradient×Input importance (for non-attention baselines)  ─────────────╮
def grad_times_input(model, data):
    """Computes node importance using the Gradient × Input method."""
    x = make_inputs(data.x).clone().detach().requires_grad_(True)
    logits = model(Data(x=x, edge_index=data.edge_index))
    # broadcast to node-vector if model returns scalar
    logits = logits if logits.dim() else logits.expand(x.size(0))
    loss = logits.sum()
    loss.backward()
    imp = (x.grad * x).abs().sum(1).cpu()
    return imp


# ╰──────────────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────  GAT importance (from attention weights)  ───────────────────────╮
def gat_importance(model, data):
    """
    Computes node importance for a GAT model by retrieving attention
    weights from its first layer.
    """
    x_padded = make_inputs(data.x)
    # The GAT model's first layer must be a GATConv layer accessible as 'l1'.
    # We call its forward method with return_attention_weights=True.
    _, (edge_index, alpha) = model.l1(
        x_padded, data.edge_index, return_attention_weights=True
    )

    # alpha shape is [num_edges, num_heads]. Average scores across heads.
    alpha_scores = alpha.mean(dim=-1).abs()

    # Sum the attention scores for each target node.
    target_nodes = edge_index[1]
    imp = scatter_sum(alpha_scores, target_nodes, dim=0, dim_size=data.num_nodes)

    return imp.cpu()


# ╰──────────────────────────────────────────────────────────────────────────────────────╯


def edge_importance_first_layer(layer, edge_index, h):
    """
    Replicates the internal computation of RefinedAFGCNLayer to obtain
    per‑node importance scores  I_i  (see paper, Eq. (8)).
    Uses the *unnormalised* node features h that are fed into layer 0.
    Returns a tensor of shape (N, ).
    """
    struct_dim = layer.struct_dim
    device = h.device
    N = h.size(0)

    # Extract structural components
    s = h[:, :struct_dim]  # (N, struct_dim)
    s_feat = s[:, :-1]  # (N, struct_dim-1)

    # Split edge index
    H = layer.embed(h) if layer.embed is not None else h
    src, tgt = edge_index
    h_src, h_tgt = H[src], H[tgt]
    s_src_feat = s_feat[src]
    s_tgt_feat = s_feat[tgt]

    # -- α_ij  ----------------------------------------------------------
    W_att_h_src = layer.W_att(h_src)
    score_e = (h_tgt * W_att_h_src).sum(dim=-1) + (s_tgt_feat * s_src_feat).sum(
        dim=-1
    )
    alpha = scatter_softmax(score_e, tgt)  # (E,)

    # -- ρ_{ij} via defender pathway  -----------------------------------
    # reverse edges: defender → attacker
    rev_src = tgt
    rev_tgt = src

    # Corrected: Use the embedded tensor H instead of the raw input h
    h_rev_att = H[rev_tgt]
    h_rev_def = H[rev_src]

    pairing = (layer.W_p(h_rev_att) * layer.W_p_prime(h_rev_def)).sum(-1)
    beta = scatter_softmax(pairing, rev_tgt)
    psi = layer.f_def(h_rev_def).squeeze(-1)
    sum_term = scatter_sum(
        beta * torch.exp(-psi), rev_tgt, dim=0, dim_size=N
    )  # (N,)
    d = -torch.log(sum_term + 1e-8)
    rho = 1 / (1 + torch.exp(-layer.gamma * (d - layer.delta)))  # (N,)
    rho_edge = rho[src]  # (E,)

    # -- φ_att(h_src)  ---------------------------------------------------
    phi_src = layer.phi_att(h_src)  # (E, hidden_dim)

    # message  m_e  = α · (1-ρ) · φ
    contrib = alpha.unsqueeze(-1) * (1 - rho_edge).unsqueeze(-1) * phi_src
    # importance per *target* node = L1‑norm of message vector
    m_att = scatter_sum(contrib, tgt, dim=0, dim_size=N)  # (N, hidden_dim)
    importance = m_att.norm(p=1, dim=-1)  # (N,)

    return importance

def refined_importance(model, data):
    """
    Retrieves the dedicated ranking scores from the dual-head RefinedAFGCN model.
    """
    # The model call is wrapped in no_grad in the main evaluate function
    _, rank_scores = model(data)
    return rank_scores.cpu()

# ╭─────────────────────────────  model registry  ───────────────────────────────────────╮
def build_refined(ckpt, hid, layers, device):
    m = RefinedAFGCN(7, hid, layers).to(device)
    # Note: State dict is loaded later in the main loop
    m.eval()
    return m

#lambda m, d: edge_importance_first_layer(
#            m.layers[0],
#            d.edge_index[:, d.edge_index[0] != d.edge_index[1]],
#            m.input_norm(d.x),
#        ).cpu()

MODEL_ZOO = {
    "refined": (
        build_refined,refined_importance
        ,
    ),
    "afgcn": (lambda *_: pyg_baselines.AFGCN(128, 128, 1, 4).eval(), grad_times_input),
    "randalign": (
        lambda *_: pyg_baselines.RandAlignGCN(128, 128, 1, 3).eval(),
        grad_times_input,
    ),
    "gcn": (lambda *_: pyg_baselines.GCN(128, 128, 1).eval(), grad_times_input),
    "graphsage": (
        lambda *_: pyg_baselines.GraphSAGE(128, 128, 1).eval(),
        grad_times_input,
    ),
    "gin": (lambda *_: pyg_baselines.GIN(128, 128, 1).eval(), grad_times_input),
    "gat": (
        lambda *_: pyg_baselines.GAT(128, 64, 1, heads=4).eval(),
        gat_importance,  # <-- FIXED
    ),
}


# ╰──────────────────────────────────────────────────────────────────────────────────────╯


# ╭─────────────────────────────  evaluation core  ──────────────────────────────────────╮
def evaluate(model, loader, imp_fn, device, model_key):
    model = model.to(device).eval()
    tot, cor = 0, 0
    y_true, y_pred, spears, kends = [], [], [], []

    for data in loader:
        data = data.to(device)

        # --- MODIFIED: Handle different model outputs ---
        if model_key == "refined":
            # Refined model returns a (logits, scores) tuple and handles its own inputs
            output = model(data)
            logits, _ = output  # Unpack tuple, we only need classification logits
        else:
            # Baseline models need input padding and return a single tensor
            padded_data = data.clone()
            padded_data.x = make_inputs(padded_data.x)
            logits = model(padded_data)

        pred = (torch.sigmoid(logits) > 0.5).float()
        cor += (pred == data.y.float()).sum().item()
        tot += data.y.size(0)
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

        imp = imp_fn(model, data).detach().cpu().numpy()
        rk = data.rank.cpu().numpy()
        if imp.std() > 1e-6 and rk.std() > 1e-6:
            ρ, _ = spearmanr(rk, imp)
            τ, _ = kendalltau(rk, imp)
            if not math.isnan(ρ):
                spears.append(ρ)
            if not math.isnan(τ):
                kends.append(τ)

    acc = cor / tot if tot > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    # Handle cases where no valid correlations were found
    mean_spearman = float(np.mean(spears)) if spears else 0.0
    mean_kendall = float(np.mean(kends)) if kends else 0.0

    return acc, mcc, mean_spearman, mean_kendall


# ╰──────────────────────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("models", nargs="+", help="key=checkpoint.pth pairs")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AFGraphDataset(args.data_dir)
    loader = DataLoader(ds, 1, shuffle=False)

    print("model,accuracy,mcc,spearman,kendall")
    for kv in args.models:
        key, ckpt = kv.split("=")
        if key not in MODEL_ZOO:
            print(f"# unknown model '{key}' – skipped", file=sys.stderr)
            continue

        build, imp_fn = MODEL_ZOO[key]
        model = build(ckpt, args.hidden_dim, args.num_layers, device)

        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device)
            # Handle checkpoints with and without the 'model_state_dict' key
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

        # --- Pass the model 'key' to the evaluate function ---
        acc, mcc, sp, kt = evaluate(model, loader, imp_fn, device, model_key=key)
        print(f"{key},{acc:.4f},{mcc:.4f},{sp:.4f},{kt:.4f}")