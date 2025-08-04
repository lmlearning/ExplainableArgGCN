#!/usr/bin/env python
"""
Evaluate Refined-AFGCN (with ablations) and six PyG baselines on:
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
# MODIFIED: Corrected import for DataLoader to resolve deprecation warning.
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# — Import dataset and the model (which now supports ablations) —
from train_refined_afgcn import AFGraphDataset
# Ensure this imports the RefinedAFGCN class from the new consolidated script.
from train_consolidated import RefinedAFGCN

# — load the baseline definitions from the user’s file —
spec = importlib.util.spec_from_file_location("pyg_baselines", "./pyg_train.py")
pyg_baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pyg_baselines)  # type: ignore


# ╭────────────────────────  helper: 128-d feature fabrication  ────────────────────────╮
def make_inputs(x, dim=128):
    """Pad / crop node feature matrix to exactly `dim` columns."""
    if x.size(1) == dim:
        return x
    if x.size(1) > dim:
        return x[:, :dim]
    pad = torch.zeros(x.size(0), dim - x.size(1), device=x.device)
    return torch.cat([x, pad], 1)


# ╰──────────────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────  Gradient×Input importance (for non-attention baselines)  ─────────────╮
def grad_times_input(model, data):
    """Computes node importance using the Gradient × Input method."""
    x = make_inputs(data.x).clone().detach().requires_grad_(True)
    # The data object passed to baselines must be a plain Data object
    logits = model(Data(x=x, edge_index=data.edge_index))
    # broadcast to node-vector if model returns scalar
    logits = logits if logits.dim() else logits.expand(x.size(0))
    loss = logits.sum()
    loss.backward()
    imp = (x.grad * x).abs().sum(1).cpu()
    return imp


# ╰──────────────────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────  GAT importance (from attention weights)  ───────────────────────╮
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


def refined_importance(model, data):
    """
    Retrieves the dedicated ranking scores from the dual-head RefinedAFGCN model.
    """
    # The model call is wrapped in no_grad in the main evaluate function
    _, rank_scores = model(data)
    return rank_scores.cpu()


# ╭─────────────────────────────  model registry  ───────────────────────────────────────╮
# --- MODIFIED: `build_refined` now accepts ablation flags ---
def build_refined(ckpt, hid, layers, device, cli_args):
    """
    Builds the RefinedAFGCN model, passing through any ablation flags
    from the command line to construct the correct model variant.
    """
    abl_flags = dict(
        disable_pairing=cli_args.no_pairing,
        disable_struct=cli_args.no_struct,
        disable_residual=cli_args.no_residual,
        disable_ln=cli_args.no_ln,
    )
    # The **abl_flags unpacks the dictionary into keyword arguments
    model = RefinedAFGCN(
        struct_dim=7, hidden=hid, layers=layers, **abl_flags
    ).to(device)
    model.eval()
    return model

# --- MODIFIED: Model builders now accept an 'args' parameter for consistency ---
MODEL_ZOO = {
    "refined": (
        build_refined,
        refined_importance,
    ),
    # The lambda functions now accept a fifth argument 'args' which they don't use.
    # This keeps the function signature consistent with `build_refined`.
    "afgcn": (
        lambda ckpt, hid, l, dev, args: pyg_baselines.AFGCN(128, hid, 1, l).eval(),
        grad_times_input
    ),
    "randalign": (
        lambda ckpt, hid, l, dev, args: pyg_baselines.RandAlignGCN(128, hid, 1, l).eval(),
        grad_times_input,
    ),
    "gcn": (
        lambda ckpt, hid, l, dev, args: pyg_baselines.GCN(128, hid, l).eval(),
        grad_times_input
    ),
    "graphsage": (
        lambda ckpt, hid, l, dev, args: pyg_baselines.GraphSAGE(128, hid, l).eval(),
        grad_times_input,
    ),
    "gin": (
        lambda ckpt, hid, l, dev, args: pyg_baselines.GIN(128, hid, l).eval(),
        grad_times_input
    ),
    "gat": (
        lambda ckpt, hid, l, dev, args: pyg_baselines.GAT(128, hid, l, heads=4).eval(),
        gat_importance,
    ),
}

# ╰──────────────────────────────────────────────────────────────────────────────────────╯


# ╭─────────────────────────────  evaluation core  ──────────────────────────────────────╮
def evaluate(model, loader, imp_fn, device, model_key):
    model = model.to(device).eval()
    tot, cor = 0, 0
    y_true, y_pred, spears, kends = [], [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            if model_key == "refined":
                output = model(data)
                logits, _ = output
            else:
                # Baseline models need input padding
                padded_data = Data(x=make_inputs(data.x), edge_index=data.edge_index)
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

    mean_spearman = float(np.mean(spears)) if spears else 0.0
    mean_kendall = float(np.mean(kends)) if kends else 0.0

    return acc, mcc, mean_spearman, mean_kendall

# ╰──────────────────────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate Refined-AFGCN (with ablations) and PyG baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data_dir", required=True, help="Directory containing the graph data.")
    p.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size.")
    p.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers.")

    # --- MODIFIED: Add arguments for RefinedAFGCN ablations ---
    p.add_argument(
        '--no_pairing', action='store_true',
        help="For RefinedAFGCN: use if checkpoint was trained with --no_pairing."
    )
    p.add_argument(
        '--no_struct', action='store_true',
        help="For RefinedAFGCN: use if checkpoint was trained with --no_struct."
    )
    p.add_argument(
        '--no_residual', action='store_true',
        help="For RefinedAFGCN: use if checkpoint was trained with --no_residual."
    )
    p.add_argument(
        '--no_ln', action='store_true',
        help="For RefinedAFGCN: use if checkpoint was trained with --no_ln."
    )
    # ---

    p.add_argument("models", nargs="+", help="key=checkpoint.pth pairs to evaluate.")
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
        
        # --- MODIFIED: Pass the full 'args' object to the model builder ---
        model = build(ckpt, args.hidden_dim, args.num_layers, device, args)

        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device)
            # Handle checkpoints with and without the 'model_state_dict' key
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
        else:
            print(f"# checkpoint not found for '{key}': {ckpt} -- skipped", file=sys.stderr)
            continue
            
        acc, mcc, sp, kt = evaluate(model, loader, imp_fn, device, model_key=key)
        print(f"{key},{acc:.4f},{mcc:.4f},{sp:.4f},{kt:.4f}")