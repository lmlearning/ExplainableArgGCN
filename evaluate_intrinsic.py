#!/usr/bin/env python
# evaluate_intrinsic.py  (full visual export: JSON + DOT)
# --------------------------------------------------------------------------
import argparse, json, math, numpy as np, torch, pathlib
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
from torch_geometric.loader import DataLoader
from torch_geometric.data   import Data
from torch_scatter import scatter_sum

from train_refined_afgcn import AFGraphDataset
from train_consolidated   import RefinedAFGCN

# ---------------------- influence + credit extractor -----------------------
@torch.no_grad()
def intrinsic_with_credit(model, data):
    struct = model.struct_dim
    h  = model.input_norm(data.x)
    sf = h[:, :struct]
    ei = data.edge_index
    src, tgt = ei
    N, E = data.num_nodes, ei.size(1)

    edge_I = torch.zeros(E, device=h.device)
    credit = {}

    for idx, layer in enumerate(model.layers):
        layer_in = h if idx == 0 else torch.cat([sf, h], 1)
        H = layer.embed(layer_in) if layer.embed else layer_in
        h_src, h_tgt = H[src], H[tgt]
        s_feat = sf[:, :-1]
        s_src, s_tgt = s_feat[src], s_feat[tgt]

        score = (h_tgt * layer.W_att(h_src)).sum(-1) + (s_src * s_tgt).sum(-1)
        score = torch.clamp(score, -10, 10)
        alpha_num = torch.exp(score)
        alpha_den = scatter_sum(alpha_num, tgt, dim=0, dim_size=N)
        α = alpha_num / (alpha_den[tgt] + 1e-9)

        if not getattr(layer, 'disable_pairing', False):
            rev_src, rev_tgt = tgt, src
            h_att, h_def = H[rev_tgt], H[rev_src]
            pair = (layer.W_p(h_att) * layer.W_p_prime(h_def)).sum(-1)
            pair = torch.clamp(pair, -10, 10)
            beta_num = torch.exp(pair)
            beta_den = scatter_sum(beta_num, rev_tgt, dim=0, dim_size=N)
            β = beta_num / (beta_den[rev_tgt] + 1e-9)
            ψ = layer.f_def(h_def).squeeze(-1)

            clamped_psi = torch.clamp(ψ, -10, 10)
            credit_term = β * torch.exp(-clamped_psi)

            sum_term = scatter_sum(credit_term, rev_tgt, dim=0, dim_size=N)
            d = -torch.log(sum_term + 1e-9)
            ρ = 1 / (1 + torch.exp(-layer.gamma * (d - layer.delta)))
            ρ_edge = ρ[src]

            for j, k, val in zip(rev_tgt.tolist(), rev_src.tolist(), credit_term.tolist()):
                key = (j, k); credit[key] = credit.get(key, 0) + val
        else:
            ρ_edge = 0.0

        val = layer.phi_att(H[src]).abs().sum(-1)
        edge_I += α * (1 - ρ_edge) * val
        h = layer(ei, layer_in)

    node_in = scatter_sum(edge_I, tgt, dim=0, dim_size=N)
    return edge_I.cpu(), node_in.cpu(), credit
# --------------------------------------------------------------------------

# ───────────────────────── Faithfulness Metrics ─────────────────────────────
def auc_deletion(model, data, edge_I, steps=(0, 5, 10, 20, 30, 40, 50)):
    steps = [s / 100 for s in steps]
    conf = []
    ei = data.edge_index.clone()
    _, idx = edge_I.sort(descending=True)

    with torch.no_grad():
        base_logits, _ = model(data)
        conf.append(torch.sigmoid(base_logits).mean().item())
        for frac in steps[1:]:
            k = int(frac * ei.size(1))
            new_ei = ei[:, idx[k:]]
            out_logits, _ = model(Data(x=data.x, edge_index=new_ei, y=data.y))
            conf.append(torch.sigmoid(out_logits).mean().item())
    return np.trapz(conf, steps)

def auc_insertion(model, data, edge_I, steps=(0, 5, 10, 20, 30, 40, 50)):
    steps = [s / 100 for s in steps]
    conf = []
    ei = data.edge_index.clone()
    _, idx = edge_I.sort(descending=True)

    with torch.no_grad():
        empty_data = Data(x=data.x, edge_index=data.edge_index[:, :0], y=data.y)
        empty_logits, _ = model(empty_data)
        conf.append(torch.sigmoid(empty_logits).mean().item())
        for frac in steps[1:]:
            k = int(frac * ei.size(1))
            new_ei = ei[:, idx[:k]]
            out_logits, _ = model(Data(x=data.x, edge_index=new_ei, y=data.y))
            conf.append(torch.sigmoid(out_logits).mean().item())
    return np.trapz(conf, steps)

def counterfactual_flip(model, data, edge_I):
    src, tgt = data.edge_index
    top = {}
    for e, (s, t, val) in enumerate(zip(src.tolist(), tgt.tolist(), edge_I.tolist())):
        if val > top.get(t, (0, 0.0))[1]:
            top[t] = (e, val)
    if not top: return 0.0

    mask = torch.ones(edge_I.size(0), dtype=torch.bool, device=edge_I.device)
    for e, _ in top.values():
        mask[e] = False

    with torch.no_grad():
        pr0_logits, _ = model(data)
        pr0 = (torch.sigmoid(pr0_logits) > 0.5).float()
        pr1_logits, _ = model(Data(x=data.x, edge_index=data.edge_index[:, mask], y=data.y))
        pr1 = (torch.sigmoid(pr1_logits) > 0.5).float()
    flips = (pr0 != pr1).sum().item()
    return flips / pr0.size(0) if pr0.numel() > 0 else 0.0

def gini(x):
    if x.sum() == 0: return 0.0
    y = x.cpu().numpy()
    y = np.sort(y)
    n = len(y); cum = np.cumsum(y)
    return (n + 1 - 2 * (cum / y.sum()).sum()) / n if n > 0 else 0.0
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--export_json', help="Path to export results as a JSON file.")
    ap.add_argument('--export_dot', help="Directory to export graphs in DOT format.")
    ap.add_argument('--top_k', nargs=2, type=int, metavar=('ATT', 'DEF'), default=[5, 3],
                    help='Top-k GLOBAL attacker edges and top-k defenders per attacker.')
    args = ap.parse_args()
    topA, topD = args.top_k

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = AFGraphDataset(args.data_dir)
    ld = DataLoader(ds, 1, shuffle=False)
    model = RefinedAFGCN(7, args.hidden_dim, args.layers).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    stats = dict(spearman=[], kendall=[], gini=[], auc_del=[], auc_ins=[], cf=[])
    vis_json = []
    dot_dir = None
    if args.export_dot:
        dot_dir = pathlib.Path(args.export_dot)
        dot_dir.mkdir(parents=True, exist_ok=True)

    for g, data in enumerate(tqdm(ld, desc="Evaluating graphs")):
        data = data.to(device)
        edge_I, node_in, credit = intrinsic_with_credit(model, data)

        # (Metric calculations remain the same)
        if node_in.numel() > 1 and data.rank.numel() > 1 and node_in.std() > 1e-6 and data.rank.std() > 1e-6:
            ρ, _ = spearmanr(node_in.numpy(), data.rank.cpu().numpy())
            stats['spearman'].append(ρ)
            τ, _ = kendalltau(node_in.numpy(), data.rank.cpu().numpy(), variant='b')
            stats['kendall'].append(τ)
        stats['gini'].append(gini(edge_I))
        stats['auc_del'].append(auc_deletion(model, data, edge_I))
        stats['auc_ins'].append(auc_insertion(model, data, edge_I))
        stats['cf'].append(counterfactual_flip(model, data, edge_I))

        # -------- build visual structures ----------------------------------
        if args.export_json or dot_dir:
            N = data.num_nodes
            src, tgt = data.edge_index

            # --- MODIFICATION: Select top k GLOBAL attacker edges ---
            # 1. Get indices of the top 'topA' most influential edges overall.
            idx = edge_I.argsort(descending=True)
            keep_edges = idx[:topA].tolist()

            # 2. defender credits: retain only if their attacker's edge was kept
            credit_trim = {}
            kept_attackers = {int(src[e]) for e in keep_edges}
            for (j, k), val in credit.items():
                if j in kept_attackers:
                    lst = credit_trim.setdefault(j, [])
                    lst.append((k, val))

            # 3. keep topD defenders for each of those attackers
            defender_edges = []
            max_cred = 1e-9
            for j, lst in credit_trim.items():
                lst_sorted = sorted(lst, key=lambda x: -x[1])[:topD]
                for k, v in lst_sorted:
                    defender_edges.append({"att": j, "def": k, "cred": float(v)})
                    if v > max_cred: max_cred = v

            # JSON export (remains unchanged)
            if args.export_json:
                nodes = [{"id": int(i), "imp": float(node_in[i]), "rank": float(data.rank[i])} for i in range(N)]
                att_edges = [{"src": int(src[e]), "tgt": int(tgt[e]), "imp": float(edge_I[e])} for e in keep_edges]
                vis_json.append({"graph_id": g, "nodes": nodes, "att_edges": att_edges, "def_edges": defender_edges})

            # DOT export
            if dot_dir:
                involved_nodes = set()
                for e in keep_edges:
                    involved_nodes.add(int(src[e]))
                    involved_nodes.add(int(tgt[e]))
                for d in defender_edges:
                    involved_nodes.add(d["att"])
                    involved_nodes.add(d["def"])

                max_I = edge_I.max().item() if edge_I.max() > 0 else 1.0
                def c(x):
                    v = min(1.0, x / max_I) if max_I > 0 else 0
                    r, g = int(255 * (1-(v*0.5))), int(255 * (1-v))
                    return f"#{r:02x}{g:02x}00"
                lines=["digraph G{", "rankdir=LR;"]

                for i in involved_nodes:
                    lines.append(f'{i} [style=filled,fillcolor="{c(node_in[i])}",label="{i}"];')
                for e in keep_edges:
                    w = max(1.0, 4 * edge_I[e] / max_I)
                    lines.append(f'{int(src[e])} -> {int(tgt[e])} [color="#e63946",penwidth={w:.2f}];')
                for d in defender_edges:
                    w = max(0.5, 3 * d["cred"] / max_cred)
                    lines.append(f'{d["def"]} -> {d["att"]} [color="#52b788",style=dashed,penwidth={w:.2f}];')
                lines.append("}")
                (dot_dir / f"graph_{g}.dot").write_text("\n".join(lines))

        # Clean up memory
        del edge_I, node_in, credit
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # (Summary print and JSON export remain the same)
    summary = {k: float(np.nanmean(v)) for k, v in stats.items()}
    summary["graphs"] = len(ds)
    print(json.dumps(summary, indent=2))

    if args.export_json:
        with open(args.export_json, 'w') as f: json.dump(vis_json, f, indent=2)
        print(f"✅ Visual JSON exported to {args.export_json}")
    if dot_dir:
        print(f"✅ DOT files exported to {dot_dir}")

if __name__ == "__main__":
    main()