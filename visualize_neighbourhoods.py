#!/usr/bin/env python
# visualize_neighborhoods.py
# --------------------------------------------------------------------------
# This script generates focused, local visualizations for a trained model.
#
# Modifications:
# - Reverted --exclude_self_attacks to the simple and robust logic of
#   setting self-loop influence to zero.
# --------------------------------------------------------------------------
import argparse
import pathlib
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum

from train_refined_afgcn import AFGraphDataset
from train_consolidated import RefinedAFGCN

# ---------------------- Influence & Credit Extractor -----------------------
@torch.no_grad()
def intrinsic_with_credit(model, data):
    """
    Performs a forward pass, layer by layer, to extract intrinsic edge
    influence (I) and defender credit (C) without using hooks.
    """
    struct = model.struct_dim
    h = model.input_norm(data.x)
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

def main():
    ap = argparse.ArgumentParser(description="Generate sampled neighborhood visualizations from a trained model.")
    ap.add_argument('--data_dir', required=True, help="Directory containing the graph dataset.")
    ap.add_argument('--ckpt', required=True, help="Path to the trained model checkpoint (.pth file).")
    ap.add_argument('--output_dir', required=True, help="Directory to save the output DOT files.")
    ap.add_argument('--hidden_dim', type=int, default=128, help="Model's hidden dimension size.")
    ap.add_argument('--layers', type=int, default=2, help="Number of layers in the model.")
    ap.add_argument('--num_samples', type=int, default=3, help="Number of nodes to sample per graph.")
    ap.add_argument('--influence_threshold', nargs=2, type=float, default=[0.85, 0.75],
                    metavar=('ATT_THRESH', 'DEF_THRESH'),
                    help="Cumulative influence thresholds for attackers and defenders.")
    ap.add_argument('--sampling_method', type=str, default='top', choices=['top', 'random'],
                    help="Method for sampling seed nodes: 'top' for highest influence, 'random' for stochastic.")
    ap.add_argument('--exclude_self_attacks', action='store_true',
                    help="If set, sets the influence of all self-attacking edges to zero.")
    ap.add_argument('--in_only', action='store_true',
                    help="If set, only selects seed nodes that are predicted as 'IN' (probability > 0.5).")
    args = ap.parse_args()
    att_thresh, def_thresh = args.influence_threshold

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = AFGraphDataset(args.data_dir)
    ld = DataLoader(ds, 1, shuffle=False)
    model = RefinedAFGCN(7, args.hidden_dim, args.layers).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    print(f"✅ Setup complete. Using '{args.sampling_method}' sampling for {args.num_samples} neighborhoods per graph.")
    if args.exclude_self_attacks:
        print("ℹ️  Influence of self-attacking edges will be set to zero.")
    if args.in_only:
        print("ℹ️  Only nodes predicted as 'IN' will be selected as seeds.")

    files_saved_count = 0
    for g, data in enumerate(tqdm(ld, desc="Processing Graphs")):
        data = data.to(device)
        edge_I, node_in, credit = intrinsic_with_credit(model, data)
        src, tgt = data.edge_index
        N = data.num_nodes

        with torch.no_grad():
            class_logits, _ = model(data)
            probabilities = torch.sigmoid(class_logits).cpu()

        # --- MODIFIED: Reverted to the simple and robust "zero-out" logic ---
        if args.exclude_self_attacks:
            self_attack_mask = (src == tgt).cpu()
            edge_I[self_attack_mask] = 0.0
            node_in = scatter_sum(edge_I, tgt.cpu(), dim=0, dim_size=N)

        if N == 0: continue

        # --- Seed selection logic ---
        candidate_indices = list(range(N))
        if args.in_only:
            in_nodes_mask = (probabilities > 0.5)
            candidate_indices = [i for i in candidate_indices if in_nodes_mask[i].item()]

        if not candidate_indices: continue
        
        seed_nodes = []
        if args.sampling_method == 'top':
            candidate_pool = sorted(candidate_indices, key=lambda i: node_in[i], reverse=True)
            num_to_sample = min(args.num_samples, len(candidate_pool))
            seed_nodes = candidate_pool[:num_to_sample]
        else: # 'random'
            num_to_sample = min(args.num_samples, len(candidate_indices))
            perm = torch.randperm(len(candidate_indices))
            selected_indices = perm[:num_to_sample]
            seed_nodes = [candidate_indices[i] for i in selected_indices]
        
        if not seed_nodes: continue

        # --- Visualization loop ---
        for seed_node_id in seed_nodes:
            local_attack_edge_indices = (tgt == seed_node_id).nonzero(as_tuple=True)[0].cpu()
            if local_attack_edge_indices.numel() == 0: continue
            
            local_edge_I = edge_I[local_attack_edge_indices]
            total_local_influence = local_edge_I.sum()
            if total_local_influence <= 0: continue

            sorted_local_indices = torch.argsort(local_edge_I, descending=True)
            influence_sum = 0.0
            keep_local_edges = []
            for idx in sorted_local_indices:
                edge_idx = local_attack_edge_indices[idx].item()
                keep_local_edges.append(edge_idx)
                influence_sum += local_edge_I[idx]
                if influence_sum >= total_local_influence * att_thresh:
                    break
            
            if not keep_local_edges: continue

            local_attackers = {int(src[e]) for e in keep_local_edges}
            defender_edges = []
            max_cred = 1e-9
            for attacker_id in local_attackers:
                defenses = sorted([item for item in credit.items() if item[0][0] == attacker_id], key=lambda x: -x[1])
                if not defenses: continue
                total_credit = sum(val for _, val in defenses)
                credit_sum = 0.0
                if total_credit > 0:
                    for (j, k), val in defenses:
                        defender_edges.append({"att": j, "def": k, "cred": float(val)})
                        if val > max_cred: max_cred = val
                        credit_sum += val
                        if credit_sum >= total_credit * def_thresh:
                            break

            involved_nodes = {seed_node_id}
            for e in keep_local_edges: involved_nodes.add(int(src[e]))
            for d in defender_edges: involved_nodes.add(d["def"]); involved_nodes.add(d["att"])
            
            def value_to_color(prob_val):
                v = max(0.0, min(1.0, prob_val))
                r = int(255 * (1 - v)); g = int(255 * v)
                return f"#{r:02x}{g:02x}00"
            
            lines = ["digraph G{", "rankdir=LR;"]
            for i in involved_nodes:
                shape = "box" if i == seed_node_id else "circle"
                color = value_to_color(probabilities[i].item())
                lines.append(f'{i} [style=filled, shape={shape}, fillcolor="{color}", label="{i}"];')
            for e in keep_local_edges:
                max_I_local = local_edge_I.max().item() if local_edge_I.numel() > 0 else 1.0
                w = max(1.0, 4 * edge_I[e] / max_I_local) if max_I_local > 0 else 1.0
                lines.append(f'{int(src[e])} -> {int(tgt[e])} [color="#e63946",penwidth={w:.2f}];')
            for d in defender_edges:
                w = max(0.5, 3 * d["cred"] / max_cred)
                lines.append(f'{d["def"]} -> {d["att"]} [color="#52b788",style=dashed,penwidth={w:.2f}];')
            lines.append("}")
            
            dot_filename = output_dir / f"graph_{g}_seed_{seed_node_id}.dot"
            dot_filename.write_text("\n".join(lines))
            files_saved_count += 1

    print(f"\n✅ Visualization complete. Saved {files_saved_count} DOT files to '{output_dir}'.")

if __name__ == "__main__":
    main()