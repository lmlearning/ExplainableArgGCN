#!/usr/bin/env python
# ──────────────────────────────────────────────────────────────
#  train_consolidated.py
#  Unified trainer for:
#     • refined      (with ablations: –Pairing, –Structural, –Residual,
#                     –LayerNorm, –RankLoss)
#     • afgcn, gcn, gat, graphsage, gin, randalign  (PyG baselines)
#  Works on mixed APX / TGF datasets via AFGraphDataset.
# ──────────────────────────────────────────────────────────────
import argparse, importlib.util, math, numpy as np, time
from tqdm import tqdm

from sklearn.metrics import matthews_corrcoef
import torch, torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.amp import autocast, GradScaler
from torch_scatter import scatter_sum

from train_refined_afgcn import AFGraphDataset, listnet_loss
from torch.amp import autocast, GradScaler
# ──────────────────────────────────────────────────────────────
#  Dataset, list-net loss (already provided in train_refined_afgcn.py)
# ──────────────────────────────────────────────────────────────


# ╭────────────────────────── Refined layer (with switches) ─────────────────╮
class RefinedLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, struct_dim=7,
                 disable_pairing=False, disable_struct=False,
                 disable_residual=False, disable_ln=False):
        super().__init__()
        self.struct_dim = struct_dim
        self.disable_pairing  = disable_pairing
        self.disable_struct   = disable_struct
        self.disable_residual = disable_residual
        self.disable_ln       = disable_ln

        self.embed = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else None
        self.W_att = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.phi_att = nn.Linear(hidden_dim, hidden_dim)

        # defender pathway
        self.W_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_p_prime = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.f_def = nn.Linear(hidden_dim, 1)
        self.gamma, self.delta = 1.0, 0.5

        # aggregation
        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        self.W_A    = nn.Linear(hidden_dim, hidden_dim)
        self.W_str  = nn.Linear(struct_dim-1, hidden_dim)

        self.act = nn.ReLU()
        if not self.disable_ln:
            self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, edge_index, h):
        H = self.embed(h) if self.embed else h                       # (N,H)
        s  = h[:, :self.struct_dim]                                  # (N,7)
        s_feat = s[:, :-1]                                           # (N,6)

        src, tgt = edge_index
        h_src, h_tgt = H[src], H[tgt]
        s_src, s_tgt = s_feat[src], s_feat[tgt]

        # attacker attention α
        score = (h_tgt * self.W_att(h_src)).sum(-1) + (s_src * s_tgt).sum(-1)
        alpha_num = torch.exp(score)
        alpha_den = scatter_sum(alpha_num, tgt, dim=0, dim_size=H.size(0))
        alpha = alpha_num / (alpha_den[tgt] + 1e-9)                  # (E,)

        # defender modulation ρ (if enabled)
        if not self.disable_pairing:
            rev_src, rev_tgt = tgt, src
            h_att, h_def = H[rev_tgt], H[rev_src]
            pair = (self.W_p(h_att) * self.W_p_prime(h_def)).sum(-1)
            beta_num = torch.exp(pair)
            beta_den = scatter_sum(beta_num, rev_tgt, dim=0, dim_size=H.size(0))
            beta = beta_num / (beta_den[rev_tgt] + 1e-9)
            psi = self.f_def(h_def).squeeze(-1)
            sum_term = scatter_sum(beta * torch.exp(-psi), rev_tgt,
                                   dim=0, dim_size=H.size(0))
            d = -torch.log(sum_term + 1e-8)
            rho = 1 / (1 + torch.exp(-self.gamma * (d - self.delta)))
            rho_edge = rho[src]
        else:
            rho_edge = 0.0

        # messages and update
        msg_edge = alpha.unsqueeze(-1) * self.phi_att(h_src) * (1 - rho_edge).unsqueeze(-1)
        m_att = scatter_sum(msg_edge, tgt, dim=0, dim_size=H.size(0))

        out = self.W_self(H) + self.W_A(m_att)
        if not self.disable_struct:
            out += self.W_str(s_feat)
        out = self.act(out)

        if not self.disable_residual:
            out = out + H
        if not self.disable_ln:
            out = self.ln(out)
        return out
# ╰───────────────────────────────────────────────────────────────────────────╯


# ╭─────────────────────────── Refined-AFGCN model ──────────────────────────╮
class RefinedAFGCN(nn.Module):
    def __init__(self, struct_dim=7, hidden=128, layers=2, **abl_flags):
        super().__init__()
        self.struct_dim = struct_dim
        self.input_norm = nn.LayerNorm(struct_dim)
        self.layers = nn.ModuleList()

        self.layers.append(RefinedLayer(struct_dim, hidden, struct_dim, **abl_flags))
        for _ in range(layers - 1):
            self.layers.append(
                RefinedLayer(hidden + struct_dim, hidden, struct_dim, **abl_flags))
        self.out = nn.Linear(hidden, 1)

    def forward(self, data):
        h = self.input_norm(data.x)
        edge_index = data.edge_index
        fixed_s = h[:, :self.struct_dim]
        h = self.layers[0](edge_index, h)
        for l in self.layers[1:]:
            h = torch.cat([fixed_s, h], 1)
            h = l(edge_index, h)
        return self.out(h).squeeze()
# ╰───────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────── load baseline builders from pyg_train.py ─────────────╮
def load_pyg_baselines():
    spec = importlib.util.spec_from_file_location("pyg_baselines", "./pyg_train.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    return dict(
        afgcn     = lambda h,l: mod.AFGCN(in_dim=128, hid_dim=h, num_classes=1, num_layers=l),
        gcn       = lambda h,l: mod.GCN(in_dim=128, hid=h, out_dim=1),
        gat       = lambda h,l: mod.GAT(in_dim=128, hid=64, out_dim=1, heads=4),
        graphsage = lambda h,l: mod.GraphSAGE(in_dim=128, hid=h, out_dim=1),
        gin       = lambda h,l: mod.GIN(in_dim=128, hid=h, out_dim=1), # The original had an incorrect kwarg `num_mlp_layers`
        randalign = lambda h,l: mod.RandAlignGCN(in_dim=128, hid_dim=h, num_classes=1, num_layers=l),
    )

# ╭──────────────────────── helper: pad to 128-dim inputs ────────────────────╮
def pad_inputs(x, dim=128):
    if x.size(1) == dim:  return x
    if x.size(1) > dim:   return x[:, :dim]
    pad = torch.zeros(x.size(0), dim - x.size(1), device=x.device)
    return torch.cat([x, pad], 1)
# ╰───────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────── training / evaluation loops ──────────────────────╮
# --------------------------- training & evaluation -------------------------
def train_epoch(model, loader, loss_fn, opt, scaler, device,
                refined=False, use_rank=False, lam_rank=0.1):
    model.train()
    pbar = tqdm(loader, desc="train", leave=False)
    running_loss = 0.0
    for data in pbar:
        data = data.to(device)
        x_in = data.x if refined else pad_inputs(data.x)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(data) if refined else model(Data(x=x_in, edge_index=data.edge_index))
            loss = loss_fn(logits, data.y.float())
            if refined and use_rank:
                loss += lam_rank * listnet_loss(logits, data.rank.to(device))
        running_loss += loss.item()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        # ----- free graph & cached blocks  (❷) -----
        del loss, logits
        torch.cuda.empty_cache()
        pbar.set_postfix(avg_loss=f"{running_loss/ (pbar.n or 1):.3f}")


def evaluate(model, loader, loss_fn, device, refined=False, desc="val"):
    model.eval()
    tot=cor=0; loss_sum=0.0; preds=[]; labels=[]
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad(), autocast(device_type=device.type):
        for data in pbar:
            data = data.to(device)
            x_in = data.x if refined else pad_inputs(data.x)
            logits = model(data) if refined else model(Data(x=x_in, edge_index=data.edge_index))
            loss_sum += loss_fn(logits, data.y.float()).item()
            pred = (torch.sigmoid(logits) > 0.5).float()
            cor += (pred == data.y.float()).sum().item(); tot += data.y.size(0)
            preds.extend(pred.cpu().numpy()); labels.extend(data.y.cpu().numpy())
    acc = cor / tot
    mcc = matthews_corrcoef(labels, preds) if len(np.unique(labels))>1 else 0
    return acc, mcc, loss_sum / len(loader)


# ---------------------------------- main -----------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   choices=['refined','afgcn','gcn','gat','graphsage','gin','randalign'])
    # data
    p.add_argument('--training_dir', required=True)
    p.add_argument('--validation_dir', required=True)
    # optimisation
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    # weighting & ranking loss
    p.add_argument('--use_class_weights', action='store_true')
    p.add_argument('--rank_loss_weight', type=float, default=0.1,
                   help='λ in BCE + λ·ListNet; ignored if --no_rank_loss')
    p.add_argument('--no_rank_loss', action='store_true',
                   help='Disable ranking-alignment term (–RankLoss ablation)')
    # refined ablations
    p.add_argument('--no_pairing',  action='store_true')
    p.add_argument('--no_struct',   action='store_true')
    p.add_argument('--no_residual', action='store_true')
    p.add_argument('--no_ln',       action='store_true')
    # misc
    p.add_argument('--checkpoint', default='ckpt.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds = AFGraphDataset(args.training_dir), AFGraphDataset(args.validation_dir)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, 1, shuffle=False)

 # ─ build model ─
    baseline_build = load_pyg_baselines()
    if args.model == 'refined':
        abl_flags = dict(disable_pairing=args.no_pairing,
                         disable_struct=args.no_struct,
                         disable_residual=args.no_residual,
                         disable_ln=args.no_ln)
        model = RefinedAFGCN(7, args.hidden_dim, args.num_layers, **abl_flags).to(device)
        refined = True
    else:
        model = baseline_build[args.model](args.hidden_dim, args.num_layers).to(device)
        refined = False

    # ─ loss ─
    if args.use_class_weights:
        pos = sum(torch.sum(d.y) for d in train_ds)
        tot = sum(d.y.size(0) for d in train_ds)
        weight = torch.tensor([(tot-pos)/(pos if pos else 1)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    opt, scaler = torch.optim.Adam(model.parameters(), lr=args.lr), GradScaler(device="cuda") 
    best_mcc = -1.0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_epoch(model, train_loader, loss_fn, opt, scaler, device,
                    refined=refined,
                    use_rank=(refined and not args.no_rank_loss),
                    lam_rank=args.rank_loss_weight)
        acc, mcc, vloss = evaluate(model, val_loader, loss_fn, device, refined)
        dt = time.time() - t0
        print(f"Epoch {epoch:03d} ┃ acc {acc:.4f}  mcc {mcc:.4f}  "
              f"val_loss {vloss:.4f}  [{dt/60:.1f} min]")
        if mcc > best_mcc:
            best_mcc = mcc
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict()},
                       args.checkpoint)
            print(f"   ↳ new best saved ▶ {args.checkpoint}")


if __name__ == "__main__":
    main()