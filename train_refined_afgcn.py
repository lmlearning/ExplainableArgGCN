# ────────────────────────────────────────────────────────────────
#  train_refined_afgcn.py
# ────────────────────────────────────────────────────────────────
#!/usr/bin/env python
"""
Train the Refined-AFGCN model on abstract-argumentation graphs.
Supports both TGF and APX formats and uses Categoriser ranking
as an auxiliary training signal (ListNet loss).

Run example:
    python train_refined_afgcn.py \
           --training_dir iccma_train \
           --validation_dir iccma_val \
           --use_class_weights \
           --use_ranking_loss --lambda_rank 0.1
"""

import os, random, argparse, pickle, math, time
import numpy as np, networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

import torch, torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_sum, scatter_softmax
from torch.amp import autocast, GradScaler

# ─────────────────────────────────────────
#  Section 1 –  Utils
# ─────────────────────────────────────────
def solve(adj_matrix):
    """Grounded extension solver (IN, OUT, BLANK = 1/2/0)."""
    BLANK, IN, OUT = 0, 1, 2
    n = adj_matrix.shape[0]
    lab = np.zeros(n, np.int8)
    unattacked = np.where(np.sum(adj_matrix, 0) == 0)[0]
    lab[unattacked] = IN
    cascade = True
    while cascade:
        new_out = np.unique(np.nonzero(adj_matrix[unattacked, :])[1])
        new_out = np.array([i for i in new_out if lab[i] != OUT])
        if new_out.size:
            lab[new_out] = OUT
            affected = np.unique(np.nonzero(adj_matrix[new_out, :])[1])
        else:
            affected = np.empty(0, dtype=int)
        new_in = []
        for idx in affected:
            attackers = np.nonzero(adj_matrix[:, idx])[0]
            if np.sum(lab[attackers] == OUT) == len(attackers):
                new_in.append(idx)
        if new_in:
            new_in = np.array(new_in)
            lab[new_in] = IN
            unattacked = new_in
        else:
            cascade = False
    return np.where(lab == IN)[0]


def categoriser_ranking(adj_matrix, max_iter=1000, tol=1e-9):
    """Besnard & Hunter Categoriser ranking."""
    n = adj_matrix.shape[0]
    r = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        new_r = 1.0 / (1.0 + adj_matrix.T @ r)
        if np.linalg.norm(new_r - r, 1) < tol:
            break
        r = new_r
    return r.astype(np.float32)


def parseTGF(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    args, atts, hash_seen = [], [], False
    for ln in lines:
        if ln == '#':
            hash_seen = True
            continue
        if not hash_seen:
            args.append(ln)
        else:
            atts.append(ln.split(' '))
    return args, atts


def parse_apx(path):
    args, atts = [], []
    with open(path) as f:
        for ln in f:
            ln = ln.strip().rstrip('.')
            if ln.startswith('arg('):
                args.append(ln[4:-1])
            elif ln.startswith('att('):
                a, b = ln[4:-1].split(',')
                atts.append([a, b])
    return args, atts


# ─────────────────────────────────────────
#  Section 2 –  PyG Dataset
# ─────────────────────────────────────────
class AFGraphDataset(Dataset):
    def __init__(
        self,
        root_dir,
        label_ext=".EE-PR",
        struct_dim=7,
    ):
        self.root_dir = root_dir
        self.label_ext = label_ext
        self.struct_dim = struct_dim

        self.graph_files = []
        for f in os.listdir(root_dir):
            if f.endswith(('.tgf', '.apx')):
                base = os.path.splitext(f)[0]
                if os.path.exists(os.path.join(root_dir, base + label_ext)):
                    self.graph_files.append(f)
        self.cache = {}

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        gfile = self.graph_files[idx]
        path = os.path.join(self.root_dir, gfile)
        ext = os.path.splitext(gfile)[1].lower()

        if ext == '.tgf':
            args_list, atts = parseTGF(path)
        elif ext == '.apx':
            args_list, atts = parse_apx(path)
        else:
            raise ValueError(ext)

        G = nx.DiGraph()
        G.add_nodes_from(args_list)
        G.add_edges_from(atts)
        G, mapping = nx.convert_node_labels_to_integers(G, label_attribute="orig"), {n:i for i,n in enumerate(G.nodes())}

        # grounded flags
        adj = nx.to_numpy_array(G, dtype=np.float32)
        in_nodes = solve(adj)
        gflag = np.zeros((G.number_of_nodes(), 1), np.float32)
        gflag[in_nodes] = 1.0

        # structural descriptors (6 features) + grounded flag
        cache_f = os.path.join(self.root_dir, os.path.splitext(gfile)[0] + "_f.pkl")
        if os.path.exists(cache_f):
            feat = pickle.load(open(cache_f, "rb"))
        else:
            deg_in  = dict(G.in_degree())
            deg_out = dict(G.out_degree())
            pr      = nx.pagerank(G)
            clos    = nx.closeness_centrality(G)
            eig     = nx.eigenvector_centrality(G, max_iter=10000)
            raw = np.array([[deg_in[v], deg_out[v],
                             pr[v], clos[v], eig[v], 0.0]  # colour placeholder
                            for v in G.nodes()], dtype=np.float32)
            scaler = StandardScaler()
            feat = scaler.fit_transform(raw)
            pickle.dump(feat, open(cache_f, "wb"))
        x = np.concatenate([feat, gflag], 1)  # (N,7)

        # labels
        lbl_p = os.path.join(self.root_dir, os.path.splitext(gfile)[0] + self.label_ext)
        lab_nodes = []
        with open(lbl_p) as f:
            line = f.readline().strip()[1:-1].replace("]]", "")
            lab_nodes = [z.strip("] ") for sub in line.split("],") for z in sub.split(',')]
        y = np.zeros(G.number_of_nodes(), np.int64)
        for n in lab_nodes:
            if n in mapping:
                y[mapping[n]] = 1

        # Categoriser ranking
        rank = categoriser_ranking(adj)

        # edge index with self-loops
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        edge_index, _ = add_self_loops(edge_index, num_nodes=G.number_of_nodes())

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
            rank=torch.tensor(rank, dtype=torch.float),
        )
        self.cache[idx] = data
        return data


# ─────────────────────────────────────────
#  Section 3 –  Model
# ─────────────────────────────────────────
class RefinedLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, struct_dim=7,
                 epsilon=0.1, lambda_att=1.0, lambda_def=1.0,
                 gamma=1.0, delta=0.5):
        super().__init__()
        self.struct_dim = struct_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.lambda_att = lambda_att
        self.lambda_def = lambda_def
        self.gamma = gamma
        self.delta = delta

        self.embed = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else None
        self.W_att = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.phi_att = nn.Linear(hidden_dim, hidden_dim)
        self.W_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_p_prime = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.f_def = nn.Linear(hidden_dim, 1)
        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        self.W_A = nn.Linear(hidden_dim, hidden_dim)
        self.W_str = nn.Linear(struct_dim - 1, hidden_dim)
        self.act = nn.ReLU()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, edge_index, h):
        if self.embed is not None:
            H = self.embed(h)
        else:
            H = h
        s = h[:, :self.struct_dim]
        s_feat = s[:, :-1]

        src, tgt = edge_index
        h_src, h_tgt = H[src], H[tgt]
        s_src, s_tgt = s_feat[src], s_feat[tgt]

        W_hs = self.W_att(h_src)
        score = (h_tgt * W_hs).sum(-1) + (s_src * s_tgt).sum(-1)
        alpha = scatter_softmax(score, tgt)

        # defender pathway
        rev_src, rev_tgt = tgt, src
        h_att, h_def = H[rev_tgt], H[rev_src]
        pair = (self.W_p(h_att) * self.W_p_prime(h_def)).sum(-1)
        beta = scatter_softmax(pair, rev_tgt)
        psi = self.f_def(h_def).squeeze(-1)
        sum_term = scatter_sum(beta * torch.exp(-psi), rev_tgt, dim_size=H.size(0))
        d = -torch.log(sum_term + 1e-8)
        rho = 1 / (1 + torch.exp(-self.gamma * (d - self.delta)))
        rho_edge = rho[src]

        m_e = alpha.unsqueeze(-1) * self.phi_att(h_src) * (1 - rho_edge).unsqueeze(-1)
        m_att = scatter_sum(m_e, tgt, dim=0, dim_size=H.size(0))
        H_new = self.act(self.W_self(H) + self.W_A(m_att) + self.W_str(s_feat)) + H
        return self.ln(H_new)


class RefinedAFGCN(nn.Module):
    def __init__(self, in_feats=7, hidden=128, layers=2, struct_dim=7):
        super().__init__()
        self.struct_dim = struct_dim
        self.input_norm = nn.LayerNorm(in_feats)
        self.layers = nn.ModuleList()
        self.layers.append(RefinedLayer(in_feats, hidden, struct_dim))
        for _ in range(layers - 1):
            self.layers.append(
                RefinedLayer(hidden + struct_dim, hidden, struct_dim))
        self.out = nn.Linear(hidden, 1)

    def forward(self, data):
        h = self.input_norm(data.x)
        edge_index = data.edge_index
        s_fixed = h[:, :self.struct_dim]
        h = self.layers[0](edge_index, h)
        for l in self.layers[1:]:
            h = torch.cat([s_fixed, h], 1)
            h = l(edge_index, h)
        return self.out(h).squeeze()


# ─────────────────────────────────────────
#  Section 4 –  Losses
# ─────────────────────────────────────────
def listnet_loss(pred, target, eps=1e-9):
    p = torch.softmax(pred, 0)
    q = torch.softmax(target, 0)
    return -(q * torch.log(p + eps)).sum()


# ─────────────────────────────────────────
#  Section 5 –  Training helpers
# ─────────────────────────────────────────
def train_epoch(model, loader, bce, opt, device, scaler,
                use_rank, lam_rank, subset=False, ratio=0.5):
    model.train()
    pbar = tqdm(loader, desc="train", leave=False)
    cum_loss = 0.0
    for data in pbar:
        data = data.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(data)
            lbl = data.y.float()
            if subset:
                k = max(1, int(ratio * lbl.size(0)))
                idx = random.sample(range(lbl.size(0)), k)
                loss_bce = bce(logits[idx], lbl[idx])
                rank_targets = data.rank[idx]
            else:
                loss_bce = bce(logits, lbl)
                rank_targets = data.rank
            loss = loss_bce + (lam_rank * listnet_loss(logits, rank_targets.to(device))
                               if use_rank else 0.0)
        cum_loss += loss.item()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        # ----- free graph & cached blocks  (❷) -----
        del loss, logits
        torch.cuda.empty_cache()
        pbar.set_postfix(avg_loss=f"{cum_loss/ (pbar.n or 1):.3f}")


def evaluate(model, loader, loss_fn, device):
    model.eval()
    tot=cor=0; cum_loss=0.0; preds_all=[]; labels_all=[]
    pbar = tqdm(loader, desc="val", leave=False)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        for data in pbar:
            data = data.to(device)
            logits = model(data)
            loss = loss_fn(logits, data.y.float())
            cum_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            cor += (preds == data.y.float()).sum().item(); tot += data.y.size(0)
            preds_all.extend(preds.cpu().numpy()); labels_all.extend(data.y.cpu().numpy())
    acc = cor / tot
    mcc = matthews_corrcoef(labels_all, preds_all) if len(np.unique(labels_all))>1 else 0
    return acc, mcc, cum_loss/len(loader)
# ────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────
#  Section 6 –  Main  (only the loop changed)
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # (same argparse arguments as before)
    # …
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = AFGraphDataset(args.training_dir)
    val_ds   = AFGraphDataset(args.validation_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = RefinedAFGCN(7, args.hidden_dim, args.num_layers).to(device)

    # BCE / class weights
    if args.use_class_weights:
        pos = sum(torch.sum(d.y) for d in train_ds)
        tot = sum(d.y.size(0) for d in train_ds)
        weight = torch.tensor([(tot-pos)/(pos if pos else 1)], device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        bce = nn.BCEWithLogitsLoss()

    opt, scaler = torch.optim.Adam(model.parameters(), lr=args.lr), GradScaler(device="cuda") 
    best_mcc = -1.0

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        train_epoch(model, train_loader, bce, opt, device, scaler,
                    args.use_ranking_loss, args.lambda_rank,
                    args.subset, 0.5)
        acc, mcc, vloss = evaluate(model, val_loader, bce, device)
        dt = (time.time()-t0)/60
        print(f"Epoch {ep:03d} │ acc {acc:.4f}  mcc {mcc:.4f}  "
              f"val_loss {vloss:.4f}  [{dt:.1f} min]")
        if mcc > best_mcc:
            best_mcc = mcc
            torch.save({'epoch':ep,'model_state_dict':model.state_dict()},
                       args.checkpoint)
            print(f"   ↳ new best ({mcc:.4f}) saved to {args.checkpoint}")