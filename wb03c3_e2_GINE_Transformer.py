# %%
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import json, random, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_fig(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", **kwargs)
    plt.close()

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import wandb

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve, confusion_matrix,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import (
    GINEConv, TransformerConv,
    global_max_pool, GlobalAttention, JumpingKnowledge,
)
from torch_geometric.utils import to_undirected

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RNG_SEED = 7
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

PROJECT_ROOT  = Path.cwd()
DATA_DIR      = PROJECT_ROOT / "DATA"
PROCESSED     = DATA_DIR / "processed"
ARRAYS_DIR    = PROCESSED / "arrays"
ARTIFACTS_DIR = PROCESSED / "artifacts"
PACK_DIR      = ARTIFACTS_DIR / "packed"
RESULTS_DIR   = PROJECT_ROOT / "results"
WB03C3_DIR    = RESULTS_DIR / "wb03c3"
WB03C3_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

SMOKE_TEST    = False
N_TRIALS      = 2  if SMOKE_TEST else 20
MAX_EPOCHS    = 5  if SMOKE_TEST else 80
PATIENCE      = 3  if SMOKE_TEST else 15
BATCH_SIZE    = 256
WANDB_PROJECT = "elliptic2-gnn"
WANDB_ENABLED = not SMOKE_TEST

print(f"SMOKE_TEST={SMOKE_TEST}  trials={N_TRIALS}  epochs={MAX_EPOCHS}")


# %%
X = np.load(ARRAYS_DIR / "node_features.npy")
subgraph_labels = {
    int(k): v for k, v in
    json.loads((ARTIFACTS_DIR / "subgraph_labels.json").read_text()).items()
}
splits = json.loads((ARTIFACTS_DIR / "splits.json").read_text())

nodes_pack        = np.load(PACK_DIR / "nodes_by_ccid.npz")
edges_pack        = np.load(PACK_DIR / "edges_by_ccid.npz")
unique_cc         = nodes_pack["unique_cc"].astype(np.int64)
node_ptr          = nodes_pack["node_ptr"].astype(np.int64)
node_row_perm     = nodes_pack["node_row_perm"].astype(np.int64)
unique_cc_edges   = edges_pack["unique_cc_edges"].astype(np.int64)
edge_ptr          = edges_pack["edge_ptr"].astype(np.int64)
edge_src_row_perm = edges_pack["edge_src_row_perm"].astype(np.int64)
edge_dst_row_perm = edges_pack["edge_dst_row_perm"].astype(np.int64)

ccid_to_i  = {int(c): i for i, c in enumerate(unique_cc)}
ccid_to_ei = {int(c): i for i, c in enumerate(unique_cc_edges)}

def label_to_int(lbl):
    return 1 if str(lbl).lower() in {"suspicious", "illicit"} else 0

y_by_cc = {int(c): label_to_int(subgraph_labels[int(c)]) for c in unique_cc}

# Edge features (from Wb03c1)
EDGE_FEATURES = np.load(ARRAYS_DIR / "edge_features.npy")
EDGE_FEAT_DIM = EDGE_FEATURES.shape[1]

# Dense subset (from Wb03c1)
EDGE_FEATURES_DENSE = np.load(ARRAYS_DIR / "edge_features_dense.npy")
DENSE_FEAT_DIM = EDGE_FEATURES_DENSE.shape[1]
dense_meta = json.loads((ARTIFACTS_DIR / "edge_feature_dense_meta.json").read_text())

IN_DIM = X.shape[1]  # 43

print(f"Node features: {X.shape}")
print(f"Edge features: {EDGE_FEATURES.shape} (all), {EDGE_FEATURES_DENSE.shape} (dense)")
print(f"Subgraphs: {len(unique_cc):,}")


# %%
class Elliptic2EdgeFeatureDataset(torch.utils.data.Dataset):
    """PyG dataset with edge features. Identical to Wb03c2."""

    def __init__(self, ccids, edge_features=None, use_random_edges=False):
        self.ccids = np.asarray(ccids, dtype=np.int64)
        self.edge_features = edge_features if edge_features is not None else EDGE_FEATURES
        self.use_random_edges = use_random_edges
        self._rng = np.random.RandomState(RNG_SEED)

    def __len__(self):
        return self.ccids.shape[0]

    def __getitem__(self, idx):
        ccid = int(self.ccids[idx])
        i    = ccid_to_i[ccid]
        rows = node_row_perm[node_ptr[i] : node_ptr[i + 1]]
        x    = torch.from_numpy(X[rows]).float()

        local = {int(r): j for j, r in enumerate(rows.tolist())}

        if ccid in ccid_to_ei:
            ei = ccid_to_ei[ccid]
            s  = edge_src_row_perm[edge_ptr[ei] : edge_ptr[ei + 1]]
            t  = edge_dst_row_perm[edge_ptr[ei] : edge_ptr[ei + 1]]

            global_edge_start = edge_ptr[ei]
            global_edge_end   = edge_ptr[ei + 1]

            src = torch.tensor([local[int(r)] for r in s], dtype=torch.long)
            dst = torch.tensor([local[int(r)] for r in t], dtype=torch.long)

            edge_index_dir = torch.stack([src, dst], dim=0)

            e_feats = self.edge_features[global_edge_start:global_edge_end]
            if self.use_random_edges:
                e_feats = self._rng.randn(*e_feats.shape).astype(np.float32)
            edge_attr_dir = torch.from_numpy(e_feats.copy()).float()

            edge_index_rev = torch.stack([dst, src], dim=0)
            edge_index = torch.cat([edge_index_dir, edge_index_rev], dim=1)
            edge_attr  = torch.cat([edge_attr_dir, edge_attr_dir], dim=0)

            from torch_geometric.utils import coalesce
            edge_index, edge_attr = coalesce(
                edge_index, edge_attr, reduce="mean")
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, self.edge_features.shape[1]),
                                     dtype=torch.float32)

        y = torch.tensor([y_by_cc[ccid]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, ccId=ccid)


train_cc = np.array(splits["train"], dtype=np.int64)
val_cc   = np.array(splits["val"],   dtype=np.int64)
test_cc  = np.array(splits["test"],  dtype=np.int64)

train_ds = Elliptic2EdgeFeatureDataset(train_cc)
val_ds   = Elliptic2EdgeFeatureDataset(val_cc)
test_ds  = Elliptic2EdgeFeatureDataset(test_cc)

train_ds_dense = Elliptic2EdgeFeatureDataset(train_cc, edge_features=EDGE_FEATURES_DENSE)
val_ds_dense   = Elliptic2EdgeFeatureDataset(val_cc,   edge_features=EDGE_FEATURES_DENSE)
test_ds_dense  = Elliptic2EdgeFeatureDataset(test_cc,  edge_features=EDGE_FEATURES_DENSE)

train_labels = torch.tensor([y_by_cc[int(c)] for c in train_cc])
n_pos = int(train_labels.sum().item())
n_neg = int((train_labels == 0).sum().item())
CLASS_WEIGHTS = torch.tensor(
    [1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

print(f"Class weights: {CLASS_WEIGHTS.tolist()}")
print(f"Sample: {train_ds[0]}")


# %%
class GINEClassifier(nn.Module):
    """GINEConv-based classifier with edge features as additive bias."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=2,
                 edge_feat_dim=95, dropout=0.1, pool="max", jk_mode="none"):
        super().__init__()
        self.dropout    = dropout
        self.num_layers = num_layers
        self.jk_mode    = jk_mode
        self.pool_type  = pool

        # Project edge features to match node dim at each layer
        self.edge_projs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            c_in = in_dim if i == 0 else hidden_dim

            # GINEConv needs edge_attr same dim as node features
            self.edge_projs.append(nn.Linear(edge_feat_dim, c_in))

            # The inner MLP for GINEConv
            mlp = nn.Sequential(
                nn.Linear(c_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # JK
        if jk_mode != "none":
            self.jk = JumpingKnowledge(
                mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
            jk_out = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim
        else:
            self.jk = None
            jk_out = hidden_dim

        # Pooling
        if pool == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(jk_out, jk_out), nn.ReLU(), nn.Linear(jk_out, 1))
            self.pool_fn = GlobalAttention(gate_nn)
        else:
            self.pool_fn = global_max_pool

        # MLP head
        self.lin1 = nn.Linear(jk_out, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)

        xs = []
        for i, (conv, bn, edge_proj) in enumerate(
                zip(self.convs, self.bns, self.edge_projs)):
            ea = edge_proj(edge_attr)
            x = conv(x, edge_index, ea)
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = self.jk(xs) if self.jk is not None else xs[-1]
        g = self.pool_fn(x, batch) if self.pool_type == "attention" else self.pool_fn(x, batch)
        g = F.relu(self.lin1(g))
        if self.dropout > 0:
            g = F.dropout(g, p=self.dropout, training=self.training)
        return self.lin2(g)


class TransformerConvClassifier(nn.Module):
    """TransformerConv-based classifier with edge features in attention."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=2,
                 edge_feat_dim=95, heads=2, dropout=0.1,
                 pool="max", jk_mode="none"):
        super().__init__()
        self.dropout    = dropout
        self.num_layers = num_layers
        self.jk_mode    = jk_mode
        self.pool_type  = pool

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            c_in = in_dim if i == 0 else hidden_dim

            if i < num_layers - 1:
                # multi-head with concat
                per_head = max(hidden_dim // heads, 1)
                self.convs.append(TransformerConv(
                    c_in, per_head, heads=heads,
                    edge_dim=edge_feat_dim, concat=True, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(per_head * heads))
            else:
                # final layer: single head
                self.convs.append(TransformerConv(
                    c_in, hidden_dim, heads=1,
                    edge_dim=edge_feat_dim, concat=False, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        # JK
        if jk_mode != "none":
            self.jk = JumpingKnowledge(
                mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
            jk_out = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim
        else:
            self.jk = None
            jk_out = hidden_dim

        # Pooling
        if pool == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(jk_out, jk_out), nn.ReLU(), nn.Linear(jk_out, 1))
            self.pool_fn = GlobalAttention(gate_nn)
        else:
            self.pool_fn = global_max_pool

        # MLP head
        self.lin1 = nn.Linear(jk_out, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch)

        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = self.jk(xs) if self.jk is not None else xs[-1]
        g = self.pool_fn(x, batch) if self.pool_type == "attention" else self.pool_fn(x, batch)
        g = F.relu(self.lin1(g))
        if self.dropout > 0:
            g = F.dropout(g, p=self.dropout, training=self.training)
        return self.lin2(g)


# Smoke test both architectures
_sample = train_ds[0]
_sample.batch = torch.zeros(_sample.x.size(0), dtype=torch.long)

print("GINEConv:")
for _h in [64, 128]:
    _m = GINEClassifier(in_dim=IN_DIM, hidden_dim=_h, num_layers=2)
    with torch.no_grad():
        _out = _m(_sample)
    _np = sum(p.numel() for p in _m.parameters())
    print(f"  hidden={_h:3d} | params={_np:>8,} | out={tuple(_out.shape)}")

print("TransformerConv:")
for _h in [64, 128]:
    _m = TransformerConvClassifier(in_dim=IN_DIM, hidden_dim=_h, num_layers=2, heads=2)
    with torch.no_grad():
        _out = _m(_sample)
    _np = sum(p.numel() for p in _m.parameters())
    print(f"  hidden={_h:3d} | params={_np:>8,} | out={tuple(_out.shape)}")

print("\nBoth architectures pass smoke test.")


# %%
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ss = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs  = F.softmax(logits, dim=1)[:, 1]
        ys.append(batch.y.view(-1).cpu().numpy())
        ss.append(probs.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def optimal_f1_threshold(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[max(idx - 1, 0)]) if len(thr) else 0.5


def compute_test_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "test_pr_auc":    float(average_precision_score(y_true, y_score)),
        "test_roc_auc":   float(roc_auc_score(y_true, y_score)),
        "test_f1":        float(f1_score(y_true, y_pred)),
        "test_precision": float(tp / max(tp + fp, 1)),
        "test_recall":    float(tp / max(tp + fn, 1)),
        "test_threshold": float(threshold),
        "test_confusion": cm.tolist(),
    }


# %%
def train_and_evaluate(*, arch, hidden_dim, num_layers, dropout, lr,
                       pool, jk_mode, heads=2,
                       datasets=None, edge_feat_dim=None,
                       trial=None, verbose=False):
    """Train one GINEConv or TransformerConv configuration."""
    t0 = time.time()

    if datasets is not None:
        _train_ds, _val_ds, _test_ds = datasets
    else:
        _train_ds, _val_ds, _test_ds = train_ds, val_ds, test_ds

    if edge_feat_dim is None:
        edge_feat_dim = EDGE_FEAT_DIM

    train_loader = PyGDataLoader(_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = PyGDataLoader(_val_ds,   batch_size=512, shuffle=False)
    test_loader  = PyGDataLoader(_test_ds,  batch_size=512, shuffle=False)

    if arch == "gine":
        model = GINEClassifier(
            in_dim=IN_DIM, hidden_dim=hidden_dim, num_layers=num_layers,
            edge_feat_dim=edge_feat_dim, dropout=dropout,
            pool=pool, jk_mode=jk_mode,
        ).to(DEVICE)
    elif arch == "transformer":
        model = TransformerConvClassifier(
            in_dim=IN_DIM, hidden_dim=hidden_dim, num_layers=num_layers,
            edge_feat_dim=edge_feat_dim, heads=heads, dropout=dropout,
            pool=pool, jk_mode=jk_mode,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    run = None
    if WANDB_ENABLED:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"wb03c3_{arch}_h{hidden_dim}",
            tags=["wb03c3", arch, f"pool_{pool}", f"jk_{jk_mode}"],
            config=dict(notebook="wb03c3", arch=arch, hidden_dim=hidden_dim,
                        num_layers=num_layers, dropout=dropout, lr=lr,
                        pool=pool, jk_mode=jk_mode, heads=heads,
                        edge_feat_dim=edge_feat_dim,
                        n_params=n_params),
            reinit=True)

    best_val_pr, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            try:
                loss = criterion(model(batch), batch.y.view(-1))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    print(f"  OOM at epoch {epoch}")
                    continue
                raise
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
        avg_loss = epoch_loss / len(_train_ds)

        yv, sv = evaluate(model, val_loader)
        vp = float(average_precision_score(yv, sv))
        vr = float(roc_auc_score(yv, sv))

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  loss={avg_loss:.4f}"
                  f"  val_pr={vp:.4f}  val_roc={vr:.4f}")

        if run:
            wandb.log({"train/loss": avg_loss, "val/pr_auc": vp,
                        "val/roc_auc": vr, "epoch": epoch})

        if trial is not None:
            trial.report(vp, epoch)
            if trial.should_prune():
                if run:
                    wandb.log({"pruned": True}); run.finish(quiet=True)
                raise optuna.TrialPruned()

        if vp > best_val_pr + 1e-4:
            best_val_pr = vp
            best_state  = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
            best_epoch  = epoch
            bad_epochs  = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    wall = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    yt, st = evaluate(model, test_loader)
    thr    = optimal_f1_threshold(yt, st)
    tm     = compute_test_metrics(yt, st, thr)

    result = dict(arch=arch, hidden_dim=hidden_dim, num_layers=num_layers,
                  dropout=dropout, lr=lr, pool=pool, jk_mode=jk_mode,
                  heads=heads, edge_feat_dim=edge_feat_dim,
                  n_params=n_params, best_val_pr_auc=best_val_pr,
                  best_epoch=best_epoch, wall_seconds=wall, **tm)

    if run:
        wandb.log(dict(best_val_pr_auc=best_val_pr, best_epoch=best_epoch,
                        n_params=n_params, wall_seconds=wall, **tm))
        run.finish(quiet=True)

    return result, best_state


# %%
def make_gine_objective():
    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        num_layers = trial.suggest_categorical("num_layers", [2, 3])
        dropout    = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)
        lr         = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        pool       = trial.suggest_categorical("pool",    ["max", "attention"])
        jk_mode    = trial.suggest_categorical("jk_mode", ["none", "cat"])

        result, _ = train_and_evaluate(
            arch="gine", hidden_dim=hidden_dim, num_layers=num_layers,
            dropout=dropout, lr=lr, pool=pool, jk_mode=jk_mode, trial=trial)
        trial.set_user_attr("result", result)
        return result["best_val_pr_auc"]
    return objective

gine_study = optuna.create_study(
    study_name="wb03c3_gine",
    direction="maximize",
    sampler=TPESampler(seed=RNG_SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
gine_study.optimize(make_gine_objective(), n_trials=N_TRIALS,
                    show_progress_bar=True)

gine_results = [t.user_attrs["result"]
                for t in gine_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]

n_pruned = sum(1 for t in gine_study.trials
               if t.state == optuna.trial.TrialState.PRUNED)
print(f"\nGINEConv — Completed: {len(gine_results)}/{N_TRIALS}  Pruned: {n_pruned}")
print(f"Best trial #{gine_study.best_trial.number}  "
      f"val PR-AUC = {gine_study.best_value:.4f}")
for k, v in gine_study.best_trial.params.items():
    print(f"  {k} = {v}")

(WB03C3_DIR / "gine_search_results.json").write_text(
    json.dumps(gine_results, indent=2))


# %%
def make_transformer_objective():
    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        num_layers = trial.suggest_categorical("num_layers", [2, 3])
        heads      = trial.suggest_categorical("heads", [1, 2, 4])
        dropout    = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)
        lr         = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        pool       = trial.suggest_categorical("pool",    ["max", "attention"])
        jk_mode    = trial.suggest_categorical("jk_mode", ["none", "cat"])

        result, _ = train_and_evaluate(
            arch="transformer", hidden_dim=hidden_dim, num_layers=num_layers,
            heads=heads, dropout=dropout, lr=lr, pool=pool, jk_mode=jk_mode,
            trial=trial)
        trial.set_user_attr("result", result)
        return result["best_val_pr_auc"]
    return objective

transformer_study = optuna.create_study(
    study_name="wb03c3_transformer",
    direction="maximize",
    sampler=TPESampler(seed=RNG_SEED + 1),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
transformer_study.optimize(make_transformer_objective(), n_trials=N_TRIALS,
                           show_progress_bar=True)

transformer_results = [t.user_attrs["result"]
                       for t in transformer_study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE]

n_pruned = sum(1 for t in transformer_study.trials
               if t.state == optuna.trial.TrialState.PRUNED)
print(f"\nTransformerConv — Completed: {len(transformer_results)}/{N_TRIALS}  "
      f"Pruned: {n_pruned}")
print(f"Best trial #{transformer_study.best_trial.number}  "
      f"val PR-AUC = {transformer_study.best_value:.4f}")
for k, v in transformer_study.best_trial.params.items():
    print(f"  {k} = {v}")

(WB03C3_DIR / "transformer_search_results.json").write_text(
    json.dumps(transformer_results, indent=2))


# %%
N_TRIALS_DENSE = 2 if SMOKE_TEST else 10

# GINEConv dense
def make_gine_dense_objective():
    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        num_layers = trial.suggest_categorical("num_layers", [2, 3])
        dropout    = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)
        lr         = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        pool       = trial.suggest_categorical("pool",    ["max", "attention"])
        jk_mode    = trial.suggest_categorical("jk_mode", ["none", "cat"])

        result, _ = train_and_evaluate(
            arch="gine", hidden_dim=hidden_dim, num_layers=num_layers,
            dropout=dropout, lr=lr, pool=pool, jk_mode=jk_mode,
            datasets=(train_ds_dense, val_ds_dense, test_ds_dense),
            edge_feat_dim=DENSE_FEAT_DIM, trial=trial)
        trial.set_user_attr("result", result)
        return result["best_val_pr_auc"]
    return objective

print(f"GINEConv dense ({DENSE_FEAT_DIM} features), {N_TRIALS_DENSE} trials")
gine_dense_study = optuna.create_study(
    study_name="wb03c3_gine_dense", direction="maximize",
    sampler=TPESampler(seed=RNG_SEED + 2),
    pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=10))
gine_dense_study.optimize(make_gine_dense_objective(),
                          n_trials=N_TRIALS_DENSE, show_progress_bar=True)

gine_dense_results = [t.user_attrs["result"]
                      for t in gine_dense_study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE]
print(f"  Best val PR-AUC: {gine_dense_study.best_value:.4f}")

# TransformerConv dense
def make_transformer_dense_objective():
    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        num_layers = trial.suggest_categorical("num_layers", [2, 3])
        heads      = trial.suggest_categorical("heads", [1, 2, 4])
        dropout    = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)
        lr         = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        pool       = trial.suggest_categorical("pool",    ["max", "attention"])
        jk_mode    = trial.suggest_categorical("jk_mode", ["none", "cat"])

        result, _ = train_and_evaluate(
            arch="transformer", hidden_dim=hidden_dim, num_layers=num_layers,
            heads=heads, dropout=dropout, lr=lr, pool=pool, jk_mode=jk_mode,
            datasets=(train_ds_dense, val_ds_dense, test_ds_dense),
            edge_feat_dim=DENSE_FEAT_DIM, trial=trial)
        trial.set_user_attr("result", result)
        return result["best_val_pr_auc"]
    return objective

print(f"\nTransformerConv dense ({DENSE_FEAT_DIM} features), {N_TRIALS_DENSE} trials")
transformer_dense_study = optuna.create_study(
    study_name="wb03c3_transformer_dense", direction="maximize",
    sampler=TPESampler(seed=RNG_SEED + 3),
    pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=10))
transformer_dense_study.optimize(make_transformer_dense_objective(),
                                 n_trials=N_TRIALS_DENSE, show_progress_bar=True)

transformer_dense_results = [t.user_attrs["result"]
                             for t in transformer_dense_study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE]
print(f"  Best val PR-AUC: {transformer_dense_study.best_value:.4f}")

# Save all dense results
(WB03C3_DIR / "gine_dense_results.json").write_text(
    json.dumps(gine_dense_results, indent=2))
(WB03C3_DIR / "transformer_dense_results.json").write_text(
    json.dumps(transformer_dense_results, indent=2))


# %%
all_results = gine_results + transformer_results
df = pd.DataFrame(all_results).sort_values("test_pr_auc", ascending=False)

cols = ["arch", "hidden_dim", "num_layers", "jk_mode", "pool", "heads",
        "dropout", "lr", "best_val_pr_auc", "test_pr_auc", "test_roc_auc",
        "test_f1", "test_precision", "test_recall", "n_params",
        "best_epoch", "wall_seconds"]
cols = [c for c in cols if c in df.columns]

print("="*90)
print("ALL COMPLETED TRIALS (by test PR-AUC)")
print("="*90)
print(df[cols].head(20).to_string(index=False, float_format="%.4f"))

# Per-architecture bests
for arch_name, study in [("GINEConv", gine_study),
                          ("TransformerConv", transformer_study)]:
    best = study.best_trial.user_attrs["result"]
    print(f"\n{arch_name} best:")
    print(f"  Test PR-AUC: {best['test_pr_auc']:.4f}  "
          f"ROC-AUC: {best['test_roc_auc']:.4f}  F1: {best['test_f1']:.4f}")
    print(f"  Params: {best['n_params']:,}")

# Dense comparison
print("\nDENSE FEATURE RESULTS:")
for name, study in [("GINEConv dense", gine_dense_study),
                     ("TransformerConv dense", transformer_dense_study)]:
    best = study.best_trial.user_attrs["result"]
    print(f"  {name}: test PR-AUC={best['test_pr_auc']:.4f}  "
          f"params={best['n_params']:,}")

# Full comparison
print("\n" + "="*90)
print("FULL BASELINE COMPARISON")
print("="*90)

gine_best = gine_study.best_trial.user_attrs["result"]
trans_best = transformer_study.best_trial.user_attrs["result"]
gine_d_best = gine_dense_study.best_trial.user_attrs["result"]
trans_d_best = transformer_dense_study.best_trial.user_attrs["result"]

for name, v in [
    ("Random (prevalence)",        0.023),
    ("LogReg pooled (Wb02)",       0.154),
    ("GLASS (Bellei 2024)",        0.208),
    ("SAGE tuned (Wb03)",          0.4848),
    ("GATv2 tuned (Wb03)",         0.4964),
    ("GINEConv (all 95)",          gine_best["test_pr_auc"]),
    (f"GINEConv (dense {DENSE_FEAT_DIM})", gine_d_best["test_pr_auc"]),
    ("TransformerConv (all 95)",   trans_best["test_pr_auc"]),
    (f"TransformerConv (dense {DENSE_FEAT_DIM})", trans_d_best["test_pr_auc"]),
]:
    print(f"  {name:35s} {v:.4f}  {'█' * int(v * 100)}")


# %%
# Pick the overall best across both architectures
best_row = df.iloc[0].to_dict()
best_arch = best_row["arch"]
print(f"Overall best: {best_arch}")
print(f"Re-training for checkpoint...")

res_best, state_best = train_and_evaluate(
    arch=best_row["arch"],
    hidden_dim=int(best_row["hidden_dim"]),
    num_layers=int(best_row["num_layers"]),
    dropout=float(best_row["dropout"]),
    lr=float(best_row["lr"]),
    pool=best_row["pool"],
    jk_mode=best_row["jk_mode"],
    heads=int(best_row.get("heads", 2)),
    verbose=True,
)

torch.save({
    "model_state_dict": state_best,
    "config": {k: best_row[k] for k in
               ["arch","hidden_dim","num_layers","dropout","lr",
                "pool","jk_mode","heads"] if k in best_row},
    "metrics": {k: res_best[k] for k in
                ["best_val_pr_auc","test_pr_auc","test_roc_auc",
                 "test_f1","test_precision","test_recall"]},
    "in_dim": IN_DIM,
    "edge_feat_dim": EDGE_FEAT_DIM,
}, WB03C3_DIR / "best_model.pt")
print(f"Saved → {WB03C3_DIR / 'best_model.pt'}  "
      f"PR-AUC={res_best['test_pr_auc']:.4f}")


# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for col, (name, study) in enumerate([("GINEConv", gine_study),
                                      ("TransformerConv", transformer_study)]):
    vals = [t.value for t in study.trials if t.value is not None]
    axes[0, col].plot(vals, "o-", ms=4, alpha=0.7)
    axes[0, col].axhline(study.best_value, color="red", ls="--", alpha=0.5,
                          label=f"Best: {study.best_value:.4f}")
    axes[0, col].axhline(0.4964, color="blue", ls=":", alpha=0.5,
                          label="GATv2 Wb03: 0.4964")
    axes[0, col].set(xlabel="Trial", ylabel="Val PR-AUC",
                      title=f"{name} — Optimisation History")
    axes[0, col].legend(fontsize=8)

    try:
        imp = optuna.importance.get_param_importances(study)
        ps = list(imp.keys())[:6]
        axes[1, col].barh(ps[::-1], [imp[p] for p in ps[::-1]])
        axes[1, col].set(xlabel="Importance",
                          title=f"{name} — Parameter Importance")
    except Exception as e:
        axes[1, col].text(0.5, 0.5, str(e), ha="center", va="center",
                           transform=axes[1, col].transAxes)

save_fig(WB03C3_DIR / "optuna_diagnostics.png")
plt.show()
print(f"Saved: {WB03C3_DIR / 'optuna_diagnostics.png'}")


