# Auto-exported from wb03c2_e2_edge_featrures_conv.ipynb
# Run with: python wb03c2_e2_edge_featrures_conv.py [--smoke-test]

# %% [markdown]
# # Workbook 03c2 — NNConv with Edge Features
#
# **Objective:** Test whether the 95 edge features from `background_edges.csv`
# improve subgraph classification via NNConv edge-conditioned message passing.
#
# **Architecture:** NNConv replaces GNN backbone. Each edge has an MLP
# that maps its 95-dim feature vector to a weight matrix for message passing:
#
# ```
# edge_nn: R^95 → R^(in_dim × out_dim)
# conv = NNConv(in_dim, out_dim, edge_nn, aggr='add')
# ```
#
# **Early stopping rule:** If after 15 completed trials, best val PR-AUC < 0.50,
# stop (edge features not helping enough — still a publishable negative result).
#
# **Budget:** 25–35 trials.

# %% [markdown]
# ## 0. Configuration
# %%
import os
# Force non-interactive backend for script/tmux runs
os.environ.setdefault("MPLBACKEND", "Agg")
import json, random, time, warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from torch_geometric.nn import NNConv, global_max_pool, GlobalAttention, JumpingKnowledge
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
WB03C2_DIR    = RESULTS_DIR / "wb03c2"
WB03C2_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import argparse

def _parse_args():
    p = argparse.ArgumentParser(description="Wb03 script")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a short smoke test (2 trials, 5 epochs).")
    return p.parse_known_args()[0]

ARGS = _parse_args()
SMOKE_TEST = ARGS.smoke_test

N_TRIALS      = 2  if SMOKE_TEST else 30
MAX_EPOCHS    = 5  if SMOKE_TEST else 80
PATIENCE      = 3  if SMOKE_TEST else 15
BATCH_SIZE    = 256
WANDB_PROJECT = "elliptic2-gnn"
WANDB_ENABLED = not SMOKE_TEST

# Early stopping rule for the search itself
EARLY_STOP_CHECK_TRIAL = 15
EARLY_STOP_THRESHOLD   = 0.50

print(f"SMOKE_TEST={SMOKE_TEST}  trials={N_TRIALS}  epochs={MAX_EPOCHS}")
print(f"Early stop: check at trial {EARLY_STOP_CHECK_TRIAL}, "
      f"threshold={EARLY_STOP_THRESHOLD}")

# %%
try:
    from IPython.display import display, clear_output
except Exception:
    display = None
    clear_output = None

def safe_display(obj):
    if display:
        safe_display(obj)
    else:
        print(obj)

def safe_clear_output():
    if clear_output:
        clear_output()

# %% [markdown]
# ## 1. Load Data + Edge Features
# %%
# ── Core arrays ──────────────────────────────────────────────
X = np.load(ARRAYS_DIR / "node_features.npy")
subgraph_labels = {
    int(k): v for k, v in
    json.loads((ARTIFACTS_DIR / "subgraph_labels.json").read_text()).items()
}
splits = json.loads((ARTIFACTS_DIR / "splits.json").read_text())

# ── Packed arrays ────────────────────────────────────────────
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

# ── Edge features (from Wb03c1) ──────────────────────────────
EDGE_FEATURES = np.load(ARRAYS_DIR / "edge_features.npy")  # (n_edges, 95)
EDGE_FEAT_DIM = EDGE_FEATURES.shape[1]
print(f"Edge features loaded: {EDGE_FEATURES.shape}")

IN_DIM = X.shape[1]  # 43
print(f"Node features: {X.shape}  Edge features: {EDGE_FEATURES.shape}")
print(f"Subgraphs: {len(unique_cc):,}")

# Dense feature subset (from Wb03c1)
EDGE_FEATURES_DENSE = np.load(ARRAYS_DIR / "edge_features_dense.npy")
DENSE_FEAT_DIM = EDGE_FEATURES_DENSE.shape[1]
dense_meta = json.loads((ARTIFACTS_DIR / "edge_feature_dense_meta.json").read_text())

print(f"Dense features loaded: {EDGE_FEATURES_DENSE.shape} "
      f"(zero rate < {dense_meta['zero_threshold']})")

# %% [markdown]
# ## 2. Dataset with Edge Features
# %%
class Elliptic2EdgeFeatureDataset(torch.utils.data.Dataset):
    """PyG dataset with edge features for NNConv.

    Returns Data objects with:
        x:          [n_nodes, 43]
        edge_index: [2, n_edges] (undirected)
        edge_attr:  [n_edges, 95] (edge features, duplicated for reverse edges)
        y:          [1]
    """

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

            # Global edge indices for feature lookup
            global_edge_start = edge_ptr[ei]
            global_edge_end   = edge_ptr[ei + 1]

            src = torch.tensor([local[int(r)] for r in s], dtype=torch.long)
            dst = torch.tensor([local[int(r)] for r in t], dtype=torch.long)

            # Original directed edges
            edge_index_dir = torch.stack([src, dst], dim=0)

            # Edge features for directed edges
            e_feats = self.edge_features[global_edge_start:global_edge_end]
            if self.use_random_edges:
                e_feats = self._rng.randn(*e_feats.shape).astype(np.float32)
            edge_attr_dir = torch.from_numpy(e_feats.copy()).float()

            # Make undirected: add reverse edges with same features
            edge_index_rev = torch.stack([dst, src], dim=0)
            edge_index = torch.cat([edge_index_dir, edge_index_rev], dim=1)
            edge_attr  = torch.cat([edge_attr_dir, edge_attr_dir], dim=0)

            # Remove duplicate edges (to_undirected equivalent)
            # Use coalesce to handle duplicates
            from torch_geometric.utils import coalesce
            edge_index, edge_attr = coalesce(
                edge_index, edge_attr, reduce="mean"
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, EDGE_FEAT_DIM), dtype=torch.float32)

        y = torch.tensor([y_by_cc[ccid]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=y, ccId=ccid)


# ── Splits ───────────────────────────────────────────────────
train_cc = np.array(splits["train"], dtype=np.int64)
val_cc   = np.array(splits["val"],   dtype=np.int64)
test_cc  = np.array(splits["test"],  dtype=np.int64)

train_ds = Elliptic2EdgeFeatureDataset(train_cc)
val_ds   = Elliptic2EdgeFeatureDataset(val_cc)
test_ds  = Elliptic2EdgeFeatureDataset(test_cc)

# Random-edge ablation datasets
train_ds_rand = Elliptic2EdgeFeatureDataset(train_cc, use_random_edges=True)
val_ds_rand   = Elliptic2EdgeFeatureDataset(val_cc, use_random_edges=True)
test_ds_rand  = Elliptic2EdgeFeatureDataset(test_cc, use_random_edges=True)

# Dense-feature-only datasets (features with zero rate < 0.5)
train_ds_dense = Elliptic2EdgeFeatureDataset(train_cc, edge_features=EDGE_FEATURES_DENSE)
val_ds_dense   = Elliptic2EdgeFeatureDataset(val_cc,   edge_features=EDGE_FEATURES_DENSE)
test_ds_dense  = Elliptic2EdgeFeatureDataset(test_cc,  edge_features=EDGE_FEATURES_DENSE)

# ── Class weights ────────────────────────────────────────────
train_labels = torch.tensor([y_by_cc[int(c)] for c in train_cc])
n_pos = int(train_labels.sum().item())
n_neg = int((train_labels == 0).sum().item())
CLASS_WEIGHTS = torch.tensor(
    [1.0, n_neg / max(n_pos, 1)], dtype=torch.float32
).to(DEVICE)

print(f"Class weights: {CLASS_WEIGHTS.tolist()}")
print(f"Sample with edge features: {train_ds[0]}")
print(f"  edge_attr shape: {train_ds[0].edge_attr.shape}")

# %% [markdown]
# ## 3. NNConv Model
#
# The edge network maps each edge's 95-dim feature vector to a weight matrix
# that conditions the message passing:
#
# ```
# f_θ : R^95 → R^(in_dim × out_dim)
# msg_ij = f_θ(edge_attr_ij) · x_j
# ```

# %%
class NNConvClassifier(nn.Module):
    """NNConv-based subgraph classifier with edge-conditioned message passing."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=2,
                 edge_feat_dim=95, edge_hidden=64, dropout=0.1,
                 pool="max", jk_mode="none", aggr="add"):
        super().__init__()
        self.dropout    = dropout
        self.num_layers = num_layers
        self.jk_mode    = jk_mode
        self.pool_type  = pool

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            c_in  = in_dim if i == 0 else hidden_dim
            c_out = hidden_dim

            # Edge network: 95 → edge_hidden → (c_in * c_out)
            edge_nn = nn.Sequential(
                nn.Linear(edge_feat_dim, edge_hidden),
                nn.ReLU(),
                nn.Linear(edge_hidden, c_in * c_out),
            )
            self.convs.append(NNConv(c_in, c_out, edge_nn, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(c_out))

        # JumpingKnowledge
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

        if self.pool_type == "attention":
            g = self.pool_fn(x, batch)
        else:
            g = self.pool_fn(x, batch)

        g = F.relu(self.lin1(g))
        if self.dropout > 0:
            g = F.dropout(g, p=self.dropout, training=self.training)
        return self.lin2(g)


# ── Smoke test ───────────────────────────────────────────────
_sample = train_ds[0]
_sample.batch = torch.zeros(_sample.x.size(0), dtype=torch.long)
for _h in [64, 128]:
    for _pool in ["max", "attention"]:
        _m = NNConvClassifier(
            in_dim=IN_DIM, hidden_dim=_h, num_layers=2,
            edge_hidden=64, pool=_pool, jk_mode="none")
        with torch.no_grad():
            _out = _m(_sample)
        _np = sum(p.numel() for p in _m.parameters())
        print(f"hidden={_h:3d} pool={_pool:9s} | "
              f"params={_np:>9,} | out={tuple(_out.shape)}")

print("NNConv model passes smoke test.")

# %% [markdown]
# ## 4. Training Infrastructure
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
def train_and_evaluate_nnconv(*, hidden_dim, num_layers, edge_hidden,
                               dropout, lr, pool, jk_mode, aggr,
                               use_random_edges=False,
                               datasets=None, edge_feat_dim=None,
                               trial=None, verbose=False):
    """Train one NNConv configuration."""
    t0 = time.time()

    if datasets is not None:
        _train_ds, _val_ds, _test_ds = datasets
    elif use_random_edges:
        _train_ds, _val_ds, _test_ds = train_ds_rand, val_ds_rand, test_ds_rand
    else:
        _train_ds, _val_ds, _test_ds = train_ds, val_ds, test_ds

    if edge_feat_dim is None:
        edge_feat_dim = EDGE_FEAT_DIM

    train_loader = PyGDataLoader(_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = PyGDataLoader(_val_ds,   batch_size=512, shuffle=False)
    test_loader  = PyGDataLoader(_test_ds,  batch_size=512, shuffle=False)

    model = NNConvClassifier(
        in_dim=IN_DIM, hidden_dim=hidden_dim, num_layers=num_layers,
        edge_feat_dim=edge_feat_dim, edge_hidden=edge_hidden,
        dropout=dropout, pool=pool, jk_mode=jk_mode, aggr=aggr,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    # Count edge network params specifically
    n_edge_params = sum(
        sum(p.numel() for p in conv.nn.parameters())
        for conv in model.convs
    )

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # W&B
    run = None
    tag_list = ["wb03c2", "nnconv", f"pool_{pool}", f"jk_{jk_mode}"]
    if use_random_edges:
        tag_list += ["ablation", "random_edges"]
    if WANDB_ENABLED:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"wb03c2_nnconv_h{hidden_dim}_eh{edge_hidden}",
            tags=tag_list,
            config=dict(notebook="wb03c2", hidden_dim=hidden_dim,
                        num_layers=num_layers, edge_hidden=edge_hidden,
                        dropout=dropout, lr=lr, pool=pool, jk_mode=jk_mode,
                        aggr=aggr, n_params=n_params, n_edge_params=n_edge_params,
                        use_random_edges=use_random_edges),
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
                    print(f"  OOM at epoch {epoch} — skipping batch")
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

    result = dict(hidden_dim=hidden_dim, num_layers=num_layers,
                  edge_hidden=edge_hidden, dropout=dropout, lr=lr,
                  pool=pool, jk_mode=jk_mode, aggr=aggr,
                  n_params=n_params, n_edge_params=n_edge_params,
                  use_random_edges=use_random_edges,
                  best_val_pr_auc=best_val_pr,
                  best_epoch=best_epoch, wall_seconds=wall, **tm)

    if run:
        wandb.log(dict(best_val_pr_auc=best_val_pr, best_epoch=best_epoch,
                        n_params=n_params, wall_seconds=wall, **tm))
        run.finish(quiet=True)

    return result, best_state

# %% [markdown]
# ## 5. Optuna Search
# %%
class EarlyStopCallback:
    """Stop search if best val PR-AUC < threshold after N completed trials."""

    def __init__(self, check_trial, threshold):
        self.check_trial = check_trial
        self.threshold   = threshold
        self._completed  = 0

    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self._completed += 1
        if self._completed >= self.check_trial:
            if study.best_value < self.threshold:
                print(f"\n⚠ Early stop: best val PR-AUC {study.best_value:.4f}"
                      f" < {self.threshold} after {self._completed} trials")
                study.stop()


def make_objective():
    def objective(trial):
        hidden_dim  = trial.suggest_categorical("hidden_dim",  [64, 128, 256])
        num_layers  = trial.suggest_categorical("num_layers",  [1, 2])
        edge_hidden = trial.suggest_categorical("edge_hidden", [64, 128])
        dropout     = trial.suggest_float("dropout", 0.05, 0.35, step=0.05)
        lr          = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        pool        = trial.suggest_categorical("pool",    ["max", "attention"])
        jk_mode     = trial.suggest_categorical("jk_mode", ["none", "cat"])
        aggr        = trial.suggest_categorical("aggr",    ["add", "mean"])

        result, _ = train_and_evaluate_nnconv(
            hidden_dim=hidden_dim, num_layers=num_layers,
            edge_hidden=edge_hidden, dropout=dropout, lr=lr,
            pool=pool, jk_mode=jk_mode, aggr=aggr, trial=trial)

        trial.set_user_attr("result", result)
        return result["best_val_pr_auc"]

    return objective


print("NNConv search space:")
print("  hidden_dim:  [64, 128, 256]")
print("  num_layers:  [1, 2]")
print("  edge_hidden: [64, 128]")
print("  dropout:     [0.05, 0.35]")
print("  lr:          [3e-4, 3e-3]")
print("  pool:        [max, attention]")
print("  jk_mode:     [none, cat]")
print("  aggr:        [add, mean]")

# %% [markdown]
# ### Execute Search
# %%
study = optuna.create_study(
    study_name="wb03c2_nnconv",
    direction="maximize",
    sampler=TPESampler(seed=RNG_SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)

early_stop_cb = EarlyStopCallback(
    check_trial=EARLY_STOP_CHECK_TRIAL,
    threshold=EARLY_STOP_THRESHOLD,
)

study.optimize(
    make_objective(),
    n_trials=N_TRIALS,
    callbacks=[early_stop_cb],
    show_progress_bar=True,
)

all_results = [t.user_attrs["result"]
               for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE]

n_pruned = sum(1 for t in study.trials
               if t.state == optuna.trial.TrialState.PRUNED)
print(f"\nCompleted: {len(all_results)}/{len(study.trials)}  Pruned: {n_pruned}")
print(f"Best trial #{study.best_trial.number}  val PR-AUC = {study.best_value:.4f}")
for k, v in study.best_trial.params.items():
    print(f"  {k} = {v}")

(WB03C2_DIR / "search_results.json").write_text(json.dumps(all_results, indent=2))
print(f"Saved {len(all_results)} results → {WB03C2_DIR / 'search_results.json'}")

# %% [markdown]
# ## 6. Fair Comparison Ablations
#
# 1. **NNConv + real edges** — best config from search (already done)
# 2. **NNConv + random edges** — same architecture, random edge features
# 3. **NNConv + simple config** — pool=max, jk=none (floor performance)

# %%
# Use best params from search
best_params = study.best_trial.params

print("\n" + "="*60)
print("ABLATION 1: NNConv + random edge features")
print("="*60)
res_rand, _ = train_and_evaluate_nnconv(
    hidden_dim=best_params["hidden_dim"],
    num_layers=best_params["num_layers"],
    edge_hidden=best_params["edge_hidden"],
    dropout=best_params["dropout"],
    lr=best_params["lr"],
    pool=best_params["pool"],
    jk_mode=best_params["jk_mode"],
    aggr=best_params["aggr"],
    use_random_edges=True,
    verbose=True,
)
print(f"  Val PR-AUC:  {res_rand['best_val_pr_auc']:.4f}")
print(f"  Test PR-AUC: {res_rand['test_pr_auc']:.4f}")

print("\n" + "="*60)
print("ABLATION 2: NNConv simplest config (floor)")
print("="*60)
res_simple, _ = train_and_evaluate_nnconv(
    hidden_dim=128, num_layers=1, edge_hidden=64,
    dropout=0.1, lr=1e-3, pool="max", jk_mode="none", aggr="add",
    verbose=True,
)
print(f"  Val PR-AUC:  {res_simple['best_val_pr_auc']:.4f}")
print(f"  Test PR-AUC: {res_simple['test_pr_auc']:.4f}")

ablation_results = [
    {**study.best_trial.user_attrs["result"], "ablation": "best_real_edges"},
    {**res_rand, "ablation": "random_edges"},
    {**res_simple, "ablation": "simple_config"},
]
(WB03C2_DIR / "ablation_results.json").write_text(
    json.dumps(ablation_results, indent=2))
print("\nSaved ablations.")

# %% [markdown]
# ## 6b. Dense Feature Search (reduced edge features)
#
# The c1 diagnostics showed ~half the edge features are near-empty (zero rate > 50%).
# Here we run a separate 15-trial search using only the dense features. With fewer
# input dimensions the edge MLP can be smaller, so we include edge_hidden=32.
# %%
N_TRIALS_DENSE = 2 if SMOKE_TEST else 15

def make_dense_objective():
    def objective(trial):
        hidden_dim  = trial.suggest_categorical("hidden_dim",  [64, 128])
        num_layers  = trial.suggest_categorical("num_layers",  [1, 2])
        edge_hidden = trial.suggest_categorical("edge_hidden", [32, 64])
        dropout     = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)
        lr          = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        pool        = trial.suggest_categorical("pool",    ["max", "attention"])
        jk_mode     = trial.suggest_categorical("jk_mode", ["none", "cat"])
        aggr        = trial.suggest_categorical("aggr",    ["add", "mean"])

        result, _ = train_and_evaluate_nnconv(
            hidden_dim=hidden_dim, num_layers=num_layers,
            edge_hidden=edge_hidden, dropout=dropout, lr=lr,
            pool=pool, jk_mode=jk_mode, aggr=aggr,
            datasets=(train_ds_dense, val_ds_dense, test_ds_dense),
            edge_feat_dim=DENSE_FEAT_DIM,
            trial=trial)

        trial.set_user_attr("result", result)
        return result["best_val_pr_auc"]
    return objective

print(f"Dense feature search: {DENSE_FEAT_DIM} features, {N_TRIALS_DENSE} trials")
print(f"  edge_hidden includes 32 (smaller MLP for fewer inputs)")

dense_study = optuna.create_study(
    study_name="wb03c2_nnconv_dense",
    direction="maximize",
    sampler=TPESampler(seed=RNG_SEED + 1),
    pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=10),
)
dense_study.optimize(make_dense_objective(), n_trials=N_TRIALS_DENSE,
                     show_progress_bar=True)

dense_results = [t.user_attrs["result"]
                 for t in dense_study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]

n_pruned_d = sum(1 for t in dense_study.trials
                 if t.state == optuna.trial.TrialState.PRUNED)
print(f"\nCompleted: {len(dense_results)}/{N_TRIALS_DENSE}  Pruned: {n_pruned_d}")
print(f"Best trial #{dense_study.best_trial.number}  "
      f"val PR-AUC = {dense_study.best_value:.4f}")
for k, v in dense_study.best_trial.params.items():
    print(f"  {k} = {v}")

(WB03C2_DIR / "dense_search_results.json").write_text(
    json.dumps(dense_results, indent=2))
print(f"Saved {len(dense_results)} dense results")

# %% [markdown]
# ## 7. Results Compilation
# %%
df = pd.DataFrame(all_results).sort_values("test_pr_auc", ascending=False)
cols = ["hidden_dim", "num_layers", "edge_hidden", "jk_mode", "pool", "aggr",
        "dropout", "lr", "best_val_pr_auc", "test_pr_auc", "test_roc_auc",
        "test_f1", "test_precision", "test_recall",
        "n_params", "n_edge_params", "best_epoch", "wall_seconds"]
cols = [c for c in cols if c in df.columns]

print("="*90)
print("ALL COMPLETED TRIALS (by test PR-AUC)")
print("="*90)
print(df[cols].head(15).to_string(index=False, float_format="%.4f"))

best = df.iloc[0]
print(f"\n{'='*90}")
print(f"BEST NNConv: hidden={best['hidden_dim']} layers={best['num_layers']}"
      f" edge_hidden={best['edge_hidden']} pool={best['pool']}")
print(f"  Test PR-AUC: {best['test_pr_auc']:.4f}")
print(f"  ROC-AUC: {best['test_roc_auc']:.4f}  F1: {best['test_f1']:.4f}")
print(f"  Prec: {best['test_precision']:.4f}  Rec: {best['test_recall']:.4f}")
print(f"  Params: {best['n_params']:,.0f}  (edge net: {best['n_edge_params']:,.0f})")
print(f"{'='*90}")

# Ablation comparison
print("\nABLATION COMPARISON:")
adf = pd.DataFrame(ablation_results)
print(adf[["ablation", "best_val_pr_auc", "test_pr_auc",
           "test_f1", "n_params"]].to_string(index=False, float_format="%.4f"))

# Dense search best
dense_best = dense_study.best_trial.user_attrs["result"]
print(f"\nDENSE FEATURE SEARCH ({DENSE_FEAT_DIM} features):")
dense_df = pd.DataFrame(dense_results).sort_values("test_pr_auc", ascending=False)
print(dense_df[["hidden_dim","num_layers","edge_hidden","pool","jk_mode",
                 "best_val_pr_auc","test_pr_auc","test_f1","n_params"]
              ].head(10).to_string(index=False, float_format="%.4f"))
print(f"\nBest dense: test PR-AUC={dense_best['test_pr_auc']:.4f}  "
      f"params={dense_best['n_params']:,.0f}")

# Signal verification
real_pr = study.best_trial.user_attrs["result"]["test_pr_auc"]
rand_pr = res_rand["test_pr_auc"]
dense_pr = dense_best["test_pr_auc"]
delta = real_pr - rand_pr
print(f"\nEdge feature signal:")
print(f"  All 95 vs random:  {real_pr - rand_pr:+.4f}")
print(f"  Dense vs random:   {dense_pr - rand_pr:+.4f}")
print(f"  All 95 vs dense:   {real_pr - dense_pr:+.4f}")
if real_pr - rand_pr > 0.01:
    print("  -> Edge features carry meaningful signal")
if abs(real_pr - dense_pr) < 0.01:
    print("  -> Sparse features add little; dense subset is sufficient")

# Full baseline comparison
print("\nFULL BASELINE COMPARISON:")
for nm, v in [("Random", 0.023), ("LogReg (Wb02)", 0.154),
              ("GLASS (2024)", 0.208), ("SAGE tuned (Wb03)", 0.4848),
              ("GATv2 tuned (Wb03)", 0.4964),
              ("NNConv best", float(best['test_pr_auc'])),
              ("NNConv random edges", res_rand['test_pr_auc']),
              (f"NNConv dense ({DENSE_FEAT_DIM} feats)", dense_best['test_pr_auc'])]:
    print(f"  {nm:30s} {v:.4f}  {'█' * int(v * 100)}")

# %% [markdown]
# ## 8. Save Best Model
# %%
best_cfg = df.iloc[0].to_dict()
print("Re-training best NNConv for checkpoint...")
res_best, state_best = train_and_evaluate_nnconv(
    hidden_dim=int(best_cfg["hidden_dim"]),
    num_layers=int(best_cfg["num_layers"]),
    edge_hidden=int(best_cfg["edge_hidden"]),
    dropout=float(best_cfg["dropout"]),
    lr=float(best_cfg["lr"]),
    pool=best_cfg["pool"],
    jk_mode=best_cfg["jk_mode"],
    aggr=best_cfg["aggr"],
    verbose=True,
)

torch.save({
    "model_state_dict": state_best,
    "config": {k: best_cfg[k] for k in
               ["hidden_dim","num_layers","edge_hidden","dropout",
                "lr","pool","jk_mode","aggr"] if k in best_cfg},
    "metrics": {k: res_best[k] for k in
                ["best_val_pr_auc","test_pr_auc","test_roc_auc",
                 "test_f1","test_precision","test_recall"]},
    "in_dim": IN_DIM,
    "edge_feat_dim": EDGE_FEAT_DIM,
}, WB03C2_DIR / "best_model.pt")
print(f"Saved → {WB03C2_DIR / 'best_model.pt'}  PR-AUC={res_best['test_pr_auc']:.4f}")


# Also save best dense model if it beats the full model
if dense_best["test_pr_auc"] >= res_best["test_pr_auc"] - 0.005:
    print("\nDense model competitive — saving dense checkpoint too...")
    dense_cfg = dense_study.best_trial.params
    res_d, state_d = train_and_evaluate_nnconv(
        hidden_dim=dense_cfg["hidden_dim"],
        num_layers=dense_cfg["num_layers"],
        edge_hidden=dense_cfg["edge_hidden"],
        dropout=dense_cfg["dropout"],
        lr=dense_cfg["lr"],
        pool=dense_cfg["pool"],
        jk_mode=dense_cfg["jk_mode"],
        aggr=dense_cfg["aggr"],
        datasets=(train_ds_dense, val_ds_dense, test_ds_dense),
        edge_feat_dim=DENSE_FEAT_DIM,
        verbose=True,
    )
    torch.save({
        "model_state_dict": state_d,
        "config": dense_cfg,
        "metrics": {k: res_d[k] for k in
                    ["best_val_pr_auc","test_pr_auc","test_roc_auc",
                     "test_f1","test_precision","test_recall"]},
        "in_dim": IN_DIM,
        "edge_feat_dim": DENSE_FEAT_DIM,
        "dense_meta": dense_meta,
    }, WB03C2_DIR / "best_model_dense.pt")
    print(f"Saved → {WB03C2_DIR / 'best_model_dense.pt'}  "
          f"PR-AUC={res_d['test_pr_auc']:.4f}")

# %% [markdown]
# ## 9. Optuna Diagnostics
# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

vals = [t.value for t in study.trials if t.value is not None]
axes[0].plot(vals, "o-", ms=4, alpha=0.7)
axes[0].axhline(study.best_value, color="red", ls="--", alpha=0.5,
                label=f"Best: {study.best_value:.4f}")
axes[0].axhline(0.4964, color="blue", ls=":", alpha=0.5,
                label="GATv2 Wb03: 0.4964")
axes[0].set(xlabel="Trial", ylabel="Val PR-AUC",
            title="NNConv Optimisation History")
axes[0].legend()

try:
    imp = optuna.importance.get_param_importances(study)
    ps = list(imp.keys())[:8]
    axes[1].barh(ps[::-1], [imp[p] for p in ps[::-1]])
    axes[1].set(xlabel="Importance", title="Parameter Importance")
except Exception as e:
    axes[1].text(0.5, 0.5, str(e), ha="center", va="center",
                 transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig(WB03C2_DIR / "optuna_diagnostics.png", dpi=150, bbox_inches="tight")
save_fig("results/fig_01.png")

# %% [markdown]
# ## 10. Summary
#
# **Outputs saved to `results/wb03c2/`:**
# - `search_results.json` — all completed trials
# - `ablation_results.json` — real vs random edges comparison
# - `best_model.pt` — best NNConv checkpoint
# - `optuna_diagnostics.png`
#
# **Key findings:**
# - Whether edge features improve over node-only models
# - Signal verification via random-edge ablation
# - Comparison against Wb03 GATv2/SAGE baselines and published GLASS

