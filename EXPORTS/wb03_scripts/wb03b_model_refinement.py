# Auto-exported from wb03b_model_refinement.ipynb
# Run with: python wb03b_model_refinement.py [--smoke-test]

# %% [markdown]
# # Workbook 03b - Architectural Refinements
#
# **Objective:** Test JumpingKnowledge aggregation and attention-based global
# pooling on top of the best Wb03 configurations (GATv2 + GraphSAGE).
#
# | Modification | Options | Rationale |
# |:---|:---|:---|
# | JumpingKnowledge | none / cat / max | Leverage all layer representations |
# | GlobalAttention pool | max / attention | Learnable node importance weights |
# | LR scheduler | none / cosine | Smoother convergence |
#
# **Baselines:** GATv2 0.4964, GraphSAGE 0.4848, GLASS 0.208 (test PR-AUC).  
# **Budget:** 30-40 Optuna trials. GCN dropped (significantly underperformed).

# %% [markdown]
# ## 0. Configuration and Reproducibility
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
    SAGEConv, GATv2Conv,
    global_max_pool, GlobalAttention,
    JumpingKnowledge,
)
from torch_geometric.utils import to_undirected

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Reproducibility
RNG_SEED = 7
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# Paths
PROJECT_ROOT  = Path.cwd()
DATA_DIR      = PROJECT_ROOT / "DATA"
PROCESSED     = DATA_DIR / "processed"
ARRAYS_DIR    = PROCESSED / "arrays"
ARTIFACTS_DIR = PROCESSED / "artifacts"
PACK_DIR      = ARTIFACTS_DIR / "packed"
RESULTS_DIR   = PROJECT_ROOT / "results"
WB03B_DIR     = RESULTS_DIR / "wb03b"
WB03B_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Experiment control
import argparse

def _parse_args():
    p = argparse.ArgumentParser(description="Wb03 script")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a short smoke test (2 trials, 5 epochs).")
    return p.parse_known_args()[0]

ARGS = _parse_args()
SMOKE_TEST = ARGS.smoke_test

N_TRIALS      = 2   if SMOKE_TEST else 35
MAX_EPOCHS    = 5   if SMOKE_TEST else 80
PATIENCE      = 3   if SMOKE_TEST else 15
BATCH_SIZE    = 256
WANDB_PROJECT = "elliptic2-gnn"
WANDB_ENABLED = not SMOKE_TEST

print(f"SMOKE_TEST={SMOKE_TEST}  trials={N_TRIALS}  epochs={MAX_EPOCHS}")

# %%
try:
    from IPython.display import display, clear_output
except Exception:
    display = None
    clear_output = None

def safe_display(obj):
    if display:
        display(obj)
    else:
        print(obj)

def safe_clear_output():
    if clear_output:
        clear_output()

# %% [markdown]
# ## 1. Load Preprocessed Artifacts
# %%
# Core arrays
X = np.load(ARRAYS_DIR / "node_features.npy")
subgraph_labels = {
    int(k): v for k, v in
    json.loads((ARTIFACTS_DIR / "subgraph_labels.json").read_text()).items()
}
splits = json.loads((ARTIFACTS_DIR / "splits.json").read_text())

# Packed index arrays (Wb02)
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

print(f"Nodes: {X.shape[0]:,}  Features: {X.shape[1]}")
print(f"Subgraphs: {len(unique_cc):,}  "
      f"Suspicious: {sum(y_by_cc.values()):,} "
      f"({100*sum(y_by_cc.values())/len(unique_cc):.2f}%)")

# %% [markdown]
# ## 2. Dataset and Data Loaders
# %%
class Elliptic2SubgraphDataset(torch.utils.data.Dataset):
    """Lazy PyG dataset."""

    def __init__(self, ccids, make_undirected=True):
        self.ccids = np.asarray(ccids, dtype=np.int64)
        self.make_undirected = make_undirected

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
            src = torch.tensor([local[int(r)] for r in s], dtype=torch.long)
            dst = torch.tensor([local[int(r)] for r in t], dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
            if self.make_undirected:
                edge_index = to_undirected(edge_index)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        y = torch.tensor([y_by_cc[ccid]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y, ccId=ccid)


# Splits
train_cc = np.array(splits["train"], dtype=np.int64)
val_cc   = np.array(splits["val"],   dtype=np.int64)
test_cc  = np.array(splits["test"],  dtype=np.int64)

train_ds = Elliptic2SubgraphDataset(train_cc)
val_ds   = Elliptic2SubgraphDataset(val_cc)
test_ds  = Elliptic2SubgraphDataset(test_cc)

# Class weights (train only)
train_labels = torch.tensor([y_by_cc[int(c)] for c in train_cc])
n_pos = int(train_labels.sum().item())
n_neg = int((train_labels == 0).sum().item())
CLASS_WEIGHTS = torch.tensor(
    [1.0, n_neg / max(n_pos, 1)], dtype=torch.float32
).to(DEVICE)
IN_DIM = X.shape[1]  # 43

print(f"Class weights: licit={CLASS_WEIGHTS[0]:.1f}  suspicious={CLASS_WEIGHTS[1]:.1f}")
print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

# %% [markdown]
# ## 3. Model Architecture - SubgraphClassifierV2
#
# Extends Wb03's `SubgraphClassifier` with:
# - **JumpingKnowledge** (`cat` / `max` / `none`)
# - **GlobalAttention** pooling (learned gate network)

# %%
class SubgraphClassifierV2(nn.Module):
    """GNN + JumpingKnowledge + attention/max pooling."""

    def __init__(self, arch, in_dim, hidden_dim=128, num_layers=2,
                 dropout=0.1, pool="max", heads=1, jk_mode="none"):
        super().__init__()
        self.arch       = arch
        self.dropout    = dropout
        self.num_layers = num_layers
        self.jk_mode    = jk_mode
        self.pool_type  = pool

        # GNN layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            c_in = in_dim if i == 0 else hidden_dim

            if arch == "sage":
                self.convs.append(SAGEConv(c_in, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            elif arch == "gatv2":
                if i < num_layers - 1:
                    per_head = max(hidden_dim // heads, 1)
                    self.convs.append(
                        GATv2Conv(c_in, per_head, heads=heads, concat=True))
                    actual_out = per_head * heads
                    self.bns.append(nn.BatchNorm1d(actual_out))
                    # override for next layer
                    hidden_dim_next = actual_out  # noqa: F841 (used implicitly)
                else:
                    self.convs.append(
                        GATv2Conv(c_in, hidden_dim, heads=1, concat=False))
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
            else:
                raise ValueError(f"Unknown arch: {arch}")

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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
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


# Quick smoke test
_sample = train_ds[0]
_sample.batch = torch.zeros(_sample.x.size(0), dtype=torch.long)
for _arch in ["sage", "gatv2"]:
    for _jk in ["none", "cat", "max"]:
        for _pool in ["max", "attention"]:
            _m = SubgraphClassifierV2(
                arch=_arch, in_dim=IN_DIM, hidden_dim=128,
                num_layers=2, jk_mode=_jk, pool=_pool, heads=1)
            with torch.no_grad():
                _out = _m(_sample)
            _np = sum(p.numel() for p in _m.parameters())
            print(f"{_arch:6s} jk={_jk:4s} pool={_pool:9s} | "
                  f"params={_np:>8,} | out={tuple(_out.shape)}")
print("All model variants pass.")

# %% [markdown]
# ## 4. Training Infrastructure
# %%
@torch.no_grad()
def evaluate(model, loader):
    """Return (y_true, y_score) arrays."""
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
                       pool, heads, jk_mode, lr_scheduler="none",
                       trial=None, verbose=False):
    """Train one configuration. Returns (result_dict, best_state_dict)."""
    t0 = time.time()

    train_loader = PyGDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = PyGDataLoader(val_ds,   batch_size=512, shuffle=False)
    test_loader  = PyGDataLoader(test_ds,  batch_size=512, shuffle=False)

    model = SubgraphClassifierV2(
        arch=arch, in_dim=IN_DIM, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout, pool=pool,
        heads=heads, jk_mode=jk_mode,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = None
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    # W&B
    run = None
    if WANDB_ENABLED:
        run = wandb.init(
            project=WANDB_PROJECT,
            mode="online",
            name=f"wb03b_{arch}_{jk_mode}_{pool}",
            tags=["wb03b", arch, f"jk_{jk_mode}", f"pool_{pool}"],
            config=dict(notebook="wb03b", arch=arch, hidden_dim=hidden_dim,
                        num_layers=num_layers, dropout=dropout, lr=lr,
                        pool=pool, heads=heads, jk_mode=jk_mode,
                        lr_scheduler=lr_scheduler, n_params=n_params),
            reinit=True)

    best_val_pr, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
        avg_loss = epoch_loss / len(train_ds)

        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        yv, sv = evaluate(model, val_loader)
        vp = float(average_precision_score(yv, sv))
        vr = float(roc_auc_score(yv, sv))

        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  loss={avg_loss:.4f}"
                  f"  val_pr={vp:.4f}  val_roc={vr:.4f}")

        if run:
            wandb.log({"train/loss": avg_loss, "val/pr_auc": vp,
                        "val/roc_auc": vr, "lr": cur_lr, "epoch": epoch})

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
                  dropout=dropout, lr=lr, pool=pool, heads=heads,
                  jk_mode=jk_mode, lr_scheduler=lr_scheduler,
                  n_params=n_params, best_val_pr_auc=best_val_pr,
                  best_epoch=best_epoch, wall_seconds=wall, **tm)

    if run:
        wandb.log(dict(best_val_pr_auc=best_val_pr, best_epoch=best_epoch,
                        n_params=n_params, wall_seconds=wall, **tm))
        run.finish(quiet=True)

    return result, best_state

# %% [markdown]
# ## 5. Optuna Hyperparameter Search
# %%
def make_objective():
    """Optuna objective searching across SAGE + GATv2 with JK/attention."""

    def objective(trial):
        arch       = trial.suggest_categorical("arch",       ["sage", "gatv2"])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        num_layers = trial.suggest_categorical("num_layers", [2, 3])
        dropout    = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)
        lr         = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        jk_mode    = trial.suggest_categorical("jk_mode",       ["none", "cat", "max"])
        pool       = trial.suggest_categorical("pool",          ["max", "attention"])
        lr_sched   = trial.suggest_categorical("lr_scheduler",  ["none", "cosine"])

        heads = 1
        if arch == "gatv2":
            heads = trial.suggest_categorical("heads", [1, 2])

        result, _ = train_and_evaluate(
            arch=arch, hidden_dim=hidden_dim, num_layers=num_layers,
            dropout=dropout, lr=lr, pool=pool, heads=heads,
            jk_mode=jk_mode, lr_scheduler=lr_sched, trial=trial)

        trial.set_user_attr("result", result)
        return result["best_val_pr_auc"]

    return objective

# %% [markdown]
# ### Execute Optimisation
# %%
study = optuna.create_study(
    study_name="wb03b_arch_refinements",
    direction="maximize",
    sampler=TPESampler(seed=RNG_SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
study.optimize(make_objective(), n_trials=N_TRIALS, show_progress_bar=True)

all_results = [t.user_attrs["result"]
               for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE]

n_pruned = sum(1 for t in study.trials
               if t.state == optuna.trial.TrialState.PRUNED)
print(f"\nCompleted: {len(all_results)}/{N_TRIALS}  "
      f"Pruned: {n_pruned} ({100*n_pruned/max(N_TRIALS,1):.0f}%)")
print(f"Best trial #{study.best_trial.number}  val PR-AUC = {study.best_value:.4f}")
for k, v in study.best_trial.params.items():
    print(f"  {k} = {v}")

(WB03B_DIR / "search_results.json").write_text(json.dumps(all_results, indent=2))
print(f"Saved {len(all_results)} results → {WB03B_DIR / 'search_results.json'}")

# %% [markdown]
# ## 6. Controlled Ablation Studies
#
# Isolate each modification against the Wb03 best GATv2 baseline:
#
# | Name | JK | Pool | What it tests |
# |:---|:---|:---|:---|
# | wb03_repro | none | max | Wb03 reproduction |
# | attention_only | none | attention | Attention pooling effect |
# | jk_cat_only | cat | max | JumpingKnowledge effect |
# | both | cat | attention | Combined effect |

# %%
BASE = dict(arch="gatv2", hidden_dim=256, num_layers=2,
            dropout=0.20, lr=0.00075, heads=1)

ablations = [
    ("wb03_repro",      dict(jk_mode="none", pool="max",       lr_scheduler="none")),
    ("attention_only",  dict(jk_mode="none", pool="attention", lr_scheduler="none")),
    ("jk_cat_only",     dict(jk_mode="cat",  pool="max",       lr_scheduler="none")),
    ("both_jk_att",     dict(jk_mode="cat",  pool="attention", lr_scheduler="none")),
]

abl_results = []
for name, overrides in ablations:
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    cfg = {**BASE, **overrides}
    res, _ = train_and_evaluate(**cfg, verbose=True)
    res["ablation_name"] = name
    abl_results.append(res)
    print(f"  Val PR-AUC:  {res['best_val_pr_auc']:.4f}")
    print(f"  Test PR-AUC: {res['test_pr_auc']:.4f}  F1: {res['test_f1']:.4f}")

(WB03B_DIR / "ablation_results.json").write_text(json.dumps(abl_results, indent=2))
print(f"\nSaved {len(abl_results)} ablations.")

# %% [markdown]
# ## 7. Results Compilation
# %%
df = pd.DataFrame(all_results).sort_values("test_pr_auc", ascending=False)
cols = ["arch", "hidden_dim", "num_layers", "jk_mode", "pool", "lr_scheduler",
        "dropout", "lr", "best_val_pr_auc", "test_pr_auc", "test_roc_auc",
        "test_f1", "test_precision", "test_recall", "n_params",
        "best_epoch", "wall_seconds"]
cols = [c for c in cols if c in df.columns]

print("="*90)
print("ALL COMPLETED TRIALS (by test PR-AUC)")
print("="*90)
print(df[cols].head(20).to_string(index=False, float_format="%.4f"))

best = df.iloc[0]
print(f"\n{'='*90}")
print(f"BEST: {best['arch']} jk={best['jk_mode']} pool={best['pool']}")
print(f"  Test PR-AUC: {best['test_pr_auc']:.4f}  (Wb03 best: 0.4964)")
print(f"  ROC-AUC: {best['test_roc_auc']:.4f}  F1: {best['test_f1']:.4f}")
print(f"  Prec: {best['test_precision']:.4f}  Rec: {best['test_recall']:.4f}")
print(f"{'='*90}")

print("\nABLATION COMPARISON:")
adf = pd.DataFrame(abl_results)
print(adf[["ablation_name","jk_mode","pool","best_val_pr_auc",
           "test_pr_auc","test_f1","n_params"]].to_string(
               index=False, float_format="%.4f"))

print("\nBASELINES:")
for nm, v in [("Random", 0.023), ("LogReg (Wb02)", 0.154),
              ("GLASS (2024)", 0.208), ("SAGE tuned (Wb03)", 0.4848),
              ("GATv2 tuned (Wb03)", 0.4964),
              (f"Best Wb03b", float(best['test_pr_auc']))]:
    print(f"  {nm:30s} {v:.4f}  {'█' * int(v * 100)}")

# %% [markdown]
# ## 8. Save Best Model
# %%
best_cfg = df.iloc[0].to_dict()
print("Re-training best configuration for checkpoint save...")
res_best, state_best = train_and_evaluate(
    arch=best_cfg["arch"],
    hidden_dim=int(best_cfg["hidden_dim"]),
    num_layers=int(best_cfg["num_layers"]),
    dropout=float(best_cfg["dropout"]),
    lr=float(best_cfg["lr"]),
    pool=best_cfg["pool"],
    heads=int(best_cfg.get("heads", 1)),
    jk_mode=best_cfg["jk_mode"],
    lr_scheduler=best_cfg.get("lr_scheduler", "none"),
    verbose=True,
)

torch.save({
    "model_state_dict": state_best,
    "config": {k: best_cfg[k] for k in
               ["arch","hidden_dim","num_layers","dropout","lr",
                "pool","heads","jk_mode","lr_scheduler"]
               if k in best_cfg},
    "metrics": {k: res_best[k] for k in
                ["best_val_pr_auc","test_pr_auc","test_roc_auc",
                 "test_f1","test_precision","test_recall"]},
    "in_dim": IN_DIM,
}, WB03B_DIR / "best_model.pt")
print(f"Saved → {WB03B_DIR / 'best_model.pt'}  PR-AUC={res_best['test_pr_auc']:.4f}")

# %% [markdown]
# ## 9. Optuna Diagnostics
# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

vals = [t.value for t in study.trials if t.value is not None]
axes[0].plot(vals, "o-", ms=4, alpha=0.7)
axes[0].axhline(study.best_value, color="red", ls="--", alpha=0.5,
                label=f"Best: {study.best_value:.4f}")
axes[0].set(xlabel="Trial", ylabel="Val PR-AUC", title="Optimisation History")
axes[0].legend()

try:
    imp = optuna.importance.get_param_importances(study)
    ps, vs = list(imp.keys())[:8], [imp[p] for p in list(imp.keys())[:8]]
    axes[1].barh(ps[::-1], vs[::-1])
    axes[1].set(xlabel="Importance", title="Parameter Importance (fANOVA)")
except Exception as e:
    axes[1].text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                 transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig(WB03B_DIR / "optuna_diagnostics.png", dpi=150, bbox_inches="tight")
save_fig("results/fig_01.png")

# %% [markdown]
# ## 10. Summary
#
# Outputs saved to `results/wb03b/`:
# - `search_results.json` - all completed trial results
# - `ablation_results.json` - four controlled ablations
# - `best_model.pt` - checkpoint of best model
# - `optuna_diagnostics.png` - search visualisation
#
# Results feed into Wb03c2 (if attention pooling wins, use as default)
# and the explainability chapter (attention weights provide interpretability).

