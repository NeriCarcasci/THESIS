# Wb03 Decision Memo & Experiment Plan

**Date:** 2026-02-14
**Author:** Claude (ML Research Advisor)
**Scope:** Analysis of Wb03 Optuna-tuned GNN results → design of Wb03b (architectural refinements) and Wb03c2 (NNConv with edge features)

---

## Part 1 — Decision Memo

### 1.1 Wb03 Results Summary

| Architecture | Best Config | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Prec | Rec | F1 | Params |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **GATv2** | h=256, L=2, d=0.20, lr=7.5e-4, max pool, heads=1 | **0.5425** | **0.4964** | **0.9284** | 0.578 | 0.416 | 0.484 | 222K |
| GraphSAGE | h=128, L=3, d=0.10, lr=1.6e-3, max pool | 0.5328 | 0.4848 | 0.9234 | **0.635** | 0.400 | 0.491 | 94K |
| GCN | h=256, L=2, d=0.15, lr=1.7e-3, max pool | 0.4757 | 0.4199 | 0.9170 | 0.496 | 0.367 | 0.422 | 144K |

**Baselines for context:**

| Model | Test PR-AUC | Source |
|:--|--:|:--|
| Random | 0.023 | Prevalence |
| LogReg (pooled features) | 0.154 | Wb02 |
| GraphSAGE (default, untuned) | 0.401 | Wb02 |
| GLASS (best published) | 0.208 | Bellei et al. 2024 |

All three tuned models substantially outperform the published GLASS baseline (which used the full 49M-node background graph but no node features). GATv2 is the overall PR-AUC winner; GraphSAGE has the best precision at 63.5%.

### 1.2 What Worked

**Max pooling universally preferred.** All three architectures converged on `pool=max` over `pool=mean`. This makes domain sense: the most anomalous node embedding in a subgraph is a stronger AML signal than the average. This is a clean, thesis-worthy finding.

**Higher capacity helps GATv2 and GCN.** Both selected `hidden_dim=256`, while SAGE performed best at 128 with 3 layers. The attention mechanism in GATv2 evidently benefits from a wider representation space.

**Optuna + MedianPruner was highly efficient.** GATv2 saw a 67% pruning rate (20/30 trials killed early), saving substantial GPU time. The TPE sampler found the best GATv2 trial on trial #29 (the final one), indicating the search was still actively improving — a signal that more trials could potentially yield further gains.

**Low-moderate dropout is optimal.** Best values clustered tightly: 0.10 (SAGE), 0.15 (GCN), 0.20 (GATv2). Heavy regularisation (>0.3) consistently hurt.

**Learning rate sweet spot around 1e-3.** All three architectures converged to a narrow lr band (7.5e-4 to 1.7e-3), well within the log-uniform search range of [1e-4, 5e-3].

**Class-weighted CrossEntropyLoss (~43:1) works well.** The severe imbalance is handled effectively by the inverse-frequency weighting.

### 1.3 What Didn't Work / Limitations

**GATv2 heads=1 was optimal.** Multi-head attention (heads=2,4) did not help, likely because the tiny subgraphs (median 3 nodes) provide too few edges for diverse attention heads to learn distinct patterns. Each head sees essentially the same 2-3 edges.

**GCN significantly underperforms.** At test PR-AUC 0.420 vs 0.496 (GATv2) and 0.485 (SAGE), GCN's fixed degree-normalised aggregation is less effective than the learnable aggregation in SAGE or the attention mechanism in GATv2. This aligns with the literature — GCN's spectral assumptions are less suitable for heterogeneous small graphs.

**Mean pooling consistently underperformed.** Not a surprise given the task — AML detection benefits from highlighting extremes rather than averaging them out.

**Val-test gap exists but is moderate.** GATv2: 0.5425 → 0.4964 (−8.5%), SAGE: 0.5328 → 0.4848 (−9.0%). Some overfitting to validation distribution, but not alarming. Early stopping + dropout are doing their job.

### 1.4 Why — Root Causes

The performance ceiling around PR-AUC ~0.50 is driven by three structural factors:

1. **Tiny subgraphs** (median 3 nodes): Most subgraphs have minimal graph structure for GNNs to exploit. The model is largely learning from node features + one hop of neighbourhood context.

2. **Anonymised binned features**: The IP-protected feature encoding (ordinal bins with varying cardinality) limits the discriminative signal available per feature compared to raw continuous values.

3. **No edge features**: The current pipeline uses topology-only edges. The 95 edge features (transaction volume, fee, timestamp) represent untapped signal — this is what Wb03c2 addresses.

### 1.5 What to Try Next (Ranked)

| Rank | Experiment | Expected Impact | Effort | Notebook |
|:--|:--|:--|:--|:--|
| 1 | **Attention pooling** (GlobalAttention) | Medium-High | Low | Wb03b |
| 2 | **NNConv with 95 edge features** | High (if edge features are informative) | Medium | Wb03c2 |
| 3 | **JumpingKnowledge** (cat/max modes) | Medium | Low | Wb03b |
| 4 | **LR scheduler** (cosine annealing) | Low-Medium | Low | Wb03b |

**Not recommended:**

- Deeper networks (>3 layers): Subgraphs are too small; oversmoothing guaranteed.
- Focal loss: Class weighting already handles imbalance well.
- Deeper MLP head: Bottleneck is graph representation, not classifier capacity.

---

## Part 2 — Experiment Plan

### 2.0 Shared Protocol

**Objective metric:** Validation PR-AUC (maximise). Same as Wb03.

**Comparison baselines:**

| Baseline | Test PR-AUC | Role |
|:--|--:|:--|
| Best Wb03 overall (GATv2) | 0.4964 | Primary comparison |
| Best Wb03 SAGE | 0.4848 | Architecture-specific comparison |
| GLASS (Bellei) | 0.208 | Published state-of-art |

**A result "wins" if it exceeds GATv2's 0.4964 test PR-AUC.** Even if it doesn't, the architectural variants provide explainability value (attention pooling weights → interpretability).

**Common infrastructure:**
- Optuna TPE sampler with MedianPruner (n_startup_trials=5, n_warmup_steps=10)
- W&B logging: all trial hyperparameters, per-epoch val PR-AUC, test metrics for best trial
- W&B project: `elliptic2-gnn` (same as Wb03)
- Deterministic seeds (RNG_SEED=7)
- Early stopping: patience=15, max_epochs=80
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Test set evaluated exactly once with best validation config
- All results saved to `results/<notebook_name>/`

---

### 2.1 Wb03b — Architectural Refinements

**Goal:** Test whether JumpingKnowledge and attention pooling improve upon the best Wb03 models. Both modifications are independently motivated by dataset properties and feed directly into the explainability chapter.

**Architectures to test:** GATv2 and GraphSAGE only (GCN dropped — significantly weaker and not needed for thesis narrative).

**Budget:** 30–40 trials total (not per architecture). Recommended split: ~20 for GATv2 variants, ~15 for SAGE variants, with a few ablation trials.

#### Optuna Search Space

| Parameter | Range | Type | Rationale |
|:--|:--|:--|:--|
| `arch` | {sage, gatv2} | Categorical | Drop GCN — underperformed and adds nothing to thesis |
| `hidden_dim` | {128, 256} | Categorical | 64 consistently underperformed in Wb03; keep 128 and 256 |
| `num_layers` | {2, 3} | Categorical | Keep both; JK changes the optimal depth |
| `dropout` | [0.05, 0.30] | Float, step=0.05 | Narrowed from [0, 0.5]; best values were 0.10–0.20 |
| `lr` | [5e-4, 3e-3] | Log-uniform | Narrowed from [1e-4, 5e-3]; sweet spot was 7.5e-4 to 1.7e-3 |
| `jk_mode` | {none, cat, max} | Categorical | `none`=Wb03 baseline; `cat`=concatenate all layers; `max`=element-wise max |
| `pool` | {max, attention} | Categorical | Drop `mean` (never won); add `attention` (GlobalAttention) |
| `heads` | {1, 2} | Categorical (GATv2 only) | Keep 1 (was best), test 2 for JK interaction |
| `lr_scheduler` | {none, cosine} | Categorical | Test cosine annealing with warmup |

**Conditional parameters:**
- `heads` only sampled when `arch=gatv2`
- When `jk_mode=cat`, MLP input dimension = `hidden_dim × num_layers` (handle in model class)
- When `pool=attention`, instantiate `GlobalAttention` with a 2-layer gate network

#### Key Ablations (to log explicitly in W&B)

Run these specific configs regardless of Optuna sampling to ensure they appear in results:

1. **Wb03 reproduction** — best GATv2 config with `jk_mode=none, pool=max`: confirms Wb03 baseline within Wb03b infrastructure
2. **Attention pooling only** — best GATv2 + `pool=attention, jk_mode=none`: isolates pooling effect
3. **JK only** — best GATv2 + `jk_mode=cat, pool=max`: isolates JK effect
4. **Both** — best GATv2 + `jk_mode=cat, pool=attention`: combined effect

#### W&B Logging

Per trial:
- `wandb.config`: all hyperparameters including `jk_mode`, `pool`, `lr_scheduler`
- Per epoch: `train/loss`, `val/pr_auc`, `val/roc_auc`, `lr` (if scheduler active)
- End of trial: `best_val_pr_auc`, `test_pr_auc`, `test_roc_auc`, `test_f1`, `test_precision`, `test_recall`, `n_params`, `wall_seconds`, `best_epoch`
- Tags: `["wb03b", arch, jk_mode, pool]`

Summary table:
- `wandb.summary` with all test metrics for completed trials
- Custom W&B table comparing Wb03b best vs Wb03 baselines

---

### 2.2 Wb03c1 — Edge Feature Preprocessing

**Goal:** Extract and cache the 95 edge features from `background_edges.csv` into the existing pipeline format.

**This is a data preprocessing notebook, not a training notebook.** No Optuna, no W&B.

#### Pipeline

1. **Load `background_edges.csv`** — columns: `clId1, clId2, txId, feat#1 … feat#95`
   - This file covers the full 196M-edge background graph. Need to filter to the ~367K edges in the labelled subgraph universe.

2. **Filter to labelled edges** — inner join with the existing `edges.csv` (clId1, clId2, txId) to retain only edges in the labelled subgraph universe. Use `txId` as the join key since the same (clId1, clId2) pair can have multiple transactions.

3. **Handle edge multiplicity** — if multiple `txId` rows exist for the same (clId1, clId2) pair in the labelled universe, aggregate (mean/sum) or keep the first. Document the choice.

4. **Extract feat#1..feat#95 as edge_attr** — save as `processed/arrays/edge_features.npy` with shape `(n_edges, 95)`, aligned to the same edge order as `edge_index.npy`.

5. **Missingness policy** — same as node features: fill NaN with 0 (sentinel for "missing bin").

6. **Diagnostics** — feature distributions, missingness rates, correlation with node features.

7. **Cache to Parquet** — intermediate filtered file for fast reload.

**Critical alignment requirement:** The row order of `edge_features.npy` must match `edge_index.npy` exactly. Each row `i` of edge_features corresponds to edge `(edge_index[0, i], edge_index[1, i])`.

**Outputs:**
- `processed/arrays/edge_features.npy` — shape (367137, 95) or similar
- `processed/artifacts/edge_feature_columns.json` — column names
- `processed/parquet/background_edges_subset.parquet` — cached filtered edges
- `results/wb03c1/edge_feature_diagnostics.json` — summary statistics

---

### 2.3 Wb03c2 — NNConv with Edge Features

**Goal:** Test whether the 95 edge features improve subgraph classification when incorporated via NNConv edge-conditioned message passing.

**Budget:** 25–35 trials with an early stopping rule: if after 15 trials the best val PR-AUC is below the Wb03 GATv2 baseline (0.5425), stop and analyse why.

#### Model Architecture

Replace the GNN backbone with NNConv layers. The edge network is a small MLP that reads the 95-dim edge feature vector and produces a transformation matrix:

```
NNConv layer: h_v = σ( Σ_u  f_θ(e_uv) · h_u )
  where f_θ: R^95 → R^(in_dim × out_dim)
```

The edge network architecture is itself a hyperparameter:

```python
# Edge MLP maps 95-dim edge features → (in_dim × out_dim) weight matrix
edge_nn = nn.Sequential(
    nn.Linear(95, edge_hidden),
    nn.ReLU(),
    nn.Linear(edge_hidden, in_dim * out_dim)
)
conv = NNConv(in_dim, out_dim, edge_nn, aggr='add')
```

#### Optuna Search Space

| Parameter | Range | Type | Rationale |
|:--|:--|:--|:--|
| `hidden_dim` | {64, 128, 256} | Categorical | NNConv is more expensive; include 64 for feasibility |
| `num_layers` | {1, 2} | Categorical | Edge features may reduce the need for depth; test 1-layer |
| `edge_hidden` | {64, 128} | Categorical | Width of the edge MLP |
| `dropout` | [0.05, 0.35] | Float, step=0.05 | Slightly wider range — NNConv may need more regularisation |
| `lr` | [3e-4, 3e-3] | Log-uniform | Similar to Wb03 range |
| `pool` | {max, attention} | Categorical | Carry forward the best pooling options from Wb03b |
| `jk_mode` | {none, cat} | Categorical | Test JK interaction with edge features |
| `aggr` | {add, mean} | Categorical | NNConv aggregation function |

**Note on parameter count:** NNConv's edge network produces a `(in_dim × out_dim)` matrix per edge, making each layer much more expensive than SAGEConv/GATv2Conv. The `edge_hidden` parameter controls this. With `hidden_dim=256` and `edge_hidden=128`, the edge MLP alone has ~128×(95×256 + 256×128) parameters. Monitor GPU memory; may need to cap `hidden_dim` at 128 if OOM occurs.

#### Fair Comparison Protocol

To isolate the effect of edge features, run one explicit ablation:

1. **NNConv with edge features** — the main experiment
2. **NNConv with random edge features** — same architecture but `edge_attr` replaced with random noise of the same shape. If this performs comparably, the edge features aren't adding signal.
3. **NNConv with edge features, pool=max, jk=none** — simplest possible NNConv to establish a clean floor

#### Early Stopping Rule

After 15 completed (non-pruned) trials:
- If `best_val_pr_auc < 0.50` → stop. Edge features via NNConv aren't helping enough to justify continued search. Write up as a negative result (still valuable for thesis).
- If `best_val_pr_auc >= 0.50` → continue to 25–35 trials.
- If `best_val_pr_auc >= 0.55` → consider extending to 40 trials to thoroughly explore the promising region.

#### W&B Logging

Same as Wb03b, plus:
- `edge_hidden` in config
- `aggr` in config
- Per-trial: `n_edge_params` (parameters in edge network specifically)
- Tags: `["wb03c2", "nnconv", pool, jk_mode]`
- Ablation trials tagged: `["wb03c2", "ablation", "random_edges"]`

---

### 2.4 Experiment Timeline Estimate

| Notebook | Trials | Est. GPU Time | Dependencies |
|:--|--:|:--|:--|
| Wb03b | 30–40 | 8–16 hours | Wb03 artifacts only |
| Wb03c1 | N/A (preprocessing) | 10–30 min (CPU) | Raw `background_edges.csv` |
| Wb03c2 | 25–35 | 12–24 hours | Wb03c1 outputs |

**Recommended execution order:**

1. **Wb03c1** first (fast, CPU-only, unblocks Wb03c2)
2. **Wb03b** second (can run on GPU while you verify Wb03c1 outputs)
3. **Wb03c2** third (depends on Wb03c1; also benefits from Wb03b findings — e.g., if attention pooling wins in Wb03b, use it as the default pool in Wb03c2)

---

### 2.5 Success Criteria

| Outcome | Interpretation | Thesis Impact |
|:--|:--|:--|
| Wb03b beats Wb03 baseline | Architectural refinements improve AML detection | Strengthens Section 4.4 |
| Wb03b attention pooling wins | Learned node importance provides explainability signal | Directly feeds Section 4.5 |
| Wb03c2 beats Wb03 baseline | Edge features add discriminative signal | Major finding — first use of Elliptic2 edge features |
| Wb03c2 fails to improve | Edge features (as binned) don't help | Still publishable negative result |
| Both improve | Full pipeline: features + architecture + edges | Strongest possible model for explainability |

**Even if neither experiment improves PR-AUC**, the attention pooling weights from Wb03b provide interpretability value for the explainability chapter, and the edge feature analysis from Wb03c2 answers an open question from the Bellei et al. paper.

---

### 2.6 Open Questions (Non-Blocking)

These do not block experiment execution but should be verified:

1. **Edge multiplicity in background_edges.csv:** Does the same (clId1, clId2) pair appear multiple times with different txId values? If so, Wb03c1 needs an aggregation strategy. *Check during Wb03c1 execution.*

2. **Edge feature scale:** Are the 95 edge features also binned integers like the node features, or are they continuous? This affects whether you need normalisation in the edge MLP. *Check during Wb03c1 diagnostics.*

3. **Wb03b timing:** GATv2's best trial was #29/30 — the search hadn't converged. Consider running 20 GATv2 trials (not 15) in Wb03b to give Optuna more room, especially with the expanded search space.





Phase 1 — Create a fresh venv (so you can always nuke it)

From the instance terminal:

cd /workspace
python -m venv venv
source venv/bin/activate
python -m pip install -U pip setuptools wheel
Phase 2 — Install the known-good Blackwell-ready PyTorch

Install PyTorch 2.7.0 cu128 from the official PyTorch cu128 index:

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

Verify:

python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

You want to see:

torch 2.7.0

cuda 12.8

GPU is the 5090

no sm_120 warning (PyTorch 2.7 is where Blackwell support lands).

Phase 3 — Install PyG using wheels only (no compiling)

This is critical.

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install torch-geometric==2.7.0