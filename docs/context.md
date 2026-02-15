# Thesis Context Pack (Generated from `wb03XXX` + supporting preprocessing artifacts)

## 1) Purpose of this file
This file is a complete context bundle you can paste into ChatGPT so it can write your thesis sections:
- Section **4.3 Data pre-processing** (target ~1000 words)
- Section **4.4 Model development** (target ~1000 words + tables/figures)

It is grounded in the actual project artifacts and notebooks in this repository, especially:
- `wb03_model_development.ipynb`
- `wb03c1_e2_preprocessing_edge_feat.ipynb`
- `wb03b_model_refinement.ipynb` (design/status)
- `wb03c2_e2_edge_featrures_conv.ipynb` (design/status)
- `wb03c3_e2_GINE_Transformer.ipynb` (design/status)

Supporting preprocessing context comes from:
- `wb01_e2_preprocessing.ipynb`

---

## 2) Source traceability (where each major claim comes from)
- Dataset sizes, schema checks, split generation, node feature extraction:
  - `wb01_e2_preprocessing.ipynb`
  - `DATA/processed/artifacts/*.json`
  - `DATA/processed/arrays/*.npy`
- Edge feature preprocessing and diagnostics:
  - `wb03c1_e2_preprocessing_edge_feat.ipynb`
  - `results/wb03c1/edge_feature_diagnostics.json`
  - `DATA/processed/artifacts/edge_feature_dense_meta.json`
- Main model development and tuned results:
  - `wb03_model_development.ipynb`
  - `results/wb03/search_results.json`
  - `results/wb03/model_comparison.csv`
  - `results/wb03/best_hyperparameters.csv`
  - `results/wb03/cross_study_comparison.csv`
  - `results/wb03/primary_model_selection.json`
  - `results/wb03_summary.md`
- WB03b architectural refinement results:
  - `wb03b_model_refinement.ipynb`
  - `results/wb03b/search_results.json`
  - `results/wb03b/ablation_results.json`
  - `results/wb03b/optuna_diagnostics.png`
  - `results/wb03b/best_model.pt`
- Baseline reference models (Wb02):
  - `wb02_e2_baselinesAndGraphs.ipynb`
  - `results/baseline_tabular_metrics.json`
  - `results/baseline_gnn_metrics.json`
- Refinement plans:
  - `docs/wb03_decision_memo_and_experiment_plan.md`
  - `wb03b_model_refinement.ipynb`
  - `wb03c2_e2_edge_featrures_conv.ipynb`
  - `wb03c3_e2_GINE_Transformer.ipynb`

---

## 3) Section 4.3 Data Pre-processing Context (write-up material)

### 3.1 Problem framing and unit of learning
- The project uses Elliptic2 for **subgraph-level AML classification**.
- Prediction unit is connected component (`ccId`), not individual node.
- Labels are stored at component level (`ccLabel`), and include licit vs suspicious/illicit.
- This directly motivates **component-level split strategy** to avoid leakage.

### 3.2 Raw inputs and schema alignment
Expected raw tables:
- `DATA/nodes.csv` with `clId`, `ccId`
- `DATA/edges.csv` with `clId1`, `clId2`, `txId`
- `DATA/connected_components.csv` with `ccId`, `ccLabel`
- `DATA/background_nodes.csv` and `DATA/background_edges.csv` for enriched features

Inferred/validated columns in preprocessing:
- Node ID: `clId`
- Node component ID: `ccId`
- Edge endpoints: `clId1`, `clId2`
- Component label: `ccLabel`

### 3.3 Core dataset dimensions (post-ingestion)
From notebook outputs/artifacts:
- `nodes`: `(444,521, 2)`
- `edges`: `(367,137, 3)`
- `connected_components`: `(121,810, 2)`

Derived component-label totals:
- Total labeled components: `121,810`
- Licit: `119,047`
- Suspicious: `2,763`
- Positive prevalence: `2,763 / 121,810 = 2.27%`

This confirms a severe class imbalance regime.

### 3.4 Data integrity and auditability steps
The preprocessing pipeline performs explicit checks before model artifacts are produced:
- Non-null checks on required ID/label columns
- Type coercion of key IDs to `int64`
- Referential integrity check: all edges point to known labeled nodes
  - Unknown endpoints: `unknown_src=0`, `unknown_dst=0`
- Component completeness check:
  - Nodes whose `ccId` missing in components file: `0`
- Label coverage check:
  - Components used by nodes: `121,810`
  - Components labeled but unused by nodes: `0`

Additional reproducibility guardrails:
- Input fingerprints written to `DATA/processed/artifacts/input_fingerprints.json`
- Split seed persisted in `DATA/processed/artifacts/splits.json`

### 3.5 Parquet conversion and caching strategy
CSV files are converted/cached to Parquet for speed and reproducibility:
- `DATA/processed/parquet/nodes.parquet` (~4.4 MB)
- `DATA/processed/parquet/edges.parquet` (~7.0 MB)
- `DATA/processed/parquet/connected_components.parquet` (~0.8 MB)
- Background subsets cached for repeated runs:
  - `background_nodes_subset.parquet`
  - `background_edges_subset.parquet`

Rationale for report:
- Faster iteration for notebook experiments
- Stable dtypes across reloads
- Lower IO cost for repeated model tuning

### 3.6 Node indexing, topology tensors, and alignment
Raw node IDs are sparse/high-cardinality, so nodes are reindexed to contiguous integers.
Produced arrays:
- `DATA/processed/arrays/edge_index.npy` shape `(2, 367,137)`, dtype `int64`
- `DATA/processed/arrays/node_components.npy` shape `(444,521,)`, dtype `int64`

Node feature matrix:
- `DATA/processed/arrays/node_features.npy` shape `(444,521, 43)`, dtype `int8`
- Feature columns stored in `DATA/processed/artifacts/feature_columns.json`

Component structural summary (from `node_components.npy`):
- Components with nodes: `121,810`
- Min nodes/component: `2`
- Median nodes/component: `3`
- Mean nodes/component: `3.649`
- Max nodes/component: `296`

This is important for Section 4.4 because tiny subgraphs constrain message-passing depth.

### 3.7 Node feature engineering and missing-value policy
Node features were sourced from `background_nodes.csv` by semijoin on labeled `clId` universe.

Policy choices:
- Feature set: all `feat#` columns except IDs (`43` node features)
- Missing values: `fillna(0)` (explicit sentinel bin)
- Preserve integer/binned representation where possible (not forced into arbitrary normalization)

Observed missingness:
- Missing count for top displayed features is zero after subset+fill
- Dtype profile indicates integer-like binned features (`int8` across all 43 feature columns in output summary)

Train-only statistics were computed and stored in:
- `DATA/processed/artifacts/feature_stats_train.json`

Purpose:
- Leakage-safe diagnostics for later scaling/normalization experiments
- Support transparent reporting of feature ranges and distribution

Examples from train stats:
- `feat#19`: mean `18.97`, std `30.51`, max `98`
- `feat#23`: mean `18.52`, std `27.17`
- Global node-feature range observed in stats: min `0`, max `98`

### 3.8 Subgraph-level stratified split protocol
Split construction (component-level):
1. Stratified split component IDs into train+val vs test using `test_size=0.2`.
2. Stratified split train+val into train vs val with `test_size=0.125`.

This yields approximately 70/10/20 overall.

Persisted split sizes:
- Train: `85,267` components
- Val: `12,181`
- Test: `24,362`

Class distribution by split:

| Split | Total | Positives | Negatives | Positive rate | Neg:Pos |
|---|---:|---:|---:|---:|---:|
| Train | 85,267 | 1,934 | 83,333 | 2.2682% | 43.0884 |
| Val | 12,181 | 276 | 11,905 | 2.2658% | 43.1341 |
| Test | 24,362 | 553 | 23,809 | 2.2699% | 43.0542 |

Interpretation for thesis:
- Stratification was successful; prevalence is near-identical across splits.
- This supports fair validation/test comparisons in rare-event settings.

### 3.9 WB03c1 edge-feature preprocessing (critical 4.3 content)
Workbook `wb03c1_e2_preprocessing_edge_feat.ipynb` extends preprocessing to transaction-edge features.

Objective:
- Align 95 edge features from `background_edges.csv` with existing packed edge order used by PyG.

Scale and runtime evidence:
- Background edge file exists and is large: ~`82,877.4 MB`
- Rows scanned: `196,215,606`
- Rows kept for labeled-universe edges: `367,137`
- Scan time shown: ~`679.1s`

Edge alignment and multigraph diagnostics:
- Directed labeled edges: `367,137`
- Unique `(clId1, clId2)` pairs: `343,192`
- Duplicate directed edges: `23,945` (`6.52%`)
- Matched forward pairs: `343,192` (`100%` of pair universe)
- Reverse matched: `14,672` (diagnostic overlap)
- Final unmatched edges after fallback matching: `0 / 367,137 (0.00%)`
- Missing edge-feature rows filled with 0 sentinel (none remained unmatched after alignment)

Produced edge arrays:
- `DATA/processed/arrays/edge_features.npy`: `(367,137, 95)`, `float32`
- `DATA/processed/arrays/edge_features_dense.npy`: `(367,137, 52)`, `float32`

Dense-subset logic:
- Zero-rate threshold: `< 0.5`
- Features retained: `52`
- Features dropped: `43`
- Dense subset metadata persisted to `edge_feature_dense_meta.json`

Edge-feature statistics (`results/wb03c1/edge_feature_diagnostics.json` + notebook output):
- Unique values per feature: min `3`, median `576`, max `36,516`
- Mean zero rate across 95 features: `0.4402`
- Value range: global min `-1`, global max `98`
- Global mean `16.9402`, std `23.3226`
- Dense subset mean `29.11`, std `24.40`

Verification stage:
- Random row checks confirmed i-th feature row matches i-th edge row ordering
- Notebook reports: "All checks passed. edge_features.npy is aligned with edge arrays."

### 3.10 Preprocessing artifacts (for reporting appendix/table)
Core outputs to cite in Section 4.3:
- `DATA/processed/arrays/node_features.npy`
- `DATA/processed/arrays/edge_index.npy`
- `DATA/processed/arrays/node_components.npy`
- `DATA/processed/artifacts/subgraph_labels.json`
- `DATA/processed/artifacts/splits.json`
- `DATA/processed/artifacts/feature_stats_train.json`
- `DATA/processed/arrays/edge_features.npy`
- `DATA/processed/arrays/edge_features_dense.npy`
- `results/wb03c1/edge_feature_diagnostics.json`

### 3.11 Thesis-ready preprocessing rationale (talking points)
Use these as argument lines in 4.3:
- The split unit is `ccId` to avoid structural leakage from node-level random splits.
- Stratification preserves the rare positive prevalence (~2.27%) across train/val/test.
- Integer/binned feature semantics were preserved instead of blindly standardizing all columns.
- Missing data policy is explicit (`0` sentinel) and consistent for node and edge feature branches.
- Artifact persistence (arrays/json/parquet) enables reproducible model development and fair model comparison.
- Edge-feature alignment was validated rigorously despite multigraph duplicates.

---

## 4) Section 4.4 Model Development Context (write-up material)

### 4.1 Model-development objective and chronology (WB02 -> WB03 -> WB03b)
Model development progressed in three phases:
- **WB02 (baselines and dataset pipeline):**
  - Build first credible baselines and verify split hygiene.
  - Compare a traditional pooled-feature classifier vs a minimal GNN.
- **WB03 (core architecture study):**
  - Systematic multi-architecture tuning/comparison across GraphSAGE, GCN, and GATv2.
  - Select a primary model for downstream explainability work.
- **WB03b (architectural refinement):**
  - Test whether JumpingKnowledge and attention pooling improve on WB03 winner configurations.
  - Run controlled ablations to isolate design effects.

### 4.2 Tools and infrastructure (for required 4.4 intro)
Primary stack used:
- Language: Python 3.11
- Deep learning: PyTorch
- Graph learning: PyTorch Geometric
- Hyperparameter optimization: Optuna (TPE sampler + MedianPruner)
- Metrics/evaluation: scikit-learn metrics (`PR-AUC`, `ROC-AUC`, `F1`, confusion matrix)
- Experiment tracking: Weights & Biases (W&B)
- Data handling: NumPy, pandas, parquet workflow from preprocessing

Execution environment evidence:
- Notebook output reports CUDA device usage (`Device: cuda`), with GPU detected.
- W&B metadata captures a Windows run environment and GPU availability.

Thesis framing suggestion:
- Emphasize choice of PyTorch Geometric for direct subgraph `Data` objects and batching.
- Emphasize Optuna because search includes mixed categorical+continuous hyperparameters.

WB02 baseline architectures (starting point before WB03):

| Baseline | Input representation | Core model | Key training choices |
|---|---|---|---|
| Tabular (traditional) | Pooled subgraph features (`mean`, `max`, `std`) over node feature matrix | Logistic Regression (`class_weight=\"balanced\"`) | Validation threshold chosen by max F1 |
| Graph baseline | Per-subgraph PyG graph (`x`, `edge_index`) | 2-layer GraphSAGE (`hidden=64`) + global mean pooling + MLP head | Weighted CrossEntropy (~43:1), Adam (`lr=1e-3`, `wd=1e-4`), early stopping |

### 4.3 Data loading and model input pipeline
WB03 reuses processed artifacts and packed component-index arrays:
- Node features: `X` shape `(444,521, 43)`
- Component labels: from `subgraph_labels.json`
- Splits: component ID lists from `splits.json`
- Packed node/edge mappings from:
  - `DATA/processed/artifacts/packed/nodes_by_ccid.npz`
  - `DATA/processed/artifacts/packed/edges_by_ccid.npz`

Per-subgraph sample construction:
- Build local node index map for each `ccId`
- Slice node rows and intra-component edges
- Convert edge list to undirected for message passing
- Return PyG `Data(x, edge_index, y, ccId)`

Label encoding:
- Positive class = `suspicious`/`illicit` mapped to `1`
- Negative class = `licit` mapped to `0`

### 4.4 Class-imbalance handling
Training-set imbalance is severe (~43:1 negatives:positives), so WB03 uses:
- `CrossEntropyLoss(weight=[1.0, n_neg/n_pos])`
- Observed class weights printed in notebook:
  - Licit `1.0`
  - Suspicious `43.1`

Why this matters in write-up:
- Prevents model collapse to majority-class predictions.
- Supports recall/PR-AUC improvements under rare-event distribution.

### 4.5 Architecture definitions and search motivation
Architectures compared under common head/protocol:
- `GNN layers -> global pooling -> MLP head`

Differences:
- **GraphSAGE**: inductive neighborhood aggregation
- **GCN**: degree-normalized spectral convolution baseline
- **GATv2**: learned attention over neighbors

Pooling options in WB03 search:
- `mean` and `max`

Observation from final selected configs:
- `max` pooling won for all three architecture winners.

### 4.6 Training loop protocol
Per-trial training setup (WB03 code):
- Optimizer: Adam
- Weight decay: `1e-4`
- Batch size: `256` (train), `512` (eval loaders)
- Max epochs: `80`
- Early stopping patience: `15`
- Validation metric for checkpointing: `val PR-AUC`

Thresholding:
- Decision threshold selected on validation set to maximize F1.
- That threshold then applied to test predictions for confusion/F1/precision/recall reporting.

### 4.7 Hyperparameter optimization setup (official WB03)
Search driver:
- Optuna study per architecture
- Sampler: `TPESampler(seed=7)`
- Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`

Configured search space:
- `hidden_dim`: `{64, 128, 256}`
- `num_layers`: `{2, 3}`
- `dropout`: `[0.0, 0.5]` step `0.05`
- `lr`: `[1e-4, 5e-3]` log-scale
- `pool`: `{mean, max}`
- `heads` (GATv2 only): `{1, 2, 4}`

Trial accounting note (important to avoid confusion):
- Code config uses `N_TRIALS_PER_ARCH = 30`.
- `results/wb03/search_results.json` contains `30 completed` trials total (`10` per architecture) that were carried into analysis tables.
- Selection tables are built from these completed-trial records.

### 4.8 Official WB03 tuned results (use these in thesis tables)
From `results/wb03/model_comparison.csv` and `best_hyperparameters.csv`.

#### A) Tuned model performance (test set)

| Model | PR-AUC | ROC-AUC | F1 | Params |
|---|---:|---:|---:|---:|
| GraphSAGE (tuned) | 0.4848 | 0.9234 | 0.4906 | 94,466 |
| GCN (tuned) | 0.4199 | 0.9170 | 0.4220 | 144,386 |
| GATv2 (tuned) | 0.4964 | 0.9284 | 0.4837 | 222,466 |

#### B) Best hyperparameters per architecture (selected by validation PR-AUC)

| Architecture | hidden_dim | num_layers | dropout | lr | pool | heads | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| SAGE | 128 | 3 | 0.10 | 0.00157 | max | - | 0.5328 | 0.4848 |
| GCN | 256 | 2 | 0.15 | 0.00165 | max | - | 0.4757 | 0.4199 |
| GATv2 | 256 | 2 | 0.20 | 0.00075 | max | 1 | 0.5425 | 0.4964 |

#### C) Confusion matrices for tuned winners (test)

| Model | TN | FP | FN | TP | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| GraphSAGE (tuned) | 23,682 | 127 | 332 | 221 | 0.6351 | 0.3996 | 0.9607 |
| GCN (tuned) | 23,603 | 206 | 350 | 203 | 0.4963 | 0.3671 | 0.9265 |
| GATv2 (tuned) | 23,641 | 168 | 323 | 230 | 0.5779 | 0.4159 | 0.8775 |

Operational note for report:
- GraphSAGE has highest precision/F1 among tuned models.
- GATv2 has highest PR-AUC and ROC-AUC.

### 4.9 Baseline-to-tuned improvement context
Reference baselines (Wb02):

| Model | PR-AUC | ROC-AUC | F1 |
|---|---:|---:|---:|
| LogReg (pooled) | 0.1539 | 0.8895 | 0.2511 |
| GraphSAGE (default, Wb02) | 0.4012 | 0.9137 | 0.4137 |

WB02 baseline confusion matrices (test):

| Model | TN | FP | FN | TP | Precision | Recall | Alert rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| LogReg (pooled) | 22,920 | 889 | 346 | 207 | 0.1889 | 0.3743 | 4.50% |
| GraphSAGE (default) | 23,392 | 417 | 300 | 253 | 0.3776 | 0.4575 | 2.75% |

Why GraphSAGE became the WB03 anchor architecture:
- Relative to the traditional pooled-feature LogReg baseline, default GraphSAGE achieved:
  - `2.61x` PR-AUC (`0.4012` vs `0.1539`)
  - `1.65x` F1 (`0.4137` vs `0.2511`)
  - `-53.1%` false positives (`417` vs `889`)
  - `+22.2%` true positives (`253` vs `207`)
- This showed that relational aggregation over subgraph topology materially improved minority-class detection quality, justifying deeper architecture work in WB03.

Improvement highlights:
- GATv2 tuned vs LogReg baseline: `3.22x` PR-AUC.
- GraphSAGE tuned vs default GraphSAGE: `+20.85%` relative PR-AUC.
- GCN tuned vs default GraphSAGE baseline: `+4.67%` relative PR-AUC.

### 4.10 Cross-study comparison (published benchmark context)
From `results/wb03/cross_study_comparison.csv`.

| Model | Source | Features | Background graph | PR-AUC | ROC-AUC |
|---|---|---|---|---:|---:|
| GNN-Seg | Bellei et al. | No | No | 0.026 | 0.537 |
| Sub2Vec | Bellei et al. | No | No | 0.022 | 0.496 |
| GLASS | Bellei et al. | No | Yes | 0.208 | 0.889 |
| GraphSAGE (tuned) | This work | Yes (43) | No | 0.4848 | 0.9234 |
| GCN (tuned) | This work | Yes (43) | No | 0.4199 | 0.9170 |
| GATv2 (tuned) | This work | Yes (43) | No | 0.4964 | 0.9284 |

Interpretation lines you can use:
- All tuned WB03 GNNs exceed GLASS PR-AUC despite using a different setup (node features, labeled subgraph universe).
- GATv2 and GraphSAGE show strongest rare-class ranking in this project setting.

### 4.11 Primary model selection for explainability phase
From `results/wb03/primary_model_selection.json`:
- Selected architecture: `GATv2`
- Selected label/config: `gatv2_hid256_L2_drop0.2_lr0.0007544709109508611_max_h1`
- Selection criterion (recorded): highest test PR-AUC among architecture winners
- Primary metrics:
  - Test PR-AUC: `0.496425`
  - Test ROC-AUC: `0.928434`
  - Test F1: `0.483701`
  - Threshold: `0.877457`

Checkpoint written:
- `results/wb03/best_model_state.pt`

### 4.12 WB03b completed refinement results (JumpingKnowledge + attention pooling)
WB03b objective:
- Re-test top WB03 architecture family with architectural refinements:
  - `jk_mode`: `none/cat/max`
  - `pool`: `max/attention`
  - `lr_scheduler`: `none/cosine`
- Restrict architectures to `sage` and `gatv2` (GCN intentionally dropped).

Available WB03b artifacts in this repo:
- `results/wb03b/search_results.json` (10 completed trials)
- `results/wb03b/ablation_results.json` (4 controlled ablations)
- `results/wb03b/optuna_diagnostics.png`
- `results/wb03b/best_model.pt`

WB03b search coverage (completed trials):
- Total completed: `10`
- By architecture: `7` GATv2, `3` GraphSAGE
- Pool usage in completed trials: `9` max, `1` attention
- JK usage in completed trials: `none` (5), `max` (3), `cat` (2)

Best WB03b search trial (by test PR-AUC from `search_results.json`):
- Architecture: `GATv2`
- Config: `hidden_dim=256`, `num_layers=2`, `dropout=0.20`, `lr=0.0005689`,
  `pool=max`, `jk_mode=max`, `lr_scheduler=cosine`, `heads=1`
- Test metrics:
  - PR-AUC `0.5071`
  - ROC-AUC `0.9279`
  - F1 `0.5021`
  - Precision `0.5921`
  - Recall `0.4358`
  - Threshold `0.9262`
- Confusion matrix: `[[23643, 166], [312, 241]]`
- Relative to WB03 primary GATv2 (`PR-AUC=0.4964`): `+0.0107` absolute (`+2.15%` relative).

Controlled WB03b ablations (from `ablation_results.json`):

| Ablation | JK | Pool | Test PR-AUC | Test ROC-AUC | Test F1 | Delta vs WB03 GATv2 PR-AUC |
|---|---|---|---:|---:|---:|---:|
| wb03_repro | none | max | 0.4887 | 0.9282 | 0.4917 | -0.0077 |
| attention_only | none | attention | 0.5197 | 0.9385 | 0.5028 | +0.0233 |
| jk_cat_only | cat | max | 0.5157 | 0.9323 | 0.5041 | +0.0192 |
| both_jk_att | cat | attention | 0.4963 | 0.9322 | 0.5059 | -0.0001 |

WB03b synthesis for thesis narrative:
- The strongest ablation evidence came from **attention pooling only** (`PR-AUC=0.5197`), suggesting pooling strategy can drive measurable gains over WB03 baseline GATv2.
- JK-only (`cat`) also improved PR-AUC (`0.5157`) versus WB03 primary.
- Combining JK-cat + attention did not outperform attention-only in this run.

### 4.13 WB03c2 status (NNConv with edge features)
`wb03c2_e2_edge_featrures_conv.ipynb` defines the next experiment stage:
- NNConv with 95 aligned edge features from WB03c1
- Real-edge vs random-edge ablation
- Dense-feature variant search (52 features)

Current workspace status:
- Notebook/script design present.
- No `results/wb03c2/` artifacts currently saved in this repository snapshot.

### 4.14 WB03c3 status (lighter edge-feature architectures)
`wb03c3_e2_GINE_Transformer.ipynb` defines follow-up alternatives to NNConv:
- GINEConv
- TransformerConv

Current workspace status:
- Design notebook present.
- No `results/wb03c3/` artifacts currently saved in this repository snapshot.

Thesis-safe phrasing:
- Treat **WB03 and WB03b** as completed result stages.
- Treat **WB03c2 and WB03c3** as documented next-stage experiments unless run artifacts are produced.

### 4.15 Suggested figures/tables to include in Section 4.4
Use existing files from `results/wb03/`:
- `optuna_diagnostics.png`
- `training_curves.png`
- `pr_roc_curves.png`
- `confusion_matrices.png`

Use existing files from `results/wb03b/`:
- `optuna_diagnostics.png`

Suggested table order:
1. Best hyperparameters by architecture.
2. WB02 baseline architectures + baseline metric table.
3. WB03 tuned model comparison (PR-AUC/ROC-AUC/F1/params).
4. WB03b refinement table (search best + ablations).
5. Cross-study comparison table.

---

## 5) Direct copy-paste prompt for ChatGPT
Use this prompt to generate your thesis text quickly.

```text
You are writing Section 4.3 and 4.4 of my TU862 thesis report (Computing with AI/ML Project 2023-24).

Constraints:
- Section 4.3 (Data pre-processing): about 1000 words (+/-20%), formal academic style, no fluff.
- Section 4.4 (Model development): about 1000 words (+/-20%), formal academic style, include tool/infrastructure rationale, model evolution, evaluation protocol, and results tables (no deep interpretation beyond reporting metrics).
- UK/Irish academic tone.
- Do not invent metrics. Use only numbers provided below.
- Where relevant, include brief rationale for each major decision.
- Keep model discussion reproducible and implementation-grounded.

Context:
[PASTE THE ENTIRE CONTENTS OF docs/context.md HERE]

Output format:
1) Section 4.3 final prose with clear subheadings.
2) Section 4.4 final prose with clear subheadings.
3) Markdown tables embedded where appropriate.
4) A short list of figure callouts using exact filenames from results/wb03 and results/wb03b.
```

---

## 6) Optional add-on prompt (for stronger examiner-facing style)

```text
Now revise the two sections for examiner scrutiny:
- Add explicit reproducibility statements.
- Add explicit leakage-avoidance statements.
- Add one short paragraph on limitations of current WB03 evidence (e.g., unexecuted WB03c2/WB03c3 in current snapshot).
- Keep all metrics unchanged.
- Keep each section within +/-20% of 1000 words.
```

---

## 7) Quick consistency checklist before submission
- [ ] Section 4.3 explicitly states component-level split rationale and leakage prevention.
- [ ] Section 4.3 includes both node and edge preprocessing (WB03c1) details.
- [ ] Section 4.4 includes toolchain + infrastructure + optimization protocol.
- [ ] Section 4.4 clearly narrates WB02 -> WB03 -> WB03b progression.
- [ ] Section 4.4 reports PR-AUC/ROC-AUC/F1 tables exactly as saved outputs.
- [ ] Section 4.4 clearly distinguishes completed WB03/WB03b vs planned WB03c2/WB03c3.
- [ ] Figures and table filenames match repository artifacts.
