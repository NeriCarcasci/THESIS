<div align="center">

# Towards Trustworthy AI in Financial Compliance
### A Study of Explainable Graph Neural Networks Using Elliptic2

X00195265 &nbsp;|&nbsp; BSc (Hons) Computing with AI & ML &nbsp;|&nbsp; TU Dublin

![topic](https://img.shields.io/badge/topic-AML%20%7C%20GNN%20%7C%20XAI-brightgreen)
![dataset](https://img.shields.io/badge/dataset-Elliptic2-orange)
![demo](https://img.shields.io/badge/demo-thesis.neri.wtf-purple)

<img src="COVER.png" alt="Towards Trustworthy AI in Financial Compliance" width="80%" />

</div>

---

## Abstract

Regulatory regimes like the EU AI Act and the ALTAI guidelines mandate that anti-money-laundering compliance teams using blockchain transaction data deploy models which are not only accurate but also auditable and interpretable. Graph neural networks are an obvious solution for transaction data; however, the published baselines for the Elliptic2 benchmark do not provide performance metrics for GNNs using node features or the ability to interpret model explanations, raising the questions of what GNNs with features are able to achieve on this task and the extent to which we can interpret their results.

We show how a state-of-the-art GNN, a two-layer Graph Attention Network v2 with Gated Attention Pooling, can be trained on the labelled subgraphs of Elliptic2 without using the 196 million-edge background graph, using only the 43 node features. The primary model achieves a test PR-AUC of **0.515** and a ROC-AUC of **0.934**. Against the reported structure-only benchmarks, this represents a **2.47× increase in PR-AUC**, but the correct interpretation is not that this work has developed a better architecture, but that the discriminative signal in Elliptic2 is in node attributes rather than macro-topology.

On the explainability of these models, we show that Integrated Gradients and Kernel SHAP agree on the global feature ranking (Spearman rank correlation of **0.9505**), and that the discriminative features are revealed in a feature signedness contrast as opposed to in their attribution magnitude. GNNExplainer reveals explanatory features that are consistent with the global feature rankings for all 200 stratified test graphs, whereas the PGExplainer algorithm fails to produce informative masks under four training conditions, which we attribute to a known issue which must be addressed in future work.

> **Live demo:** [thesis.neri.wtf](https://thesis.neri.wtf) - interactive prediction, explanation, and threshold exploration over the 200-subgraph stratified sample.

---

## Research question

> *Can an explainable GNN trained on node features only, on the labelled subgraphs of the Elliptic2 dataset, match or exceed structure-based baselines on the AML task, and do the resulting explanations yield meaningful insights for analysts?*

## Contributions

1. **Feature-aware subgraph-local benchmark.** A two-layer GATv2 with global attention pooling reaches 0.515 PR-AUC on the held-out test set, ~2.47× the best structure-only baseline. The regime caveat is that this comparison locates the signal in node attributes, not in architectural choice.
2. **Cross-method feature attribution.** Integrated Gradients and Kernel SHAP agree on the global feature ranking at Spearman ρ = 0.9505 with sign concordance on the top-five features. Discriminative power is in the *signed contrast* between classes, not in attribution magnitude.
3. **Broad structural-explainer evaluation.** GNNExplainer, GATv2 attention, and SubgraphX evaluated on the same stratified sample; mask entropy proposed as a diagnostic on the median three-node subgraph where standard fidelity breaks down.
4. **ALTAI / EU AI Act mapping.** The pipeline is assessed against the seven ALTAI dimensions and Articles 9–15 of the AI Act, identifying which gaps in current GNN explainability prevent compliance-grade deployment.

---

## Contents

- [Dataset](#dataset-elliptic2)
- [Headline results](#headline-results)
- [Repository layout](#repository-layout)
- [Workbooks → thesis chapters](#workbooks--thesis-chapters)
- [Reproducibility artefacts](#reproducibility-artefacts)
- [Environment setup](#environment-setup)
- [How to run](#how-to-run)
- [ALTAI self-assessment](#altai-self-assessment-summary)

---

## Dataset (Elliptic2)

[Elliptic2](https://arxiv.org/abs/2404.19109) (Bellei et al., 2024) is the largest fully labelled public AML benchmark, jointly released by MIT CSAIL, the MIT-IBM Watson AI Lab, and Elliptic. It comprises **121,810 labelled connected components** drawn from a background graph of approximately **49.4M nodes and 196.2M directed edges**, with a **2.27% positive rate** (negative-to-positive ratio of 43:1).

Labels are at the **subgraph level** (`ccLabel`), matching the unit at which an AML compliance team makes a decision (the filing of a Suspicious Activity Report). Each node carries 43 binned ordinal features and each edge 95 binned ordinal features; all anonymised by Elliptic to preserve their commercial feature dictionary.

**Scope choice.** This thesis operates on the labelled-subgraph universe (444,521 nodes, 367,137 edges, 121,810 components) without exploiting the 196M-edge background graph. Future work can extend through the GLASS labelling trick over the background graph (Wang & Zhang, 2022).

---

## Headline results

**Test split (24,362 components, 553 suspicious, 2.27% prevalence). Threshold chosen on validation to maximise F1.**

| Group | Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Bellei et al. 2024 | GNN-Seg (structure-only) | 0.026 | 0.537 | - | - | - |
| Bellei et al. 2024 | Sub2Vec (structure-only) | 0.022 | 0.496 | - | - | - |
| Bellei et al. 2024 | GLASS (structure + background) | 0.208 | 0.889 | - | - | - |
| Wb02 baseline | LogReg (pooled mean/max/std) | 0.154 | 0.890 | 0.251 | 0.189 | 0.374 |
| Wb02 baseline | GraphSAGE (default) | 0.401 | 0.914 | 0.414 | 0.378 | 0.458 |
| Wb03 sweep | GraphSAGE (tuned) | 0.485 | 0.923 | 0.491 | 0.635 | 0.400 |
| Wb03 sweep | GCN (tuned) | 0.420 | 0.917 | 0.422 | 0.496 | 0.367 |
| Wb03 sweep | GATv2 (tuned) | 0.496 | 0.928 | 0.484 | 0.578 | 0.416 |
| **Wb03b primary** | **GATv2 + attention pool** | **0.515** | **0.934** | **0.516** | **0.649** | 0.429 |
| Wb03b alt | GATv2 + JK(cat) - recall-leaning | 0.516 | 0.932 | 0.504 | 0.575 | 0.449 |

**Operating characteristics of the primary model.** At threshold 0.913: confusion matrix TP = 237, FP = 128, FN = 316, TN = 23,681. False-positive rate **0.54%**, alert rate **1.50%**. A flagged subgraph is approximately **29×** more likely to be suspicious than a uniformly drawn one given the 2.27% base rate.

**Read as a family,** the four tuned backbones in this work all exceed the structure-only GLASS baseline by at least a factor of two on PR-AUC (GraphSAGE 2.33×, GCN 2.02×, GATv2 2.39×, refined GATv2 2.47×). The takeaway is not that this work has produced a better architecture, but that the discriminative signal in Elliptic2 is concentrated in node attributes rather than macro-topology.

### Explainability

- **Feature attribution agreement.** Integrated Gradients vs. Kernel SHAP, global per-feature importance over the 200-subgraph stratified sample: **Spearman ρ = 0.9505**, with sign concordance on the top-five discriminative features (F23, F27, F35, F19, F29).
- **Structural explainers.** GNNExplainer informative on **200/200** subgraphs (zero near-uniform rate). GATv2 attention informative on all 2-edge and 3+-edge subgraphs (degenerate on 1-edge inputs by construction). SubgraphX retains a median 3 / 6 nodes per explanation. **PGExplainer collapsed to all-zero masks under four training conditions** - left out of cross-method comparison; consistent with prior reports of PGExplainer instability under class imbalance and small subgraph sizes.
- **Fidelity caveat.** On the median three-node Elliptic2 subgraph, removing the top-50% of explanation edges typically disconnects the graph, so standard Fidelity+ / Fidelity- metrics carry standard deviations an order of magnitude above their means. Mask entropy is reported as the more robust diagnostic (0.489 / 1.091 / 1.926 bits at 2-/3-/4+-node strata).
- **Structural bias surfaced.** 28 of 40 false positives in the explanation sample (70%) are 2-node subgraphs. A user whose on-chain activity forms a 2-node component is more likely to be falsely flagged than one forming a larger component - an operational bias to be mitigated by size-stratified thresholding in production.

---

## Repository layout

```
.
├── DATA/                                   # NOT COMMITTED - raw Elliptic2 CSVs (~90 GB)
│   ├── nodes.csv, edges.csv, connected_components.csv
│   └── background_nodes.csv, background_edges.csv
├── processed/                              # NOT COMMITTED - derived artefacts
├── results/                                # metrics, predictions, explanation outputs
│   ├── wb03/, wb03b/, wb03c1/              # architecture sweep & refinement & edge features
│   ├── wb04/, wb05/, wb06/                 # explainability outputs
│   └── baseline_*_metrics.json
├── scripts/                                # supporting CLI utilities
├── thesis.pdf                              # final submitted thesis
├── wb01_e2_preprocessing.ipynb
├── wb02_e2_baselinesAndGraphs.ipynb
├── wb02.5_visualizations_panel_review.ipynb
├── wb03_model_development.ipynb
├── wb03b_model_refinement.ipynb
├── wb03b_retrain_attention.ipynb
├── wb03c1_e2_preprocessing_edge_feat.ipynb
├── wb03c2_e2_edge_featrures_conv.ipynb
├── wb03c3_e2_GINE_Transformer.ipynb (+ .py)
├── wb04_05_06_explainability.ipynb
├── requirements.txt
└── README.md
```

---

## Workbooks → thesis chapters

| Workbook | Chapter | Purpose |
|---|---|---|
| `wb01_e2_preprocessing` | Ch. 4 | Extract the labelled-subgraph universe; build component-level 70/10/20 stratified split; align node features; emit reproducibility artefacts and SHA256 fingerprints. |
| `wb02_e2_baselinesAndGraphs` | Ch. 5 (Phase Wb02) | Logistic regression on pooled mean/max/std features and a default 2-layer GraphSAGE; lazy PyG dataset object; structural diagnostics; layout-based subgraph visualisations. |
| `wb02.5_visualizations_panel_review` | Ch. 4 / Ch. 6 figs | Diagnostic and figure panels for the thesis (component-size distribution, edge-feature diagnostics, etc.). |
| `wb03_model_development` | Ch. 5 (Phase Wb03) | Systematic architecture sweep across GraphSAGE / GCN / GATv2 - Optuna TPE, 30 trials per architecture, median pruner, fANOVA importance. |
| `wb03b_model_refinement` | Ch. 5 (Phase Wb03b) | GATv2 refinement search adding JumpingKnowledge, gated attention pooling, cosine LR scheduler; four-corner ablation. |
| `wb03b_retrain_attention` | Ch. 5 / Ch. 6 | Strict-checkpoint retrain of the best Wb03b configuration. **This is the canonical primary model** used for all explainability experiments. |
| `wb03c1_e2_preprocessing_edge_feat` | Ch. 4.3 / 5.7 | Extract the 95 edge features by streamed predicate-pushdown scan over the 196.2M-row background edge file (~679s), deterministic multigraph alignment to PyG packed ordering. |
| `wb03c2_e2_edge_featrures_conv` | Ch. 5.7 (NNConv) | NNConv search; pre-registered stopping rule (best of 5 trials = 0.4063 val PR-AUC) fires due to the O(d_e · d²) parameter blow-up at median degree 3. |
| `wb03c3_e2_GINE_Transformer` | Ch. 5.7 (Wb03c3) | GINEConv (30 trials) and TransformerConv (23 trials) edge-aware searches. GINEConv reaches val 0.5393 / test 0.5087, within noise of GATv2 + attention pool. TransformerConv tops out at 0.4307. |
| `wb04_05_06_explainability` | Ch. 6 | Stratified 200-subgraph sample (4 quadrants × 3 size buckets); Integrated Gradients & Kernel SHAP; GNNExplainer, PGExplainer, GATv2 attention, SubgraphX; near-uniform rates, fidelity, mask entropy, Jaccard agreement. |

---

## Reproducibility artefacts

Every metric in the thesis is derived from six files emitted by the preprocessing notebooks (Section 4.6):

- `edge_index.npy` - shape `(2, 367,137)`, dtype `int64`
- `node_features.npy` - shape `(444,521, 43)`, dtype `int8`
- `edge_features.npy` - shape `(367,137, 95)`, dtype `float32`
- `node_components.npy` - length `444,521`, dtype `int64`
- `splits.json` - train/val/test component IDs (70.0% / 10.0% / 20.0%, stratified, positive rates 2.2682% / 2.2658% / 2.2699%)
- `feature_stats_train.json` - per-feature mean / std / min / max, **train-only** (leakage-safe)

All experiments use a fixed random seed of **7** across NumPy, PyTorch, and the Optuna TPE sampler. `torch.backends.cudnn.deterministic=True`, `cudnn.benchmark=False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` disable non-deterministic kernel auto-tuning. Running the same code on the same hardware and software environment produces bit-identical results.

---

## Environment setup

```bash
conda create -n elliptic2-xgnn python=3.11 -y
conda activate elliptic2-xgnn

# Install PyTorch first (pick the build matching your CUDA), then PyG:
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric

pip install -r requirements.txt
```

See `requirements.txt` for the full PyG extension wheel guidance and pinned dependencies.

---

## How to run

1. Place the five Elliptic2 CSVs in `DATA/` (raw release from Bellei et al., 2024).
2. Run **`wb01_e2_preprocessing`** end-to-end to emit the six reproducibility artefacts under `processed/`.
3. (Optional, edge-feature path) Run **`wb03c1_e2_preprocessing_edge_feat`** to emit `edge_features.npy` and `edge_feature_dense_meta.json`.
4. Run **`wb02_e2_baselinesAndGraphs`** for the LogReg + default-GraphSAGE baselines.
5. Run **`wb03_model_development`** for the architecture sweep (GraphSAGE / GCN / GATv2).
6. Run **`wb03b_model_refinement`** then **`wb03b_retrain_attention`** to produce the canonical primary checkpoint.
7. (Optional, edge-feature ablations) Run **`wb03c2`** and **`wb03c3`** for NNConv / GINEConv / TransformerConv.
8. Run **`wb04_05_06_explainability`** to reproduce the stratified 200-subgraph sample, attribution rankings, and structural-explainer outputs.

Outputs are written to `results/<wbXX>/`; checkpoints live under `results/wb03b/`.

---

## ALTAI self-assessment summary

| ALTAI requirement | Applicability | Key measures |
|---|---|---|
| Human agency and oversight | High | Decision-support framing; precision-weighted threshold (t = 0.913); per-prediction explanations |
| Technical robustness and safety | High | Stratified component-level splits; early stopping; weighted loss; multi-method explanation cross-validation |
| Privacy and data governance | Moderate | Anonymised features; no PII; research licence; leakage-safe preprocessing |
| Transparency | High | Three-layer explanations; cross-method agreement; semantic ceiling from anonymisation explicitly stated |
| Diversity and fairness | Limited | Protected-attribute auditing impossible on Elliptic2; structural bias on 2-node subgraphs (28/40 FPs); size-stratified threshold proposed |
| Environmental and societal well-being | Moderate | ~43 kWh / ~14 kg CO₂ across the full study; pruned search; compact architectures (95k–551k params) |
| Accountability | High | Reproducible pipeline; per-prediction attribution and edge-mask reports; open limitation reporting |

---

## Citation

If you build on this work, please cite the thesis:

> Carcasci, N. (2026). *Towards Trustworthy AI in Financial Compliance: A Study of Explainable Graph Neural Networks Using Elliptic2.* BSc (Hons) Computing with AI & ML thesis, TU Dublin (Tallaght). Supervisor: Dr. Musfira Jilani.

And the Elliptic2 dataset:

> Bellei, C. et al. (2024). *The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset.* arXiv:2404.19109.
