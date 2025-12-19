# Towards Trustworthy AI in Financial Compliance
X00195265 Thesis Project

## Abstract
Money laundering poses a major threat to global financial stability. Each year, up to 5% of global GDP, or €1.7 trillion, is laundered (UNODC, 2024), so financial institutions have to allocate vast resources to counter these illicit activities. In the EMEA region alone, compliance costs reached US$85 billion in 2023 (LexisNexis Risk Solutions, 2024). Despite these investments, Anti-Money Laundering (AML) systems continue to face challenges, including high false-positive rates and limited interpretability. 
While machine learning (ML) has improved classification accuracy, it is frequently applied to isolated transactions, neglecting the relational nature of financial crime. Graph Neural Networks (GNNs) offer a solution because they can model transactions as networks, preserving interactions over time steps. However, most current GNN models, like all complex models, lack transparency, which impedes their adoption in regulatory settings that require explainability and fairness.
This research proposes the design, implementation, and evaluation of an explainable GNN-based model for detecting illicit financial transactions using the Elliptic2 dataset (Bellei et al., 2024), developed by MIT CSAIL, IBM Watson AI Lab, and Elliptic. The primary focus is to explore how post-hoc and graph-specific explainability methods, such as SHAP, LIME, GNNExplainer, and Counterfactual-GNN, can improve the transparency and interpretability of GNN decisions in financial crimes detection. 
To support this goal, the study will include a preliminary bias analysis to examine whether the generated explanations differ across transaction types or network structures, helping to identify potential limitations in the transparency and fairness of the model’s reasoning.
In addition, a comparative evaluation against a traditional baseline model will be conducted to understand how graph-based representations affect both predictive accuracy and the interpretability of explanations. These supporting analyses will help contextualise the explainability results and clarify their practical relevance for financial compliance systems.


## Scope and research objectives
This repository is structured around four research threads:
1. **Performance**: Compare GNN subgraph classification against traditional ML baselines under severe class imbalance.   
2. **Explainability**: Evaluate post-hoc explainers (feature-based and graph-native) for transparency and stability on AML subgraphs.   
3. **Bias / uneven treatment**: Probe whether prediction behaviour and explanation outputs differ across structural subgroups (e.g., size/density buckets).   
4. **Workflow integration**: Produce explanation outputs that can plausibly integrate into AML analyst decision-making.   



## Dataset characteristics (Elliptic2)
Elliptic2 is treated as a **subgraph-labelled** benchmark:
- `clId`: node / entity cluster identifier (graph vertex)
- `ccId`: connected-component identifier (one **subgraph instance**)
- `ccLabel`: supervision at the **component/subgraph** level (target for classification)

The workflow operates on the **labelled subgraph universe** (nodes/edges induced by labelled components) while sourcing node attributes from the much larger background graph.



## Repository layout
```
.
├── DATA/                      # NOT COMMITTED (raw Elliptic2 CSVs)
│   ├── nodes.csv
│   ├── edges.csv
│   ├── connected_components.csv
│   ├── background_nodes.csv
│   └── background_edges.csv
├── processed/                 # NOT COMMITTED (derived artifacts)
│   ├── parquet/
│   ├── artifacts/
│   └── arrays/
├── results/                   # metrics + predictions per run
├── notebooks/
│   ├── wb01_e2_preprocessing.ipynb
│   └── wb02_e2_baselinesAndGraphs.ipynb
├── requirements.txt
└── README.md
```


## Workbooks overview

### 01: Preprocessing and artifact generation
**Purpose:** Convert the Elliptic2 labelled subset into a reproducible, model-ready research state with strict integrity checks.

**Key operations**
- Locks schema (explicit column names) and validates inputs.
- Converts core CSVs → Parquet for iterative speed.
- Enforces referential integrity (edges closed over labelled nodes).
- Builds **subgraph labels** (`ccId → ccLabel`) and **component-level splits** (train/val/test).
- Extracts node features by subsetting `background_nodes.csv` to labelled `clId`s; produces `node_features.npy`.
- Computes train-only feature diagnostics for leakage-safe downstream scaling/analysis.

**Primary outputs**
- `processed/arrays/node_features.npy` : node features aligned to `nodes.csv` order  
- `processed/arrays/node_components.npy` : component membership per node  
- `processed/arrays/edge_index.npy` : global edge index in contiguous node space  
- `processed/artifacts/subgraph_labels.json` : `ccId → ccLabel`  
- `processed/artifacts/splits.json` : train/val/test component IDs  
- `processed/artifacts/feature_stats_train.json` : leakage-safe statistics  



### 02: Baselines, dataset objects, and graph visualisation
**Purpose:** Build credible baselines and a trainable dataset interface, plus structural diagnostics and visual intuition.

**Key operations**
- Loads Workbook 01 artifacts and core labelled CSVs.
- Computes component-level structural stats (`n_nodes`, `n_edges`, density proxies).
- Visualises representative subgraphs (small/medium/large) using NetworkX layouts.
- Packs nodes/edges by component into compact sampling arrays.
- Implements a lazy **PyTorch Geometric** dataset: one `Data` object per `ccId`.
- Verifies split hygiene (disjointness and coverage).
- Trains:
  - **Baseline A**: pooled node-feature model (mean/max/std pooling → logistic regression)
  - **Baseline B**: minimal **GraphSAGE** subgraph classifier (global pooling)

**Primary outputs**
- `processed/artifacts/packed/nodes_by_ccid.npz` : nodes grouped by component
- `processed/artifacts/packed/edges_by_ccid.npz` : intra-component edges grouped by component
- `results/baseline_tabular_metrics.json` : pooled-feature baseline metrics
- `results/baseline_gnn_metrics.json` : GraphSAGE baseline metrics
- Optional prediction tables (if enabled in notebook)



## Environment setup (conda)
```bash
conda create -n elliptic2-xgnn python=3.11 -y
conda activate elliptic2-xgnn
pip install -r requirements.txt
```



## How to run
1. Place Elliptic2 CSVs in `DATA/`
2. Run **Workbook 01** end-to-end to generate `processed/`.
3. Run **Workbook 02** to produce baselines, diagnostics, and visualisations.
4. Track outputs in `results/` and keep artifacts under `processed/`.


