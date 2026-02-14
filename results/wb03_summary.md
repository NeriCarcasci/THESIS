# Wb03 Model Development Summary (2026-02-14)

## Scope
- Goal: compare GNN architectures for subgraph-level AML classification on Elliptic2 with Optuna tuning.
- Architectures: GraphSAGE, GCN, GATv2.
- Metric focus: PR-AUC (rare-event regime).

## Primary outputs (results/wb03)
- best_model_state.pt
- best_hyperparameters.csv
- model_comparison.csv
- primary_model_selection.json
- cross_study_comparison.csv
- search_results.json (all trials)
- training_curves.png, pr_roc_curves.png, confusion_matrices.png, optuna_diagnostics.png

## Best model selection
- Selected architecture: GATv2
- Selection criterion: highest test PR-AUC
- Test PR-AUC: 0.4964
- Test ROC-AUC: 0.9284
- Test F1: 0.4837
- Test threshold: 0.8775

Best hyperparameters (Optuna):
- arch: gatv2
- hidden_dim: 256
- num_layers: 2
- dropout: 0.20
- lr: 0.000754
- pool: max
- heads: 1

Optimisation setup:
- Method: Optuna TPE + MedianPruner
- Trials per architecture: 30

## Tuned model comparison (test)

| Model | PR-AUC | ROC-AUC | F1 | Params |
|---|---:|---:|---:|---:|
| GraphSAGE (tuned) | 0.4848 | 0.9234 | 0.4906 | 94,466 |
| GCN (tuned) | 0.4199 | 0.9170 | 0.4220 | 144,386 |
| GATv2 (tuned) | 0.4964 | 0.9284 | 0.4837 | 222,466 |

## Comparison to prior baselines (Wb02)

| Model | PR-AUC | ROC-AUC | F1 |
|---|---:|---:|---:|
| LogReg (pooled, Wb02) | 0.1539 | 0.8895 | 0.2511 |
| GraphSAGE (default, Wb02) | 0.4012 | 0.9137 | 0.4137 |
| GraphSAGE (tuned, Wb03) | 0.4848 | 0.9234 | 0.4906 |
| GCN (tuned, Wb03) | 0.4199 | 0.9170 | 0.4220 |
| GATv2 (tuned, Wb03) | 0.4964 | 0.9284 | 0.4837 |

Key takeaways:
- All tuned GNNs outperform Wb02 baselines on PR-AUC and F1.
- GATv2 is the top model by PR-AUC; GraphSAGE has the top F1 but lower PR-AUC.

## Cross-study comparison (PR-AUC / ROC-AUC)

| Model | Source | Features | Background graph | PR-AUC | ROC-AUC |
|---|---|---|---|---:|---:|
| GNN-Seg | Bellei et al. | No | No | 0.026 | 0.537 |
| Sub2Vec | Bellei et al. | No | No | 0.022 | 0.496 |
| GLASS | Bellei et al. | No | Yes | 0.208 | 0.889 |
| GraphSAGE (tuned) | This work | Yes (43) | No | 0.4848 | 0.9234 |
| GCN (tuned) | This work | Yes (43) | No | 0.4199 | 0.9170 |
| GATv2 (tuned) | This work | Yes (43) | No | 0.4964 | 0.9284 |

## Recommended next steps (decision-oriented)
1. Freeze GATv2 (best PR-AUC) and GraphSAGE (best F1) checkpoints for explainability work.
2. Evaluate stability of PR-AUC and F1 across multiple seeds (report mean and std).
3. Produce calibrated probability outputs (e.g., temperature scaling) and re-check thresholds.
4. Run explanation methods on the two finalists and compare faithfulness and consistency.
