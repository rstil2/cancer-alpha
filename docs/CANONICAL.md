# Canonical results (official numbers)

**Use this table for README, papers, talks, and reviewer responses.**  
Do not cite legacy values (96.5%, 97.6% CUP, 4,913 samples, competitive leaderboard ranks).

Full detail: [science/jbi_revision/supplementary/CANONICAL_RESULTS.md](../science/jbi_revision/supplementary/CANONICAL_RESULTS.md)

---

## Study 2 — large balanced (canonical pipeline)

**Reproduce:** `python src/pipeline/step4_train_evaluate.py`  
**Output:** `data/real_model_results/model_results.json`

| Item | Value |
|------|-------|
| Samples | 1,248 (156/class) |
| Train / test | 998 / 250 |
| Features | 4,063 |
| Champion | LightGBM **98.4%** test balanced accuracy (LR tied) |
| Deep baselines | MLP 93.6%, TabTransformer 91.6% |
| Validation | Internal TCGA held-out split only |

## Study 1 — small-n

| Item | Value |
|------|-------|
| Samples | 158 imbalanced TCGA |
| Features | 110 (6 modalities) |
| Champion | LightGBM + SMOTE **95.0% ± 5.4%** |
| Transformers | 83.2% ± 6.8% |
| External | ICGC ARGO n=76 → **92.1%** (no retraining) |

## Imbalance robustness (Study 2 related)

Train balanced → test natural prevalence ([`experiments/imbalance_stress_test.py`](../experiments/imbalance_stress_test.py)):

| Test set | Balanced accuracy | Macro F1 |
|----------|------------------:|---------:|
| Balanced holdout | 96.4% | 96.3% |
| Natural prevalence | 95.7% | 89.8% |
| **Drop** | **0.7 pp** | 6.8 pp |

Results: [science/jbi_revision/supplementary/imbalance_stress_test_results.json](../science/jbi_revision/supplementary/imbalance_stress_test_results.json)

---

## Do not cite publicly

- **97.6% CUP** — simulated held-out TCGA splits only; removed from demo/README
- **Competitive leaderboard** vs FoundationOne — different tasks/metrics ([historical doc](../archive/business/docs/Competitive_Analysis_Methodology.md))
- **16.55% on 9,660 samples** — exploratory archive experiment
- **Demo Streamlit accuracy ~70%** — illustrative sample data
