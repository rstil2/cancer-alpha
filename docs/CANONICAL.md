# Canonical results (official numbers)

**Use this table for README, talks, GitHub, and reviewer responses.**  
Do not cite legacy values (96.5%, 97.6% CUP, 4,913 samples, competitive leaderboard ranks).

**Manuscript under review:** JBI received the **2026-06-21 snapshot** (`science/jbi_revision/submitted_snapshot/`). That PDF still states Study 1 as **95.0% ± 5.4%** and ICGC **92.1%**. Do not edit those files in place. Use the **reproduced** column below for anything public that must match the code.

Full detail: [science/jbi_revision/supplementary/CANONICAL_RESULTS.md](../science/jbi_revision/supplementary/CANONICAL_RESULTS.md)

---

## Study 2 — large balanced (canonical pipeline)

**Reproduce:** `python src/pipeline/step4_train_evaluate.py`  
**Output:** `data/real_model_results/model_results.json` (verified 2026-03-08)

| Item | Value |
|------|-------|
| Samples | 1,248 (156/class) |
| Train / test | 998 / 250 |
| Features | 4,063 |
| Champion | LightGBM **98.4%** test balanced accuracy (LR tied) |
| Deep baselines | MLP 93.6%, TabTransformer 91.6% |
| Validation | Internal TCGA held-out split only |

## Study 1 — small-n

**Reproduce:** `python src/pipeline_study1/run_all.py`  
**Outputs:** `data/study1_results/model_results.json`, `features_110.pkl`, `canonical_patient_manifest.json`

| Item | Submitted MS (frozen) | **Reproduced pipeline (2026-06-23)** |
|------|----------------------|--------------------------------------|
| Samples | 158 imbalanced TCGA | 158 (deterministic modality completeness) |
| Features | 110 (6 modalities) | 110 (KIRC panel; CN-derived fragmentomics + expression gene proxies) |
| LightGBM + SMOTE CV | **95.0% ± 5.4%** | **82.4% ± 2.6%** (10×5-fold) |
| TCGA held-out external (4 types) | — | **83.7%** (n=68; frozen model) |
| ICGC external (n=76) | **92.1%** | **25.0%** partial (ICGC Xena hub; mutation-only + imputed methylation) |
| Original 158 patient ID list | Implied in MS | **Not found in repo**; see `cohort_lineage_report.json` |
| 12-model benchmark / R² | R² = 0.78 (MS) | Re-run `step4b_benchmark_architectures.py` after cohort changes |

**Interpretation:** Study 1 reproduced scores are **lower** than the submitted manuscript. Likely causes: no archived patient manifest, proxy feature blocks, and partial ICGC harmonization. Cohort **size and class targets** match the paper; **accuracy does not**.

**ICGC note:** Open Xena data excludes US TCGA-overlap projects. Full ARGO multi-omics requires DACO. See `data/icgc_argo/dataset_info.json`.

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

- **97.6% CUP** — simulated held-out TCGA splits only
- **95.0% / 92.1% Study 1** — submitted MS only unless reproduced in your run
- **Competitive leaderboard** vs FoundationOne — different tasks/metrics
- **Demo Streamlit ~70%** — illustrative sample data
