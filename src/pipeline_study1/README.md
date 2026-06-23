# Study 1 reproduction map (small-n, n=158)

Study 1 is **not** a single end-to-end script like Study 2. This document maps manuscript claims to the closest repository artifacts.

## Manuscript claims (Study 1)

| Claim | Value |
|-------|-------|
| Samples | 158 authenticated TCGA |
| Features | 110 across 6 modalities |
| Architectures | 12 (logistic regression → transformers) |
| Champion | LightGBM + SMOTE |
| Balanced accuracy | 95.0% ± 5.4% |
| Transformer | 83.2% ± 6.8% |
| External validation | ICGC ARGO n=76 → 92.1% (no retraining) |

Cancer types: BRCA, LUAD, COAD, PRAD, STAD, HNSC, **KIRC**, LIHC (note: Study 2 uses LUSC instead of KIRC).

---

## Closest reproduction paths

### 1. Demo models (158 samples, partial)

The Streamlit demo ships models trained on the minimal-data setting:

```bash
cd cancer_genomics_ai_demo_minimal
python setup.py
# Models: models/multimodal_real_tcga_*.pkl (158 samples, 110 features)
```

This demonstrates the **workflow**, not the full 12-architecture comparison or 95.0% LightGBM+SMOTE benchmark.

### 2. Model comparison on real TCGA CSV (related, n≈1,200)

```bash
python compare_models_real_data.py
```

Uses `data/real_tcga_large/real_tcga_features_cleaned.csv` (1,200 balanced samples, 2,000 features) with SMOTE — **different cohort size/features** than Study 1 but same methodology family.

### 3. Notebooks

Check `notebooks/` for interactive Study 1-era analyses (preprocessing, SMOTE, architecture comparison).

### 4. ICGC external validation

- Data: [ICGC ARGO platform](https://platform.icgc-argo.org/)
- Manuscript: 76 held-out samples, LightGBM+SMOTE without retraining
- **No standalone reproduction script is pinned in this repo yet** — priority for a future `src/pipeline_study1/run_external_validation.py`

---

## Data locations (local only, gitignored)

| Path | Description |
|------|-------------|
| `data/production_tcga/` | Processed real TCGA for early experiments |
| `data/raw_tcga/` | Raw GDC downloads |
| `real_tcga_inventory.json` | Sample inventory audit (Aug 2025) |

---

## Planned consolidation

A unified `src/pipeline_study1/` script chain is planned to match Study 2's reproducibility standard. Until then, cite Study 1 numbers from the [canonical results](../../docs/CANONICAL.md) and manuscript PDF only.

---

## SMOTE note

Study 1 uses SMOTE (k=4). Study 2 does **not**. Do not describe the project globally as "zero synthetic data" — specify per study.
