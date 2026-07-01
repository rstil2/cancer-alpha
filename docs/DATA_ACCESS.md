# Data access and reproduction

`data/` is **gitignored** (TCGA files are large and restricted). A fresh clone does not include processed feature matrices by default.

---

## Study 2 — recommended path

### Option A: Regenerate from GDC (full reproduction)

1. Download TCGA data via [GDC Data Portal](https://portal.gdc.cancer.gov/) (RNA-seq, methylation, mutations for BRCA, LUAD, COAD, PRAD, STAD, HNSC, LUSC, LIHC).
2. Run the canonical pipeline:

```bash
pip install -r requirements.txt
python src/pipeline/step1_file_mapping.py
python src/pipeline/step2_expression_features.py
python src/pipeline/step2b_methylation_features.py
python src/pipeline/step3_mutation_features.py
python src/pipeline/step4_train_evaluate.py
```

3. Verify: `data/real_model_results/model_results.json` → LightGBM test balanced accuracy ≈ **0.984**.

See [science/jbi_revision/supplementary/REPRODUCTION_GUIDE.md](../science/jbi_revision/supplementary/REPRODUCTION_GUIDE.md).

### Option B: Use local processed outputs (developer machine)

If you already ran the pipeline, these files exist locally:

| File | Purpose |
|------|---------|
| `data/real_model_results/expression_features.pkl` | Gene expression matrix |
| `data/real_model_results/methylation_features.pkl` | Methylation matrix |
| `data/real_model_results/mutation_features.pkl` | Mutation features |
| `data/real_model_results/model_results.json` | All model metrics |
| `data/real_model_results/best_model.pkl` | Trained LightGBM |
| `data/file_patient_mapping.csv` | GDC file → patient map |

### Option C: Zenodo (planned)

Processed feature matrices, predictions, and SHAP values will be deposited on **Zenodo upon JBI acceptance**. Until then, use Option A or contact the author.

---

## Study 1 — minimal data

Processed Study 1-style features (when present locally):

| Path | Notes |
|------|-------|
| `data/production_tcga/` | Early real TCGA processing |
| `data/raw_tcga/` | Raw downloads |
| Demo package models | 158-sample LR/RF in `demo/models/` |

See [src/pipeline_study1/README.md](../src/pipeline_study1/README.md).

---

## Imbalance stress test

Uses `data/real_tcga_large/real_tcga_features_cleaned.csv` (1,200 balanced samples, 2,000 features) when available locally:

```bash
python experiments/imbalance_stress_test.py
```

---

## What is not manuscript data

| Path | Status |
|------|--------|
| `demo/streamlit_app.py` | Streamlit workflow demo (sample data in UI) |
| `data/processed_50k/`, `data/advanced_multi_omics/` | Archive experiments |
| `archive/experiments/` | Legacy scripts |

---

## Contact

For processed data sharing before Zenodo release: **craig.stillwell@gmail.com**
