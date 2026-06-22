# Reproduction Guide — Study 2 (JBI Manuscript)

Reproduces the **98.4% held-out test balanced accuracy** reported in Study 2.

---

## Prerequisites

- Python 3.10+
- ~16 GB RAM recommended
- TCGA data processed through pipeline steps 1–3 (feature pickles in `data/real_model_results/`)

If feature pickles already exist in the repo clone, skip to Step 4.

---

## Quick reproduction (feature pickles present)

```bash
cd /Users/stillwell/projects/cancer-alpha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train and evaluate all Study 2 models
python src/pipeline/step4_train_evaluate.py
```

**Expected output file:** `data/real_model_results/model_results.json`

**Expected LightGBM test balanced accuracy:** ~0.984 (98.4%)

---

## Full pipeline (from raw GDC mapping)

```bash
# Step 1: Map GDC files to patient IDs
python src/pipeline/step1_file_mapping.py

# Step 2: Gene expression features (top 2000 by variance)
python src/pipeline/step2_expression_features.py

# Step 2b: DNA methylation features (top 2000 CpG probes)
python src/pipeline/step2b_methylation_features.py

# Step 3: Somatic mutation features (63 variables)
python src/pipeline/step3_mutation_features.py

# Step 4: Integrate, balance, train, evaluate
python src/pipeline/step4_train_evaluate.py
```

---

## What Step 4 does

1. Joins expression + methylation + mutation features by patient ID
2. Subsamples to **156 patients per cancer type** (1,248 total)
3. Splits **80/20** train/test (998/250), stratified, seed=42
4. Trains LightGBM, XGBoost, Random Forest, Logistic Regression
5. Reports 5-fold CV on training set and held-out test metrics
6. Saves `best_model.pkl`, SHAP importances, `dataset_info.json`

---

## Verify results

```bash
python -c "
import json
from pathlib import Path
r = json.loads(Path('data/real_model_results/model_results.json').read_text())
lgb = r['LightGBM']
print('LightGBM test balanced accuracy:', round(lgb['test_balanced_accuracy']*100, 2), '%')
print('CV mean:', round(lgb['cv_mean']*100, 2), '% ±', round(lgb['cv_std']*100, 2), '%')
"
```

---

## What is NOT part of manuscript reproduction

| Path | Reason |
|------|--------|
| `cancer_genomics_ai_demo_minimal/` | Demo uses sample/synthetic data (~70% accuracy) |
| `train_transformer_real_tcga_balanced.py` | Deep learning comparison; separate experiment |
| Root `*_50k_*`, `ultra_massive_*` scripts | Exploratory scaling experiments (poor feature engineering in some runs) |
| `COMPREHENSIVE_MODEL_RESULTS_SUMMARY.md` | 9,660-sample experiment with ~16% accuracy — not manuscript |

---

## For reviewers

Point reviewers to:

1. This guide: `science/jbi_revision/supplementary/REPRODUCTION_GUIDE.md`
2. Canonical numbers: `science/jbi_revision/supplementary/CANONICAL_RESULTS.md`
3. Frozen submitted PDF: `science/jbi_revision/submitted_snapshot/Combined_Manuscript_JBI.pdf`

Legacy scripts remain in the repository for development history but are explicitly excluded from manuscript claims in the revised Data Availability statement.
