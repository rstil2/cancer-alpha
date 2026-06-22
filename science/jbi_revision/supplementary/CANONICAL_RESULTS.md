# Canonical Results — Single Source of Truth

Use these numbers in all manuscripts, responses, and public materials.  
**Do not cite legacy README values (e.g. 4,913 samples, 96.5%, 97.6%) unless explicitly marked historical.**

Last verified from pipeline output: `data/real_model_results/model_results.json` (2026-03-08).

---

## Study 2 (large balanced) — canonical pipeline

**Pipeline:** `src/pipeline/step1_file_mapping.py` → `step2_expression_features.py` → `step2b_methylation_features.py` → `step3_mutation_features.py` → `step4_train_evaluate.py`

| Parameter | Value |
|-----------|-------|
| Total samples | 1,248 |
| Samples per class (full cohort) | 156 |
| Train / test split | 998 / 250 (80/20, stratified) |
| Features | 4,063 (2,000 expression + 2,000 methylation + 63 mutation) |
| Cancer types | BRCA, COAD, HNSC, LIHC, LUAD, LUSC, PRAD, STAD |
| Validation | Stratified 5-fold CV on training set + held-out test |
| Champion model | LightGBM (Bayesian tuning via Optuna, 20 trials) |
| Random seed | 42 |

### Study 2 — test set balanced accuracy (held-out n=250)

| Model | CV mean ± std | Test balanced accuracy |
|-------|---------------|------------------------|
| LightGBM | 97.8% ± 0.5% | **98.4%** |
| Logistic Regression | 97.0% ± 1.0% | **98.4%** |
| XGBoost | 96.8% ± 0.7% | 98.0% |
| Random Forest | 96.0% ± 0.3% | 97.2% |

### Study 2 — per-class (LightGBM test set)

| Type | Precision | Recall | Notes |
|------|-----------|--------|-------|
| BRCA | 100% | 100% | |
| COAD | 100% | 100% | |
| PRAD | 100% | 100% | |
| STAD | 100% | 100% | |
| LUAD | 100% | 97.4% | |
| HNSC | 96.8% | 96.8% | |
| LIHC | 100% | 93.8% | |
| LUSC | 91.2% | 100% | Most confusions with squamous types |

---

## Study 1 (small-n) — manuscript values

| Parameter | Value |
|-----------|-------|
| Samples | 158 real TCGA |
| Features | 110 (6 modalities) |
| Cancer types | 8 (includes **KIRC**, not LUSC) |
| Imbalance handling | SMOTE (k=4) |
| Architectures compared | 12 |
| Champion | LightGBM + SMOTE |
| Balanced accuracy | **95.0% ± 5.4%** |
| Multi-modal transformer | 83.2% ± 6.8% |
| Complexity–accuracy R² | 0.78 (p < 0.001) |
| External validation | ICGC ARGO, n=76, **92.1% ± 6.8%** (no retraining) |

---

## Deep learning baselines (Study 2, manuscript)

| Model | Test balanced accuracy | vs LightGBM |
|-------|------------------------|-------------|
| MLP (6-layer) | 93.6% | −4.8 pp |
| TabTransformer | 91.6% | −6.8 pp |

---

## Numbers that must NOT appear in revised text

These come from older manuscript versions or repo marketing and contradict the canonical record:

| Wrong | Correct |
|-------|---------|
| 4,913 samples | 1,248 (Study 2) or 158 (Study 1) |
| 1,200 samples | 1,248 |
| 96.5% / 97.6% / 95.33% | 98.4% (Study 2 test) or 95.0% (Study 1) |
| 270 features | 4,063 (Study 2) or 110 (Study 1) |
| Transformer as champion | LightGBM (or LR tied at 98.4%) |
| "Zero synthetic" for Study 1 | SMOTE used in Study 1 only |

---

## Repository map (what reviewers should run)

| Purpose | Path |
|---------|------|
| **Manuscript Study 2 reproduction** | `src/pipeline/` |
| Study 2 outputs | `data/real_model_results/` |
| Imbalance robustness (planned) | `imbalance_stress_test.py` |
| Demo (NOT manuscript data) | `cancer_genomics_ai_demo_minimal/` |
| Legacy / experimental | Root-level `*_50k_*`, `ultra_*` scripts — **not cited in paper** |
