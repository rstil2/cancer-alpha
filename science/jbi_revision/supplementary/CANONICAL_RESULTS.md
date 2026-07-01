# Canonical Results — Single Source of Truth

Use these numbers in **GitHub, reviewer responses, and revision drafts** (`working/`).  
**Do not cite legacy README values (4,913 samples, 96.5%, 97.6%) unless marked historical.**

| Artifact | Role |
|----------|------|
| `submitted_snapshot/Combined_Manuscript_JBI.pdf` | **Frozen** — what JBI received 2026-06-21. Do not edit. |
| `working/Combined_Manuscript_JBI_REVISED.docx` | Editable revision draft when invited. |
| This file + `docs/CANONICAL.md` | Numbers that must match `src/pipeline/` outputs. |

---

## Study 2 (large balanced) — canonical pipeline

**Verified:** `data/real_model_results/model_results.json` (2026-03-08)

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

## Study 1 (small-n) — submitted vs reproduced

Study 1 has **two rows** in the canonical record. Only one should appear in a given document.

### A. Submitted manuscript (frozen 2026-06-21)

*What JBI reviewers see today. Do not change `submitted_snapshot/`.*

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

### B. Reproduced pipeline (2026-06-23)

*What `src/pipeline_study1/` produces from `data/production_tcga/`.*

**Pipeline:** `src/pipeline_study1/run_all.py`  
**Primary outputs:** `data/study1_results/model_results.json`, `external_validation_results.json`, `data/icgc_argo/features_110.pkl`

| Parameter | Value |
|-----------|-------|
| Samples | 158 (class targets in `config.TARGET_CLASS_COUNTS`) |
| Cohort selection | Deterministic by modality completeness; IDs in `canonical_patient_manifest.json` |
| Legacy 158 barcode list | **Not found** — see `cohort_lineage_report.json` |
| Features | 110; fragmentomics = CN proxies; ICGC block = expression gene proxies on TCGA |
| LightGBM + SMOTE CV | **82.4% ± 2.6%** (10 runs × 5-fold) |
| TCGA held-out external | **83.7%** balanced accuracy (n=68; BRCA/LUAD/COAD/PRAD; no retrain) |
| ICGC Xena external | **25.0%** balanced accuracy (n=76; non-US projects; mutation real, methylation imputed, no expression on hub) |
| 12-model benchmark | Re-run `step4b_benchmark_architectures.py` after cohort pin (stale JSON may predate step 3) |
| Complexity–accuracy R² (reproduced) | ~0.03 (p ≈ 0.6) — **does not reproduce MS claim** |

**Revision guidance:** If JBI invites revision, replace Study 1 headline numbers in `working/` with section B **or** add a limitations paragraph that reproduced CV is ~82% and ICGC partial validation is ~25% until DACO-backed ARGO harmonization is complete. Do **not** silently equate A and B.

---

## Deep learning baselines (Study 2, manuscript)

| Model | Test balanced accuracy | vs LightGBM |
|-------|------------------------|-------------|
| MLP (6-layer) | 93.6% | −4.8 pp |
| TabTransformer | 91.6% | −6.8 pp |

---

## Numbers that must NOT appear in revised text (unless marked historical)

| Wrong / stale | Use instead |
|---------------|-------------|
| 4,913 samples | 1,248 (Study 2) or 158 (Study 1) |
| 1,200 samples | 1,248 |
| 96.5% / 97.6% / 95.33% | 98.4% (Study 2 test) |
| 95.0% Study 1 as “reproduced” | 82.4% CV from `pipeline_study1` **or** label as submitted-only |
| 92.1% ICGC as “reproduced” | 25.0% partial Xena **or** label as submitted-only |
| 270 features | 4,063 (Study 2) or 110 (Study 1) |
| Transformer as champion | LightGBM (or LR tied at 98.4%) |
| “Zero synthetic” globally | SMOTE in Study 1 only |
| R² = 0.78 (Study 1 complexity) | Reproduced ~0.03 unless step4b re-run shows otherwise |

---

## Repository map (what reviewers should run)

| Purpose | Path |
|---------|------|
| **Manuscript Study 2 reproduction** | `src/pipeline/` |
| Study 2 outputs | `data/real_model_results/` |
| **Study 1 reproduction** | `src/pipeline_study1/` |
| Study 1 outputs | `data/study1_results/`, `data/icgc_argo/` |
| Imbalance robustness | `experiments/imbalance_stress_test.py` |
| Frozen submitted PDF | `science/jbi_revision/submitted_snapshot/Combined_Manuscript_JBI.pdf` |
| Editable revision draft | `science/jbi_revision/working/Combined_Manuscript_JBI_REVISED.docx` |
| Demo (NOT manuscript data) | `cancer_genomics_ai_demo_minimal/` |
