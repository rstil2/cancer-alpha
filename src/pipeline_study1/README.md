# Study 1 reproduction pipeline (small-n, n≈158, 110 features)

Rebuilds the Study 1 cohort from **real TCGA files** in `data/production_tcga/` using the **KIRC panel** (not LUSC).

## Quick start

```bash
cd /path/to/cancer-alpha
python src/pipeline_study1/run_all.py
```

Or step by step:

```bash
python src/pipeline_study1/step1_file_mapping.py   # GDC map + KIRC (needs network)
python src/pipeline_study1/step2_extract_features.py
python src/pipeline_study1/step3_build_cohort.py
python src/pipeline_study1/step4_train_evaluate.py
```

## Outputs (`data/study1_results/`)

| File | Description |
|------|-------------|
| `features_110.pkl` | Final feature matrix (n≈158 × 110) |
| `labels.csv` | Cancer type per patient |
| `dataset_info.json` | Cohort summary and class counts |
| `model_results.json` | LightGBM+SMOTE CV scores |
| `lightgbm_smote_study1.pkl` | Trained pipeline |

## Feature layout (110)

| Modality | n | Source |
|----------|---|--------|
| Methylation | 20 | Top variable CpGs (frozen in `methylation_probe_panel.json`) |
| Mutation | 25 | TMB summaries + 22 driver genes |
| Copy number | 20 | Segment statistics + chromosome means |
| Fragmentomics | 15 | **Proxies** from CN segment length/amp profiles |
| Clinical | 10 | TCGA clinical XML (`data_integration/tcga_large_cache/clinical/`) |
| ICGC proxy | 20 | log2(TPM+1) of 20 pan-cancer genes from expression |

## Cohort selection

Patients must have **methylation + mutation + copy number + expression** (for ICGC proxy).  
Per-class counts are subsampled to manuscript targets in `config.TARGET_CLASS_COUNTS` (sum = 158).

## Not yet implemented

- **ICGC ARGO controlled tier** — DACO approval required; step2b uses open **ICGC Xena hub** data instead
- **True TabTransformer / 58M transformer** — step4b uses MLP proxies for deep rows

## ICGC external validation (step 2b)

```bash
pip install xenaPython
ICGC_SKIP_CN=1 python src/pipeline_study1/step2b_icgc_fetch_features.py  # CN fetch is slow
```

Outputs land in `data/icgc_argo/`; step5 scores them automatically.

## Cohort pinning (step 0 + 3)

- `step0_trace_cohort_lineage.py` documents whether a legacy 158-barcode list exists
- Step 3 uses **deterministic** selection by modality completeness (not random seed)
- Pinned IDs: `data/study1_results/canonical_patient_manifest.json`

## Steps

| Step | Script |
|------|--------|
| 4b | `step4b_benchmark_architectures.py` — 12 models, complexity–accuracy R² |
| 5 | `step5_external_validation.py` — frozen LightGBM+SMOTE on held-out TCGA (n=76, 4 types) |
| 3b | `step3b_build_external_cohort.py` — builds external feature matrix |

## Manuscript comparison

Official historical numbers: **95.0% ± 5.4%** (10 runs × 5-fold CV).  
After running step4, compare `model_results.json` to `docs/CANONICAL.md`.
