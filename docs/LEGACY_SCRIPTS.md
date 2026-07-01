# Legacy scripts

Scripts in [`archive/legacy_root/scripts/`](../archive/legacy_root/scripts/) are historical utilities from early development. **Do not use them for JBI manuscript reproduction.**

## Use instead

| Goal | Path |
|------|------|
| Study 2 (98.4%) | [`src/pipeline/`](../src/pipeline/) |
| Study 1 | [`src/pipeline_study1/`](../src/pipeline_study1/) |
| Imbalance test | [`experiments/imbalance_stress_test.py`](../experiments/imbalance_stress_test.py) |
| Official numbers | [`docs/CANONICAL.md`](CANONICAL.md) |

## Script categories

| Scripts | Purpose |
|---------|---------|
| `compare_models_real_data.py`, `compare_models_no_synthetic.py` | Model comparisons on local CSV data |
| `create_*package*.py`, `convert_*word*.py`, `generate_figures.py` | Manuscript / packaging utilities |
| `inventory_real_tcga.py`, `tcga_dataset_assessment.py`, `audit_datasets.py` | Data inventory |
| `run_full_pipeline.py` | Older orchestration — superseded by `src/pipeline/` |
| `serve_demo.py`, `test_updated_demo.py` | Demo helpers (use `cancer_genomics_ai_demo_minimal/` instead) |

Scaling experiments (`*_50k_*`, `ultra_*`) are in [`archive/experiments/`](../archive/experiments/).
