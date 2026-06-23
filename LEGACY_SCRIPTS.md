# Legacy root scripts

Scripts at the **repository root** (outside `src/pipeline/`) are mostly historical utilities from the Warp build era. **Do not use them for JBI manuscript reproduction.**

## Use instead

| Goal | Path |
|------|------|
| Study 2 (98.4%) | [`src/pipeline/`](src/pipeline/) |
| Study 1 map | [`src/pipeline_study1/`](src/pipeline_study1/) |
| Imbalance test | [`experiments/imbalance_stress_test.py`](experiments/imbalance_stress_test.py) |
| Official numbers | [`docs/CANONICAL.md`](docs/CANONICAL.md) |

## Root script categories

| Scripts | Purpose |
|---------|---------|
| `compare_models_real_data.py`, `compare_models_no_synthetic.py` | Model comparisons on local CSV data |
| `create_*package*.py`, `convert_*word*.py`, `generate_figures.py` | Manuscript / packaging utilities |
| `inventory_real_tcga.py`, `tcga_dataset_assessment.py`, `audit_datasets.py` | Data inventory |
| `train_transformer_real_tcga_balanced.py`, `train_massive_tcga_models.py` | Exploratory training (not canonical) |
| `process_authentic_tcga_55k.py`, `process_massive_tcga_mutations.py` | Large-scale processing experiments |
| `run_full_pipeline.py` | Older orchestration — superseded by `src/pipeline/` |
| `serve_demo.py`, `test_updated_demo.py` | Demo helpers |

## Archived scaling experiments

Moved to [`archive/experiments/`](archive/experiments/) (`*_50k_*`, `ultra_*`, `massive_*`, multi-omics downloaders).
