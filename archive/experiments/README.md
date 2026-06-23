# Archived experimental scripts

These scripts were used during exploratory TCGA scaling (50k samples, multi-omics integration, transformer training). They are **not** part of the JBI manuscript reproduction path.

## Manuscript reproduction (use instead)

- Study 2: [`src/pipeline/`](../../src/pipeline/)
- Study 1 map: [`src/pipeline_study1/`](../../src/pipeline_study1/)
- Robustness: [`experiments/imbalance_stress_test.py`](../../experiments/imbalance_stress_test.py)

## Why archived

- Mixed feature engineering quality (some runs report ~16% accuracy on 9k samples)
- Synthetic data generators in demo package, not here — but naming overlap confused reviewers
- Hardcoded paths and duplicate downloaders

Scripts remain available for historical reference but may not run without local data paths.
