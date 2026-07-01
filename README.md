# Oncura

Multi-modal cancer classification from TCGA genomics — research codebase and interactive demo.

**Start here for numbers and reproduction:** [RESEARCH.md](RESEARCH.md) · [docs/CANONICAL.md](docs/CANONICAL.md)

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.07.22.666135-blue)](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1)
[![Manuscript](https://img.shields.io/badge/JBI-under%20review-orange)](science/Combined_Manuscript_JBI.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use%20Only-red.svg)](LICENSE)

**Author:** [R. Craig Stillwell, PhD](mailto:craig.stillwell@gmail.com)

> **Research only.** Not a medical device. Do not use for diagnosis or treatment decisions.

---

## Summary

Oncura evaluates whether **experimental design** (cohort curation, leakage control, class balance) matters more than **model architecture** for multi-modal TCGA cancer classification.

The repository contains two complementary studies on the same 8-type task under different sample regimes. **Study 2** is the primary reproducible result; **Study 1** is a small-n sensitivity analysis with a documented gap between submitted manuscript numbers and the current pipeline.

| | Study 2 (primary) | Study 1 (small-n) |
|---|-------------------|-------------------|
| Samples | 1,248 balanced TCGA | 158 imbalanced TCGA |
| Features | 4,063 | 110 (6 modalities) |
| Best model | LightGBM | LightGBM + SMOTE |
| **Reproduced result** | **98.4%** test balanced accuracy | **82.4% ± 2.6%** CV |
| Validation | Internal TCGA held-out (n=250) | TCGA ext. 83.7%; ICGC partial 25.0% |
| Pipeline | [`src/pipeline/`](src/pipeline/) | [`src/pipeline_study1/`](src/pipeline_study1/) |

Study 1 **submitted manuscript** values (frozen PDF, not reproduced here): 95.0% ± 5.4% CV, 92.1% ICGC. See [docs/CANONICAL.md](docs/CANONICAL.md) for the full submitted vs reproduced table.

---

## Manuscript

**[*Experimental Design Dominates Model Architecture in Multi-Modal Cancer Classification*](science/Combined_Manuscript_JBI.pdf)**  
*Journal of Biomedical Informatics* — submitted June 2026

| Resource | Path |
|----------|------|
| Submitted PDF | [`science/Combined_Manuscript_JBI.pdf`](science/Combined_Manuscript_JBI.pdf) |
| Revision workspace | [`science/jbi_revision/`](science/jbi_revision/) |
| Canonical numbers | [`docs/CANONICAL.md`](docs/CANONICAL.md) |
| Reproduction guide | [`science/jbi_revision/supplementary/REPRODUCTION_GUIDE.md`](science/jbi_revision/supplementary/REPRODUCTION_GUIDE.md) |

Preprint: [bioRxiv 10.1101/2025.07.22.666135](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1)

---

## Reproduce Study 2

```bash
git clone https://github.com/rstil2/cancer-alpha.git
cd cancer-alpha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# If feature pickles already exist:
python src/pipeline/step4_train_evaluate.py

# Full pipeline from GDC-mapped files:
python src/pipeline/step1_file_mapping.py
python src/pipeline/step2_expression_features.py
python src/pipeline/step2b_methylation_features.py
python src/pipeline/step3_mutation_features.py
python src/pipeline/step4_train_evaluate.py
```

Expected LightGBM test balanced accuracy: **~98.4%** (`data/real_model_results/model_results.json`).

Study 1: `python src/pipeline_study1/run_all.py` → `data/study1_results/`

---

## Interactive demo

The Streamlit demo in [`cancer_genomics_ai_demo_minimal/`](cancer_genomics_ai_demo_minimal/) uses **illustrative sample data** (~70% accuracy). It does **not** reproduce the 98.4% research pipeline.

```bash
cd cancer_genomics_ai_demo_minimal
python setup.py
./start_demo.sh    # http://localhost:8501
```

---

## Repository layout

```
cancer-alpha/
├── src/pipeline/              # Study 2 — canonical reproduction
├── src/pipeline_study1/       # Study 1 — small-n pipeline
├── experiments/               # Robustness tests (imbalance stress, etc.)
├── science/                   # Manuscript and JBI revision workspace
├── docs/                      # Installation, data access, canonical results
├── cancer_genomics_ai_demo_minimal/  # Streamlit demo (not paper data)
├── archive/                   # Legacy scripts, internal notes, old artifacts
│   ├── legacy_root/scripts/ # Former root-level Python utilities
│   ├── experiments/         # Scaling / download experiments (50k era)
│   └── business/              # Historical commercial docs (not research claims)
├── models/                    # Saved model artifacts
└── data/                      # Local data (not in git; see DATA_ACCESS.md)
```

Historical root scripts and internal working documents live under [`archive/`](archive/). Do not use them for manuscript reproduction.

---

## Documentation

- [Research entry point](RESEARCH.md)
- [Canonical results](docs/CANONICAL.md)
- [Data access](docs/DATA_ACCESS.md)
- [Legacy scripts index](docs/LEGACY_SCRIPTS.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

---

## Citation

```bibtex
@article{stillwell2025oncura,
  title   = {Oncura: Multi-Modal AI for Precision Oncology},
  author  = {Stillwell, R. Craig},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.07.22.666135}
}
```

---

## License

Academic and non-commercial research use under [LICENSE](LICENSE). Commercial use requires a separate agreement — contact **craig.stillwell@gmail.com**. Provisional patent 63/847,316 has **lapsed**; see [PATENTS.md](PATENTS.md).
