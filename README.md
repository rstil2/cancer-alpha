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

Oncura asks whether **experimental design** (cohort curation, leakage control, class balance) matters more than **model architecture** for multi-modal TCGA cancer classification.

The **reproducible headline result** is **Study 2**: LightGBM reaches **98.4%** held-out balanced accuracy on 1,248 balanced TCGA samples (4,063 features, 8 cancer types). Classical models converge within ~1 pp; deep baselines lag by several points on the same split.

The submitted JBI manuscript also reports a **small-n sensitivity analysis** (Study 1, n=158). That regime is documented in [RESEARCH.md](RESEARCH.md) with **submitted vs reproduced** numbers — the current pipeline does **not** reproduce the manuscript's 95% / 92.1% ICGC claims.

---

## Primary result (Study 2)

| Item | Value |
|------|-------|
| Samples | 1,248 (156/class, real TCGA) |
| Features | 4,063 (expression + methylation + mutations) |
| Train / test | 998 / 250 (held-out) |
| Champion | LightGBM **98.4%** test balanced accuracy (logistic regression tied) |
| Deep baselines | MLP 93.6%, TabTransformer 91.6% |
| Validation | Internal TCGA held-out split only |
| Pipeline | [`src/pipeline/`](src/pipeline/) |

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
| Study 1 (small-n) detail | [RESEARCH.md](RESEARCH.md) |

Preprint: [bioRxiv 10.1101/2025.07.22.666135](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1)

---

## Reproduce

```bash
git clone https://github.com/rstil2/cancer-alpha.git
cd cancer-alpha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Study 2 — if feature pickles already exist:
python src/pipeline/step4_train_evaluate.py

# Study 2 — full pipeline from GDC-mapped files:
python src/pipeline/step1_file_mapping.py
python src/pipeline/step2_expression_features.py
python src/pipeline/step2b_methylation_features.py
python src/pipeline/step3_mutation_features.py
python src/pipeline/step4_train_evaluate.py
```

Expected LightGBM test balanced accuracy: **~98.4%** (`data/real_model_results/model_results.json`).

Study 1 (exploratory): `python src/pipeline_study1/run_all.py` → see [RESEARCH.md](RESEARCH.md) for expected scores.

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
├── src/pipeline_study1/       # Study 1 — small-n pipeline (see RESEARCH.md)
├── experiments/               # Robustness tests (imbalance stress, etc.)
├── science/                   # Manuscript and JBI revision workspace
├── docs/                      # Installation, data access, canonical results
├── cancer_genomics_ai_demo_minimal/  # Streamlit demo (not paper data)
└── archive/                   # Legacy scripts and internal notes
```

---

## Documentation

- [Research entry point (both studies)](RESEARCH.md)
- [Canonical results](docs/CANONICAL.md)
- [Data access](docs/DATA_ACCESS.md)
- [Contributing](CONTRIBUTING.md)

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
