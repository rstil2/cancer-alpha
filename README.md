# Oncura

Multi-modal TCGA cancer classification — interactive demo and reproducible research code.

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.07.22.666135-blue)](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1)
[![Manuscript](https://img.shields.io/badge/JBI-under%20review-orange)](science/Combined_Manuscript_JBI.pdf)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use%20Only-red.svg)](LICENSE)

**Author:** [R. Craig Stillwell, PhD](mailto:craig.stillwell@gmail.com)

> **Research only.** Not a medical device. Do not use for diagnosis or treatment decisions.

---

## Try the demo (2 minutes)

Streamlit app for the **upload → classify → explain** workflow: multi-modal genomic input, cancer-type prediction, confidence scores, and SHAP feature attribution.

```bash
git clone https://github.com/rstil2/cancer-alpha.git
cd cancer-alpha/demo
pip install -r requirements_streamlit.txt
./start_demo.sh
```

Open **http://localhost:8501**

| | Demo | Research pipeline |
|---|------|-------------------|
| Purpose | Interactive workflow prototype | Paper reproduction |
| Models | LR & Random Forest (158-sample training set) | LightGBM **98.4%** on Study 2 |
| Data | Built-in sample generator or CSV upload | Real TCGA via GDC |
| Run | `demo/start_demo.sh` | `src/pipeline/step4_train_evaluate.py` |

Docker: `cd demo && docker compose up` (Streamlit on port 8501).

Full demo docs: [`demo/README.md`](demo/README.md)

---

## Research summary

Oncura tests whether **experimental design** matters more than **model architecture** for multi-modal TCGA classification.

**Study 2** (primary, reproducible): LightGBM **98.4%** held-out balanced accuracy, n=1,248, 4,063 features → [`src/pipeline/`](src/pipeline/)

**Study 1** (small-n sensitivity analysis): documented in [RESEARCH.md](RESEARCH.md) with submitted vs reproduced numbers.

Canonical metrics: [docs/CANONICAL.md](docs/CANONICAL.md)

---

## Manuscript

**[*Experimental Design Dominates Model Architecture in Multi-Modal Cancer Classification*](science/Combined_Manuscript_JBI.pdf)** — *Journal of Biomedical Informatics*, submitted June 2026

Preprint: [bioRxiv 10.1101/2025.07.22.666135](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1)

---

## Reproduce Study 2

```bash
cd cancer-alpha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline/step4_train_evaluate.py   # expects feature pickles in data/
```

Expected: **~98.4%** test balanced accuracy. Guide: [science/jbi_revision/supplementary/REPRODUCTION_GUIDE.md](science/jbi_revision/supplementary/REPRODUCTION_GUIDE.md)

---

## Repository layout

```
cancer-alpha/
├── demo/                  # Streamlit workflow demo (start here for interviews)
├── src/pipeline/          # Study 2 reproduction
├── src/pipeline_study1/   # Study 1 small-n pipeline
├── science/               # Manuscript & revision workspace
├── docs/                  # Canonical results, data access
└── archive/               # Legacy scripts (not for papers)
```

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

## License

Academic use under [LICENSE](LICENSE). Provisional patent 63/847,316 has **lapsed**; see [PATENTS.md](PATENTS.md).
