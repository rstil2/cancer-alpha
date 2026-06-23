# Oncura — Cancer Genomics AI Classifier (Demo)

**Interactive UI demo** — not the manuscript reproduction pipeline.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research pipeline](../RESEARCH.md)](../RESEARCH.md)

## What this is

A **Streamlit front-end** to explore classification and SHAP explainability. It uses:

- Demo models: Logistic Regression & Random Forest
- Training context: **158 TCGA samples**, **110 features** (Study 1 scale)
- UI sample data: illustrative (~70% accuracy on generated inputs)

For **98.4% Study 2 results**, run [`src/pipeline/`](../src/pipeline/) — see [RESEARCH.md](../RESEARCH.md).

## Research results (from manuscript — not this demo)

| Study | Setting | Best result |
|-------|---------|-------------|
| **Study 2** | 1,248 balanced, 4,063 features | LightGBM **98.4%** (held-out TCGA test) |
| **Study 1** | 158 imbalanced + SMOTE, 110 features | LightGBM+SMOTE **95.0%** |
| **Study 1 external** | ICGC ARGO n=76 | **92.1%** (no retraining) |

Full-cohort rankings (Study 2 test): LightGBM & LR **98.4%**, XGBoost 98.0%, RF 97.2%.

Paper: [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1) · [Main README](../README.md)

## Quick start

```bash
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
# or: python setup.py && ./start_demo.sh
```

Open **http://localhost:8501**

## Demo vs research pipeline

| | This demo | Research (`src/pipeline/`) |
|---|-----------|----------------------------|
| Purpose | UI exploration | Paper reproduction |
| Data | Sample / 158-sample models | Real TCGA multi-omics |
| Accuracy | ~70% on demo inputs | **98.4%** Study 2 |
| Features | 110 | 4,063 |

## Disclaimer

Research software only — **not for clinical use**. See [LICENSE](../LICENSE).
