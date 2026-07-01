# Oncura — Cancer Genomics AI Classifier (Demo)

**Interactive UI demo** — not the manuscript reproduction pipeline.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research pipeline](../RESEARCH.md)](../RESEARCH.md)

## What this is

A **Streamlit front-end** to explore classification and SHAP explainability. It uses demo models and illustrative sample data (~70% accuracy on generated inputs).

For **98.4% Study 2 results**, run [`src/pipeline/`](../src/pipeline/) — see [RESEARCH.md](../RESEARCH.md).

## Research results (not this demo)

| Setting | Best reproduced result | Pipeline |
|---------|------------------------|----------|
| 1,248 balanced TCGA, 4,063 features | LightGBM **98.4%** (held-out test) | [`src/pipeline/`](../src/pipeline/) |
| 158 imbalanced + SMOTE, 110 features | **82.4% ± 2.6%** CV | [`src/pipeline_study1/`](../src/pipeline_study1/) |

Submitted manuscript Study 1 values (95.0%, 92.1% ICGC) are **not reproduced** by the current pipeline. See [docs/CANONICAL.md](../docs/CANONICAL.md).

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
