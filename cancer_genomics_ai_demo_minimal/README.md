# Oncura — Cancer Genomics AI Classifier (Demo)

**Interactive demo of the Oncura multi-modal cancer classification system**

[![Patent Protected](https://img.shields.io/badge/Patent-Protected-blue.svg)](../PATENTS.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Full System Performance

The complete Oncura system achieves the following on real TCGA clinical data:

| Setting | Samples | Features | Modalities | Best Accuracy |
|---------|---------|----------|------------|---------------|
| **Full cohort** | 1,248 (156/type) | 4,063 | Gene expression, DNA methylation, somatic mutations | **98.4%** (LightGBM) |
| **Minimal data** | 158 (imbalanced) | 110 | Methylation, mutations, CNA, fragmentomics, clinical, ICGC ARGO | **95.0%** (LightGBM+SMOTE) |
| **CUP validation** | 2,500 predictions | 4,063 | 3 modalities | **97.6%** balanced accuracy |

**Full-cohort model rankings (held-out test, n=250):**
- LightGBM: **98.4%** — 4/8 cancer types at 100% precision & recall
- Logistic Regression: **98.4%**
- XGBoost: **98.0%**
- Random Forest: **97.2%**

**Cancer types**: BRCA, LUAD, COAD, PRAD, STAD, HNSC, LUSC, LIHC

For full details, see the [main project README](../README.md) and published manuscript on [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1).

## What This Demo Includes

This demo provides a **simplified interactive interface** using models trained on the minimal-data setting (158 TCGA samples, 110 features). It demonstrates the classification workflow and SHAP explainability.

**Demo models:**
- Logistic Regression (80 selected features)
- Random Forest (110 features)

**Demo cancer types**: BRCA, LUAD, COAD, PRAD, STAD, KIRC, HNSC, LIHC

**Demo features:**
- Interactive Streamlit web interface
- Cancer type classification with confidence scoring
- SHAP-based feature importance explanations
- Modality-level importance breakdown
- Sample data generation for testing

## Quick Start

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the demo
streamlit run streamlit_app.py
```

Access the demo at **http://localhost:8501**

### Alternative: One-Command Setup
```bash
python setup.py && ./start_demo.sh
```

## Demo vs Full System

| Feature | This Demo | Full System |
|---------|-----------|-------------|
| **Samples** | 158 (imbalanced) | 1,248 (balanced, 156/type) |
| **Features** | 110 (6 modalities) | 4,063 (3 modalities) |
| **Models** | LR, Random Forest | LightGBM, LR, XGBoost, RF |
| **Accuracy** | Demo-level | 98.4% balanced accuracy |
| **Validation** | Cross-validation | 5-fold CV + held-out test (n=250) |
| **CUP validation** | Not included | 97.6% (2,500 predictions) |
| **Subtype prediction** | Not included | 80–92% across 3 cancer types |
| **Explainability** | SHAP feature importance | SHAP + biomarker validation |

## Requirements

- Python 3.8+
- 4GB RAM minimum
- Dependencies listed in `requirements_streamlit.txt`

## Patent Notice

This repository contains a demonstration of patent-protected technology.

- **Patent**: Provisional Application No. 63/847,316
- **Patent Holder**: Dr. R. Craig Stillwell
- **Commercial Use**: Requires separate patent license
- **Contact**: craig.stillwell@gmail.com

---

**© 2025–2026 Dr. R. Craig Stillwell. All rights reserved.**
