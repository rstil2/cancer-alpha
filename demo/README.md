# Oncura interactive demo

Streamlit prototype for a **multi-modal cancer classification workflow**:

1. **Input** — sample data, manual features, or CSV upload (110 genomic features, 6 modalities)
2. **Classify** — predict one of 8 TCGA cancer types with confidence scores
3. **Explain** — SHAP attribution by feature and modality

---

## Download (recommended)

[![Download Demo ZIP](https://img.shields.io/badge/Download-Oncura--Demo.zip-2563eb?style=for-the-badge)](https://github.com/rstil2/cancer-alpha/releases/download/demo/Oncura-Demo.zip)

Cross-platform ZIP (~same file for Windows, macOS, Linux). Requires [Python 3.8+](https://www.python.org/downloads/).

| OS | After unzipping |
|----|-----------------|
| **Windows** | Double-click `start_demo.bat` |
| **macOS / Linux** | `chmod +x start_demo.sh && ./start_demo.sh` |

Then open **http://localhost:8501**. See `INSTALL.txt` inside the ZIP.

Rebuild ZIP locally: `python build_package.py` → `../dist/Oncura-Demo.zip`

This is a **workflow demonstration**, not the manuscript reproduction pipeline. Study 2 (**98.4%**) lives in [`../src/pipeline/`](../src/pipeline/).

---

## Quick start (from git clone)

```bash
cd demo
pip install -r requirements_streamlit.txt
./start_demo.sh
```

Browser: **http://localhost:8501** · From repo root: `./start_demo.sh`

### First run in the UI

1. Sidebar → **Sample Data**
2. Choose **Cancer Sample** → **Generate Sample Data**
3. Review prediction, class probabilities, and SHAP plots

---

## What's under the hood

| Component | Detail |
|-----------|--------|
| Models | Logistic Regression & Random Forest trained on 158 real TCGA samples |
| Features | 110 (methylation, mutation, CNA, fragmentomics, clinical, expression proxies) |
| Classes | BRCA, LUAD, COAD, PRAD, STAD, KIRC, HNSC, LIHC |
| Sample data | Synthetic patterns for demo inputs (~70% accuracy on generated samples) |

Research pipeline comparison: [docs/CANONICAL.md](../docs/CANONICAL.md)

---

## Docker

```bash
cd demo
docker compose up --build
```

Streamlit at **http://localhost:8501**

---

## Files

| File | Role |
|------|------|
| `streamlit_app.py` | UI and inference |
| `models/` | Pre-trained demo classifiers + scalers |
| `requirements_streamlit.txt` | Python dependencies |
| `start_demo.sh` / `start_demo.bat` | Launch scripts |

Optional: `pip install shap` for full explanation plots.

---

## Disclaimer

Research and demonstration software only. **Not for clinical use.**
