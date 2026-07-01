# Oncura interactive demo

Streamlit prototype for a **multi-modal cancer classification workflow**:

1. **Input** — sample data, manual features, or CSV upload (110 genomic features, 6 modalities)
2. **Classify** — predict one of 8 TCGA cancer types with confidence scores
3. **Explain** — SHAP attribution by feature and modality

This is a **workflow demonstration**, not the manuscript reproduction pipeline. For **98.4%** Study 2 results, use [`../src/pipeline/`](../src/pipeline/).

---

## Quick start

```bash
cd demo
pip install -r requirements_streamlit.txt
./start_demo.sh
```

Browser: **http://localhost:8501**

From repo root: `./start_demo.sh`

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
