# Oncura

Multi-modal TCGA cancer classification — interactive demo and reproducible research code.

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.07.22.666135-blue)](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1)
[![Manuscript](https://img.shields.io/badge/JBI-under%20review-orange)](science/Combined_Manuscript_JBI.pdf)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use%20Only-red.svg)](LICENSE)

**Author:** [R. Craig Stillwell, PhD](mailto:craig.stillwell@gmail.com)

> **Research only.** Not a medical device. Do not use for diagnosis or treatment decisions.

---

## Download the demo app

Native desktop app — **double-click the Oncura icon**. No Python install required.

[![Download for Mac](https://img.shields.io/badge/Download-Mac%20(.app)-555555?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/rstil2/cancer-alpha/releases/download/demo/Oncura-Demo-mac.zip)
[![Download for Windows](https://img.shields.io/badge/Download-Windows%20(.exe)-0078d4?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/rstil2/cancer-alpha/releases/download/demo/Oncura-Demo-Windows.exe)

| Platform | Steps |
|----------|--------|
| **macOS** | Download `Oncura-Demo-mac.zip` → unzip → double-click **Oncura Demo.app** |
| **Windows** | Download `Oncura-Demo-Windows.exe` → double-click to run |

**First launch (Mac):** if macOS blocks the app, right-click **Oncura Demo.app** → **Open** → **Open** once.

**Workflow in the app:** Load sample → Classify → view cancer type, confidence, and probability chart.

[All demo downloads](https://github.com/rstil2/cancer-alpha/releases/tag/demo)

---

## Research summary

Oncura tests whether **experimental design** matters more than **model architecture** for multi-modal TCGA classification.

**Study 2** (primary, reproducible): LightGBM **98.4%** held-out balanced accuracy, n=1,248 → [`src/pipeline/`](src/pipeline/)

Canonical metrics: [docs/CANONICAL.md](docs/CANONICAL.md) · Full detail: [RESEARCH.md](RESEARCH.md)

---

## Manuscript

**[*Experimental Design Dominates Model Architecture in Multi-Modal Cancer Classification*](science/Combined_Manuscript_JBI.pdf)** — *Journal of Biomedical Informatics*, submitted June 2026

Preprint: [bioRxiv 10.1101/2025.07.22.666135](https://www.biorxiv.org/content/10.1101/2025.07.22.666135v1)

---

## Reproduce Study 2

```bash
git clone https://github.com/rstil2/cancer-alpha.git
cd cancer-alpha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline/step4_train_evaluate.py
```

Expected: **~98.4%** test balanced accuracy.

---

## Repository layout

```
cancer-alpha/
├── demo/                  # Native desktop demo + source
├── src/pipeline/          # Study 2 reproduction
├── science/               # Manuscript
└── docs/                  # Canonical results
```

---

## License

Academic use under [LICENSE](LICENSE). Provisional patent 63/847,316 has **lapsed**; see [PATENTS.md](PATENTS.md).
