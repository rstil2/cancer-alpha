# Oncura Demo — native desktop app

Double-click **Oncura Demo** to open a self-contained classification workflow:

1. **Load sample** (cancer-like or control-like genomic features)
2. **Classify** → predicted cancer type + confidence
3. **Chart** → per-type probabilities

No Python or Terminal required when using the packaged app.

---

## Download

[![Mac app](https://img.shields.io/badge/Mac-Oncura--Demo.app-555555?style=for-the-badge&logo=apple)](https://github.com/rstil2/cancer-alpha/releases/download/demo/Oncura-Demo-mac.zip)
[![Windows exe](https://img.shields.io/badge/Windows-Oncura--Demo.exe-0078d4?style=for-the-badge&logo=windows)](https://github.com/rstil2/cancer-alpha/releases/download/demo/Oncura-Demo-Windows.exe)

| Platform | File | How to run |
|----------|------|------------|
| **macOS** | `Oncura-Demo-mac.zip` | Unzip → double-click **Oncura Demo.app** |
| **Windows** | `Oncura-Demo-Windows.exe` | Double-click the `.exe` |

**macOS first launch:** right-click the app → **Open** → **Open** (unsigned developer).

---

## Build from source

From the **cloned repository** (not the downloaded app ZIP):

```bash
git clone https://github.com/rstil2/cancer-alpha.git
cd cancer-alpha/demo
python3 -m venv .build-venv
.build-venv/bin/pip install -r requirements_build.txt
.build-venv/bin/python build_native.py
# → ../dist/Oncura-Demo-mac.zip or Oncura-Demo-Windows.exe
```

Developer Streamlit version (needs Python): `./start_demo.sh`

**Mac Gatekeeper:** unsigned CI builds need right-click → Open once. With an Apple Developer account, sign + notarize — see [SIGNING.md](SIGNING.md).

---

## Disclaimer

Research demonstration only. **Not for clinical use.**
