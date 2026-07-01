#!/usr/bin/env python3
"""Build a self-contained Oncura-Demo.zip for Windows, macOS, and Linux."""

from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
DEFAULT_OUT = REPO_ROOT / "dist" / "Oncura-Demo.zip"

INCLUDE = [
    "README.md",
    "QUICKSTART.md",
    "INSTALL.txt",
    "streamlit_app.py",
    "usage_tracker.py",
    "requirements_streamlit.txt",
    "setup.py",
    "start_demo.sh",
    "start_demo.bat",
    "Start Oncura Demo.bat",
    "Start Oncura Demo.command",
    "models/feature_selector.pkl",
    "models/label_encoder.pkl",
    "models/model_metadata.json",
    "models/multimodal_real_tcga_logistic_regression.pkl",
    "models/multimodal_real_tcga_random_forest.pkl",
    "models/scalers.pkl",
]

EXECUTABLE_SUFFIXES = {".sh", ".command"}


def build(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [p for p in INCLUDE if not (DEMO_DIR / p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing demo files: {missing}")

    prefix = "Oncura-Demo"
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE:
            src = DEMO_DIR / rel
            arcname = f"{prefix}/{rel.replace(chr(92), '/')}"
            info = zipfile.ZipInfo(arcname)
            info.external_attr = (stat.S_IRUSR | stat.S_IWUSR) << 16
            if Path(rel).suffix in EXECUTABLE_SUFFIXES:
                info.external_attr |= (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) << 16
            with src.open("rb") as fh:
                zf.writestr(info, fh.read())

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB, {len(INCLUDE)} files)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
