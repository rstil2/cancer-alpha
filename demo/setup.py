#!/usr/bin/env python3
"""Install demo dependencies and verify model files."""

import subprocess
import sys
from pathlib import Path

REQUIRED = [
    "models/multimodal_real_tcga_random_forest.pkl",
    "models/multimodal_real_tcga_logistic_regression.pkl",
    "models/scalers.pkl",
    "streamlit_app.py",
]


def main() -> None:
    if sys.version_info < (3, 8):
        print("Python 3.8+ required.")
        sys.exit(1)

    root = Path(__file__).parent
    missing = [p for p in REQUIRED if not (root / p).exists()]
    if missing:
        print("Missing files:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"]
    )
    print("Setup complete. Run: ./start_demo.sh")


if __name__ == "__main__":
    main()
