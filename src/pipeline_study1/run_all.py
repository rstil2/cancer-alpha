#!/usr/bin/env python3
"""Run Study 1 pipeline steps 1–4 sequentially."""

import subprocess
import sys
from pathlib import Path

STEPS = [
    "step0_trace_cohort_lineage.py",
    "step1_file_mapping.py",
    "step2_extract_features.py",
    "step3_build_cohort.py",
    "step3b_build_external_cohort.py",
    "step4_train_evaluate.py",
    "step4b_benchmark_architectures.py",
    "step2b_icgc_fetch_features.py",
    "step5_external_validation.py",
]


def main():
    root = Path(__file__).resolve().parent
    for step in STEPS:
        print(f"\n>>> Running {step}\n")
        result = subprocess.run([sys.executable, str(root / step)], check=False)
        if result.returncode != 0:
            print(f"FAILED: {step}")
            sys.exit(result.returncode)
    print("\nStudy 1 pipeline complete.")


if __name__ == "__main__":
    main()
