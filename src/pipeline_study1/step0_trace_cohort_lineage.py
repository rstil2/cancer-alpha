#!/usr/bin/env python3
"""Study 1 Step 0: Trace provenance of the n=158 cohort and document gaps."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR, PROJECT_ROOT, TARGET_CLASS_COUNTS

BARCODE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")
SEARCH_SUFFIXES = (".json", ".csv", ".txt", ".md", ".py", ".pkl", ".npy", ".npz")
SKIP_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", "archive", "cancer_genomics_ai_demo_minimal",
}


def _search_repo_for_barcodes() -> dict[str, list[str]]:
    hits: dict[str, set[str]] = {}
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SEARCH_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(PROJECT_ROOT))
        if rel.startswith("data/study1_results/"):
            continue
        try:
            if path.stat().st_size > 5_000_000:
                continue
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        found = set(BARCODE_RE.findall(text))
        if len(found) >= 5:
            hits[str(path.relative_to(PROJECT_ROOT))] = found
    return {k: sorted(v) for k, v in hits.items()}


def _inspect_synthetic_tcga() -> dict:
    tcga_dir = PROJECT_ROOT / "data" / "tcga"
    info = {"path": str(tcga_dir), "exists": tcga_dir.exists(), "files": []}
    if not tcga_dir.exists():
        return info
    for npy in sorted(tcga_dir.glob("*.npy")):
        arr = np.load(npy, mmap_mode="r")
        info["files"].append({
            "name": npy.name,
            "shape": list(arr.shape),
            "synthetic_layout": npy.name.startswith("gene_") or "synthetic" in npy.name.lower(),
        })
    return info


def main():
    print("=" * 60)
    print("Study 1 Step 0: Cohort lineage trace")
    print("=" * 60)

    report: dict = {
        "manuscript_target_n": sum(TARGET_CLASS_COUNTS.values()),
        "target_class_counts": TARGET_CLASS_COUNTS,
        "original_158_barcode_list_found": False,
        "findings": [],
    }

    # Current reproduced cohort
    train_path = OUTPUT_DIR / "train_patient_ids.json"
    if train_path.exists():
        with open(train_path) as f:
            current_ids = json.load(f)
        report["current_reproduced_train_ids"] = {
            "path": str(train_path.relative_to(PROJECT_ROOT)),
            "n": len(current_ids),
            "sample": current_ids[:10],
        }

    # Search for files listing ~158 TCGA barcodes
    barcode_hits = _search_repo_for_barcodes()
    report["files_with_multiple_tcga_barcodes"] = {
        k: {"n_barcodes": len(v), "sample": v[:8]} for k, v in barcode_hits.items()
    }

    pinned_candidates = []
    for path, ids in barcode_hits.items():
        if 150 <= len(ids) <= 165:
            pinned_candidates.append({"path": path, "n": len(ids)})
    report["candidate_pinned_lists_150_165"] = pinned_candidates

    legacy_model = PROJECT_ROOT / "models" / "performance_metrics_production.json"
    if legacy_model.exists():
        with open(legacy_model) as f:
            legacy = json.load(f)
        report["legacy_production_script_metrics"] = legacy
        report["findings"].append(
            "models/performance_metrics_production.json reports 46.5% CV on 158 samples "
            "(synthetic generator in purged lightgbm_smote_production.py)."
        )

    report["synthetic_data_tcga_npy"] = _inspect_synthetic_tcga()
    if report["synthetic_data_tcga_npy"]["files"]:
        report["findings"].append(
            "data/tcga/*.npy holds layout-matched synthetic arrays (no TCGA barcodes), "
            "not the manuscript's claimed authentic cohort."
        )

    if not pinned_candidates:
        report["findings"].append(
            "No committed file contains a pinned list of ~158 TCGA patient barcodes. "
            "The manuscript 95.0% figure is not tied to a reproducible ID manifest in this repo."
        )
    else:
        report["original_158_barcode_list_found"] = True
        report["findings"].append(
            f"Found {len(pinned_candidates)} file(s) with ~158 barcodes; "
            "inspect candidate_pinned_lists_150_165 before pinning."
        )

    report["recommended_actions"] = [
        "Use step3 deterministic selection (modality completeness) for stable cohort pinning.",
        "Store chosen IDs in data/study1_results/canonical_patient_manifest.json.",
        "Run step2b_icgc_fetch_features.py for real ICGC external features via Xena hub.",
    ]

    out = OUTPUT_DIR / "cohort_lineage_report.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\nFindings:")
    for line in report["findings"]:
        print(f"  - {line}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
