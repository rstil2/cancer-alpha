#!/usr/bin/env python3
"""Study 1 Step 3b: Build held-out TCGA external cohort (ICGC-matched 4-type panel, n=76)."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_utils import build_feature_matrix, common_patients, load_modality_dicts, load_meth_probes
from cohort_selection import load_completeness_scores, select_patients
from config import OUTPUT_DIR

# Manuscript ICGC ARGO panel (4 types, n=76)
ICGC_EXTERNAL_TYPES = {"TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD", "TCGA-PRAD"}
ICGC_TARGET_COUNTS = {
    "TCGA-BRCA": 28,
    "TCGA-LUAD": 22,
    "TCGA-COAD": 16,
    "TCGA-PRAD": 10,
}


def main():
    print("=" * 60)
    print("Study 1 Step 3b: Build external validation cohort")
    print("=" * 60)

    train_ids_path = OUTPUT_DIR / "train_patient_ids.json"
    if not train_ids_path.exists():
        print("ERROR: run step3 first (train_patient_ids.json missing).")
        sys.exit(1)

    with open(train_ids_path) as f:
        train_ids = set(json.load(f))

    meth, mut, cn, frag, icgc, clin, cancer_types = load_modality_dicts()
    meth_probes = load_meth_probes()
    pool = sorted(common_patients(meth, mut, cn, frag, icgc) - train_ids)
    print(f"Held-out patients (full intersection, not in training): {len(pool)}")

    X_all, y_all = build_feature_matrix(
        pool, meth_probes, meth, mut, cn, frag, icgc, clin, cancer_types,
        allowed_types=ICGC_EXTERNAL_TYPES,
    )
    print(f"Held-out in ICGC-matched types: {len(X_all)}")
    print(y_all.value_counts().sort_index())

    completeness = load_completeness_scores()
    selected = []
    for ct, target in ICGC_TARGET_COUNTS.items():
        candidates = y_all[y_all == ct].index.tolist()
        n = min(target, len(candidates))
        if n < target:
            print(f"  WARNING: {ct} held-out only {len(candidates)}, target {target}")
        chosen = select_patients(candidates, ct, n, completeness, method="deterministic")
        selected.extend(chosen)

    X_ext = X_all.loc[selected].copy()
    y_ext = y_all.loc[selected].copy()

    # Impute using training medians when available
    medians_path = OUTPUT_DIR / "imputation_medians.pkl"
    if medians_path.exists():
        medians = pd.read_pickle(medians_path)
        X_ext = X_ext.fillna(medians)
    else:
        X_ext = X_ext.fillna(X_ext.median())

    X_ext.to_pickle(OUTPUT_DIR / "external_features_110.pkl")
    y_ext.to_csv(OUTPUT_DIR / "external_labels.csv", header=True)

    info = {
        "total_samples": int(len(X_ext)),
        "class_counts": y_ext.value_counts().to_dict(),
        "target_class_counts": ICGC_TARGET_COUNTS,
        "cancer_types": sorted(ICGC_EXTERNAL_TYPES),
        "source": "tcga_held_out_not_in_training_158",
        "note": (
            "Same 110-feature schema as Study 1 training cohort; patients disjoint from "
            "the 158 training set. Manuscript ICGC ARGO used independent samples — "
            "place real ICGC features at data/icgc_argo/features_110.pkl to score those."
        ),
    }
    with open(OUTPUT_DIR / "external_dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nExternal cohort: {len(X_ext)} samples")
    print(y_ext.value_counts().sort_index())
    print(f"Saved -> {OUTPUT_DIR / 'external_features_110.pkl'}")


if __name__ == "__main__":
    main()
