#!/usr/bin/env python3
"""Study 1 Step 2b: Fetch real ICGC donors from Xena and harmonize to 110 features."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_utils import ordered_feature_names
from config import OUTPUT_DIR, PROJECT_ROOT
from icgc_xena import (
    ICGC_HOST,
    ICGC_TARGET_COUNTS,
    clinical_features_from_donor,
    fetch_copy_number_features,
    fetch_donor_table,
    fetch_expression,
    fetch_mutations,
    select_icgc_donors,
    _available_sets,
)

ICGC_DIR = PROJECT_ROOT / "data" / "icgc_argo"
EXCLUDED_PROJECTS = sorted({
    "ALL-US", "BRCA-US", "COAD-US", "LUAD-US", "LUSC-US", "PRAD-US", "READ-US",
})


def main():
    print("=" * 60)
    print("Study 1 Step 2b: ICGC Xena fetch + 110-feature harmonization")
    print("=" * 60)

    try:
        import xenaPython  # noqa: F401
    except ImportError:
        print("ERROR: pip install xenaPython")
        sys.exit(1)

    ICGC_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = ICGC_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "methylation_probe_panel.json") as f:
        meth_probes = json.load(f)
    feature_names = ordered_feature_names(meth_probes)

    print("Fetching ICGC donor metadata from Xena hub...")
    donors = fetch_donor_table(ICGC_HOST)
    available = _available_sets(ICGC_HOST)
    print(f"  Eligible non-US donors with mappable labels: {len(donors)}")
    print(f"  Mutation donors: {len(available['mutation'] | available['mutation_non_us'])}")

    chosen = select_icgc_donors(donors, available, ICGC_TARGET_COUNTS)
    print(f"\nSelected {len(chosen)} donors (target {sum(ICGC_TARGET_COUNTS.values())}):")
    labels = pd.Series({r["donor_id"]: r["tcga_label"] for r in chosen})
    print(labels.value_counts().sort_index())

    manifest = {
        "source": "icgc_xena_hub",
        "host": ICGC_HOST,
        "excluded_projects": EXCLUDED_PROJECTS,
        "target_class_counts": ICGC_TARGET_COUNTS,
        "donors": chosen,
        "note": (
            "Real ICGC legacy hub data (non-US projects). Methylation unavailable — "
            "imputed from Study 1 training medians. ARGO controlled tier requires DACO."
        ),
    }
    with open(raw_dir / "icgc_donor_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    donor_ids = [r["donor_id"] for r in chosen]
    print("\nFetching expression (20 genes)...")
    expr = fetch_expression(ICGC_HOST, donor_ids)
    print(f"  donors with expression: {len(expr)}")

    print("Fetching mutations (25 features)...")
    mut = fetch_mutations(ICGC_HOST, donor_ids)
    print(f"  donors with mutations: {len(mut)}")

    print("Fetching copy number + fragmentomics proxies (chr1-10)...")
    skip_cn = os.environ.get("ICGC_SKIP_CN", "0") == "1"
    if skip_cn:
        print("  (skipped — set ICGC_SKIP_CN=0 to fetch CN segments; slow)")
        cn = {}
    else:
        cn = fetch_copy_number_features(ICGC_HOST, donor_ids)
    print(f"  donors with CN segments: {len(cn)}")

    medians_path = OUTPUT_DIR / "imputation_medians.pkl"
    train_medians = pd.read_pickle(medians_path) if medians_path.exists() else pd.Series(dtype=float)

    rows = []
    y = []
    kept = []
    for row in chosen:
        pid = row["donor_id"]
        feats = {f"meth_{p}": np.nan for p in meth_probes}
        feats.update(mut.get(pid, {}))
        feats.update(cn.get(pid, {}))
        feats.update(clinical_features_from_donor(row))
        feats.update(expr.get(pid, {}))
        rows.append([feats.get(k, np.nan) for k in feature_names])
        y.append(row["tcga_label"])
        kept.append(pid)

    X = pd.DataFrame(rows, index=kept, columns=feature_names, dtype=float)
    y_ser = pd.Series(y, index=kept, name="cancer_type")

    if len(train_medians):
        meth_cols = [c for c in feature_names if c.startswith("meth_")]
        clin_cols = [c for c in feature_names if c.startswith("clin_")]
        impute_from_train = meth_cols + clin_cols
        X[impute_from_train] = X[impute_from_train].fillna(train_medians.reindex(impute_from_train))
        remaining = [c for c in feature_names if c not in impute_from_train and not c.startswith("mut_")]
        X[remaining] = X[remaining].fillna(0.0)
    else:
        X = X.fillna(X.median()).fillna(0.0)

    X.to_pickle(ICGC_DIR / "features_110.pkl")
    y_ser.to_csv(ICGC_DIR / "labels.csv", header=True)

    info = {
        "n_samples": int(len(X)),
        "class_counts": y_ser.value_counts().to_dict(),
        "target_class_counts": ICGC_TARGET_COUNTS,
        "modalities_real": {
            "mutation": int(sum(1 for d in donor_ids if d in mut)),
            "expression_icgc": int(sum(1 for d in donor_ids if d in expr)),
            "copy_number_fragmentomics": int(sum(1 for d in donor_ids if d in cn)),
            "clinical_partial": int(len(chosen)),
            "methylation": 0,
        },
        "methylation_imputed_from_tcga_training": True,
        "cn_frag_icgc_imputed_zero_if_missing": True,
        "argo_daco_required_for_full_omics": True,
    }
    with open(ICGC_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved ICGC cohort -> {ICGC_DIR}")
    print(f"  features_110.pkl ({X.shape[0]} x {X.shape[1]})")
    print("  labels.csv")


if __name__ == "__main__":
    main()
