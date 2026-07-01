#!/usr/bin/env python3
"""Study 1 Step 3: Integrate modalities into 110 features and build n≈158 cohort."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohort_selection import (
    load_completeness_scores,
    select_cohort,
    write_canonical_manifest,
)
from config import (
    FEATURE_COUNTS,
    MUTATION_GENES,
    OUTPUT_DIR,
    TARGET_CLASS_COUNTS,
    TOTAL_FEATURES,
)


def _ordered_feature_names(meth_probes: list[str]) -> list[str]:
    names = [f"meth_{p}" for p in meth_probes]
    names += ["tmb_nonsyn", "tmb_total", "frameshift_count"]
    names += [f"mut_{g}" for g in MUTATION_GENES]
    names += [
        "cn_n_segments", "cn_mean_segment_mb", "cn_std_segment_mb",
        "cn_weighted_mean", "cn_weighted_std", "cn_frac_amp", "cn_frac_del",
        "cn_max_amp", "cn_max_del", "cn_genome_altered",
        "cn_chr1", "cn_chr2", "cn_chr3", "cn_chr4", "cn_chr5",
        "cn_chr6", "cn_chr7", "cn_chr8", "cn_chr9", "cn_chr10",
    ]
    names += [
        "frag_median_kb", "frag_mean_kb", "frag_std_kb", "frag_short_median_kb",
        "frag_long_median_kb", "frag_short_long_ratio", "frag_amp_burden",
        "frag_del_burden", "frag_abs_mean", "frag_abs_std", "frag_skew",
        "frag_high_amp_frac", "frag_high_del_frac",
        "frag_heterogeneity", "frag_length_entropy",
    ]
    names += [
        "clin_age", "clin_gender", "clin_vital", "clin_stage", "clin_grade",
        "clin_days_to_death", "clin_days_to_last_followup", "clin_tumor_status",
        "clin_neoplasm", "clin_lymph_nodes",
    ]
    names += [f"icgc_{g}" for g in [
        "ESR1", "ERBB2", "EGFR", "KRAS", "BRAF", "PIK3CA", "TP53", "BRCA1",
        "BRCA2", "MET", "ALK", "RET", "FGFR2", "CDK4", "CCND1", "MDM2",
        "PTEN", "STK11", "NRAS", "MYC",
    ]]
    assert len(names) == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES}, got {len(names)}"
    return names


def load_pickle_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    return pd.read_pickle(path)


def main():
    print("=" * 60)
    print("Study 1 Step 3: Build 110-feature cohort")
    print("=" * 60)

    required = [
        "methylation_raw.pkl", "mutation_raw.pkl", "copy_number_raw.pkl",
        "fragmentomics_raw.pkl", "icgc_expression_raw.pkl", "methylation_probe_panel.json",
        "patient_cancer_types.csv",
    ]
    for fname in required:
        if not (OUTPUT_DIR / fname).exists():
            print(f"ERROR: missing {fname}. Run step2 first.")
            sys.exit(1)

    meth = load_pickle_dict(OUTPUT_DIR / "methylation_raw.pkl")
    mut = load_pickle_dict(OUTPUT_DIR / "mutation_raw.pkl")
    cn = load_pickle_dict(OUTPUT_DIR / "copy_number_raw.pkl")
    frag = load_pickle_dict(OUTPUT_DIR / "fragmentomics_raw.pkl")
    icgc = load_pickle_dict(OUTPUT_DIR / "icgc_expression_raw.pkl")
    clin = load_pickle_dict(OUTPUT_DIR / "clinical_raw.pkl")
    cancer_types = pd.read_csv(OUTPUT_DIR / "patient_cancer_types.csv", index_col=0)["cancer_type"]

    with open(OUTPUT_DIR / "methylation_probe_panel.json") as f:
        meth_probes = json.load(f)
    feature_names = _ordered_feature_names(meth_probes)

    # Require core genomic modalities (clinical optional — imputed)
    core_sets = [set(meth), set(mut), set(cn), set(frag), set(icgc)]
    common = set.intersection(*core_sets)
    print(f"Patients with methylation+mutation+CN+fragmentomics+ICGC-proxy: {len(common)}")

    rows = []
    labels = []
    patient_ids = []
    for pid in sorted(common):
        ct = cancer_types.get(pid)
        if ct not in TARGET_CLASS_COUNTS:
            continue
        row = {}
        row.update(meth.get(pid, {}))
        row.update(mut.get(pid, {}))
        row.update(cn.get(pid, {}))
        row.update(frag.get(pid, {}))
        row.update(clin.get(pid, {k: np.nan for k in feature_names if k.startswith("clin_")}))
        row.update(icgc.get(pid, {}))
        rows.append([row.get(k, np.nan) for k in feature_names])
        labels.append(ct)
        patient_ids.append(pid)

    X = pd.DataFrame(rows, index=patient_ids, columns=feature_names, dtype=float)
    y = pd.Series(labels, index=patient_ids, name="cancer_type")

    print("\nAvailable per class before subsampling:")
    print(y.value_counts().sort_index())

    completeness = load_completeness_scores()
    print(f"\nModality completeness scores loaded for {len(completeness)} patients")

    selected_idx = select_cohort(y, TARGET_CLASS_COUNTS, completeness, method="deterministic")
    for ct, target in TARGET_CLASS_COUNTS.items():
        have = sum(1 for pid in selected_idx if y.get(pid) == ct)
        if have < target:
            print(f"  WARNING: {ct} has {have} selected, target {target}")

    X_cohort = X.loc[selected_idx].copy()
    y_cohort = y.loc[selected_idx].copy()

    # Median impute within cohort
    medians = X_cohort.median(numeric_only=True)
    X_cohort = X_cohort.fillna(medians)

    print(f"\nFinal cohort: {len(X_cohort)} samples, {X_cohort.shape[1]} features")
    print(y_cohort.value_counts().sort_index())

    X_cohort.to_pickle(OUTPUT_DIR / "features_110.pkl")
    y_cohort.to_csv(OUTPUT_DIR / "labels.csv", header=True)
    pd.Series(feature_names).to_csv(OUTPUT_DIR / "feature_names.csv", index=False, header=["feature"])
    medians.to_pickle(OUTPUT_DIR / "imputation_medians.pkl")
    train_ids = list(X_cohort.index)
    with open(OUTPUT_DIR / "train_patient_ids.json", "w") as f:
        json.dump(train_ids, f, indent=2)
    write_canonical_manifest(train_ids, y, "deterministic_modality_completeness")

    dataset_info = {
        "total_samples": int(len(X_cohort)),
        "n_features": int(X_cohort.shape[1]),
        "feature_layout": FEATURE_COUNTS,
        "cancer_types": sorted(y_cohort.unique().tolist()),
        "class_counts": y_cohort.value_counts().to_dict(),
        "target_class_counts": TARGET_CLASS_COUNTS,
        "data_source": "real_tcga_production_tcga",
        "cohort_selection": "deterministic_modality_completeness",
        "modalities_required": ["methylation", "mutation", "copy_number", "fragmentomics_proxy", "icgc_expression_proxy"],
        "clinical_imputed": True,
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nSaved cohort to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
