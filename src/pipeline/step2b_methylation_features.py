#!/usr/bin/env python3
"""Step 2b: Extract methylation features from real TCGA methylation beta value files.

Reads SeSAMe level3betas.txt files (~482K CpG probes per file),
applies NA filtering, variance filtering to top N probes.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_CSV = PROJECT_ROOT / "data" / "file_patient_mapping.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_model_results"

TOP_N_PROBES = 2000  # Keep top N most variable CpG probes
MAX_NA_FRAC = 0.1  # Drop probes with >10% NA across samples


def parse_methylation_betas(filepath):
    """Parse a SeSAMe level3betas.txt file, return probe_id -> beta dict."""
    probes = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            probe_id, beta_str = parts
            if not probe_id.startswith("cg"):
                continue
            if beta_str == "NA" or beta_str == "":
                probes[probe_id] = np.nan
            else:
                try:
                    probes[probe_id] = float(beta_str)
                except ValueError:
                    probes[probe_id] = np.nan
    return probes


def main():
    print("=" * 60)
    print("Step 2b: Extracting methylation features from real TCGA data")
    print("=" * 60)

    # Load file mapping
    if not MAPPING_CSV.exists():
        print(f"ERROR: {MAPPING_CSV} not found. Run step1 first.")
        sys.exit(1)

    mapping_df = pd.read_csv(MAPPING_CSV)
    meth_map = mapping_df[
        (mapping_df["data_type"] == "methylation")
        & (mapping_df["sample_type"] == "Primary Tumor")
        & (mapping_df["has_local_file"] == True)
    ].copy()

    # Deduplicate: keep one file per patient per cancer type
    meth_map = meth_map.drop_duplicates(subset=["case_submitter_id", "project_id"], keep="first")
    print(f"\nPrimary Tumor methylation files: {len(meth_map)}")
    print("Per cancer type:")
    for ct in sorted(meth_map["project_id"].unique()):
        n = len(meth_map[meth_map["project_id"] == ct])
        print(f"  {ct}: {n}")

    # Parse all methylation files - two-pass approach for memory efficiency
    # Pass 1: Get probe universe from first few files
    print(f"\nPass 1: Determining probe universe from first 10 files...")
    probe_universe = None
    for idx, (_, row) in enumerate(meth_map.head(10).iterrows()):
        probes = parse_methylation_betas(row["local_path"])
        if probe_universe is None:
            probe_universe = set(probes.keys())
        else:
            probe_universe &= set(probes.keys())

    print(f"  Common probes across first 10 files: {len(probe_universe)}")

    # Pass 2: Parse all files, collecting beta values for common probes
    print(f"\nPass 2: Parsing all {len(meth_map)} methylation files...")
    patient_ids = []
    cancer_types = []
    all_betas = []  # list of dicts

    for idx, (_, row) in enumerate(meth_map.iterrows()):
        filepath = row["local_path"]
        patient_id = row["case_submitter_id"]
        cancer_type = row["project_id"]

        probes = parse_methylation_betas(filepath)
        if probes:
            patient_ids.append(patient_id)
            cancer_types.append(cancer_type)
            all_betas.append(probes)

        if (idx + 1) % 100 == 0:
            print(f"  Parsed {idx+1}/{len(meth_map)} files...")

    print(f"  Successfully parsed {len(patient_ids)} samples")

    # Build probe counts for NA filtering
    print("\nFiltering probes by NA rate...")
    probe_counts = {}
    for betas in all_betas:
        for probe_id, val in betas.items():
            if probe_id not in probe_counts:
                probe_counts[probe_id] = {"total": 0, "valid": 0}
            probe_counts[probe_id]["total"] += 1
            if not np.isnan(val):
                probe_counts[probe_id]["valid"] += 1

    min_valid = int((1 - MAX_NA_FRAC) * len(patient_ids))
    min_present = int(0.9 * len(patient_ids))  # Present in >90% of samples
    valid_probes = sorted([
        p for p, c in probe_counts.items()
        if c["valid"] >= min_valid and c["total"] >= min_present and p.startswith("cg")
    ])
    print(f"  Probes with <{MAX_NA_FRAC*100:.0f}% NA and >90% presence: {len(valid_probes)}")

    # Build methylation matrix
    print("Building methylation matrix...")
    meth_matrix = np.full((len(patient_ids), len(valid_probes)), np.nan, dtype=np.float32)
    probe_to_idx = {p: i for i, p in enumerate(valid_probes)}

    for i, betas in enumerate(all_betas):
        for probe_id, val in betas.items():
            if probe_id in probe_to_idx:
                meth_matrix[i, probe_to_idx[probe_id]] = val

    # Impute remaining NAs with column median
    print("Imputing remaining NAs with probe medians...")
    na_count = np.isnan(meth_matrix).sum()
    for j in range(meth_matrix.shape[1]):
        col = meth_matrix[:, j]
        mask = np.isnan(col)
        if mask.any():
            median_val = np.nanmedian(col)
            meth_matrix[mask, j] = median_val
    print(f"  Imputed {na_count} NA values ({na_count / meth_matrix.size * 100:.2f}%)")

    # Variance filter: keep top N probes
    print(f"Applying variance filter (top {TOP_N_PROBES})...")
    variances = np.var(meth_matrix, axis=0)
    top_indices = np.argsort(variances)[-TOP_N_PROBES:]
    top_indices = np.sort(top_indices)

    meth_filtered = meth_matrix[:, top_indices]
    top_probe_names = [valid_probes[i] for i in top_indices]

    print(f"  Final methylation matrix: {meth_filtered.shape}")

    # Build metadata DataFrame
    metadata = pd.DataFrame({
        "patient_id": patient_ids,
        "cancer_type": cancer_types,
    })

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meth_df = pd.DataFrame(meth_filtered, columns=top_probe_names, index=patient_ids)
    meth_df.to_pickle(OUTPUT_DIR / "methylation_features.pkl")
    metadata.to_csv(OUTPUT_DIR / "methylation_metadata.csv", index=False)

    # Also save probe names
    pd.Series(top_probe_names).to_csv(OUTPUT_DIR / "methylation_probe_names.csv", index=False, header=False)

    print(f"\nSaved to {OUTPUT_DIR}:")
    print(f"  methylation_features.pkl: {meth_filtered.shape}")
    print(f"  methylation_metadata.csv: {len(metadata)} samples")
    print(f"  methylation_probe_names.csv: {len(top_probe_names)} probes")

    # Summary stats
    print(f"\nPer cancer type in final dataset:")
    for ct in sorted(metadata["cancer_type"].unique()):
        n = len(metadata[metadata["cancer_type"] == ct])
        print(f"  {ct}: {n} samples")

    # Beta value distribution summary
    print(f"\nBeta value statistics (filtered matrix):")
    print(f"  Mean: {np.mean(meth_filtered):.4f}")
    print(f"  Std: {np.std(meth_filtered):.4f}")
    print(f"  Min: {np.min(meth_filtered):.4f}")
    print(f"  Max: {np.max(meth_filtered):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
