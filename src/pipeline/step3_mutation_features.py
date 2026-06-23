#!/usr/bin/env python3
"""Step 3: Extract mutation features from real TCGA MAF files.

Computes TMB, driver gene mutation status, and variant classification
distributions from GDC masked somatic mutation MAF files.
"""

import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_CSV = PROJECT_ROOT / "data" / "file_patient_mapping.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_model_results"

# Top 50 cancer driver genes (Cancer Gene Census + literature)
DRIVER_GENES = [
    "TP53", "KRAS", "PIK3CA", "APC", "BRAF", "PTEN", "EGFR", "CDKN2A",
    "RB1", "NF1", "ARID1A", "ATM", "BRCA1", "BRCA2", "CDH1", "CTNNB1",
    "ERBB2", "FBXW7", "GATA3", "IDH1", "KMT2D", "MAP3K1", "MTOR",
    "MYC", "NFE2L2", "NOTCH1", "NRAS", "PIK3R1", "PTCH1", "SETD2",
    "SMAD4", "SMARCA4", "SOX9", "STK11", "TERT", "VHL", "WT1",
    "AKT1", "ALK", "AXIN1", "BAP1", "CCND1", "CDK4", "CREBBP",
    "CTCF", "EP300", "FGFR2", "FGFR3", "HRAS", "KIT",
]

# Variant classifications of interest
NONSYNONYMOUS_TYPES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site",
}

ALL_VARIANT_CLASSES = [
    "Missense_Mutation", "Silent", "Nonsense_Mutation",
    "Splice_Site", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Other",
]


def parse_maf_file(filepath):
    """Parse a MAF file, return mutation summary dict and TCGA barcode."""
    open_fn = gzip.open if filepath.endswith(".gz") else open
    mode = "rt" if filepath.endswith(".gz") else "r"

    mutations = []
    barcode = None

    try:
        with open_fn(filepath, mode) as f:
            header = None
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if header is None:
                    header = parts
                    continue
                if len(parts) < len(header):
                    continue

                row = dict(zip(header, parts))
                gene = row.get("Hugo_Symbol", "")
                var_class = row.get("Variant_Classification", "")

                if barcode is None:
                    bc = row.get("Tumor_Sample_Barcode", "")
                    if bc.startswith("TCGA-"):
                        # Extract patient-level barcode (first 3 parts: TCGA-XX-XXXX)
                        barcode = "-".join(bc.split("-")[:3])

                mutations.append({
                    "gene": gene,
                    "variant_class": var_class,
                })
    except Exception as e:
        print(f"  Warning: could not parse {filepath}: {e}")
        return None, None

    if not mutations:
        return None, barcode

    # Compute features
    features = {}

    # TMB: count of nonsynonymous mutations
    nonsyn_count = sum(1 for m in mutations if m["variant_class"] in NONSYNONYMOUS_TYPES)
    features["tmb_nonsynonymous"] = nonsyn_count
    features["tmb_total"] = len(mutations)
    features["tmb_log"] = np.log1p(nonsyn_count)

    # Driver gene binary status
    mutated_genes = {m["gene"] for m in mutations if m["variant_class"] in NONSYNONYMOUS_TYPES}
    for gene in DRIVER_GENES:
        features[f"driver_{gene}"] = 1 if gene in mutated_genes else 0

    # Driver gene count
    features["driver_gene_count"] = sum(features[f"driver_{gene}"] for gene in DRIVER_GENES)

    # Variant classification distribution (fractions)
    total = len(mutations)
    class_counts = {}
    for m in mutations:
        vc = m["variant_class"]
        if vc not in set(ALL_VARIANT_CLASSES[:-1]):
            vc = "Other"
        class_counts[vc] = class_counts.get(vc, 0) + 1

    for vc in ALL_VARIANT_CLASSES:
        features[f"varclass_frac_{vc}"] = class_counts.get(vc, 0) / total if total > 0 else 0

    return features, barcode


def main():
    print("=" * 60)
    print("Step 3: Extracting mutation features from real TCGA MAFs")
    print("=" * 60)

    # Load file mapping
    if not MAPPING_CSV.exists():
        print(f"ERROR: {MAPPING_CSV} not found. Run step1 first.")
        sys.exit(1)

    mapping_df = pd.read_csv(MAPPING_CSV)
    mut_map = mapping_df[
        (mapping_df["data_type"] == "mutations")
        & (mapping_df["sample_type"] == "Primary Tumor")
        & (mapping_df["has_local_file"] == True)
    ].copy()

    # Deduplicate
    mut_map = mut_map.drop_duplicates(subset=["case_submitter_id", "project_id"], keep="first")
    print(f"\nPrimary Tumor mutation files: {len(mut_map)}")

    # Parse all MAF files
    print(f"\nParsing {len(mut_map)} MAF files...")
    all_features = {}
    for idx, (_, row) in enumerate(mut_map.iterrows()):
        filepath = row["local_path"]
        patient_id = row["case_submitter_id"]
        cancer_type = row["project_id"]

        features, maf_barcode = parse_maf_file(filepath)

        if features is not None:
            # Verify barcode matches if available
            if maf_barcode and maf_barcode != patient_id:
                # Use the barcode from the MAF file as ground truth
                patient_id = maf_barcode

            all_features[patient_id] = {
                "cancer_type": cancer_type,
                "features": features,
            }

        if (idx + 1) % 100 == 0:
            print(f"  Parsed {idx+1}/{len(mut_map)} files...")

    print(f"  Successfully parsed {len(all_features)} samples")

    # Build feature matrix
    print("\nBuilding mutation feature matrix...")
    patient_ids = sorted(all_features.keys())
    feature_names = sorted(all_features[patient_ids[0]]["features"].keys())

    mut_matrix = np.zeros((len(patient_ids), len(feature_names)), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        for j, fname in enumerate(feature_names):
            mut_matrix[i, j] = all_features[pid]["features"].get(fname, 0.0)

    print(f"  Mutation feature matrix: {mut_matrix.shape}")

    # Build metadata
    cancer_types = [all_features[pid]["cancer_type"] for pid in patient_ids]
    metadata = pd.DataFrame({
        "patient_id": patient_ids,
        "cancer_type": cancer_types,
    })

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mut_df = pd.DataFrame(mut_matrix, columns=feature_names, index=patient_ids)
    mut_df.to_pickle(OUTPUT_DIR / "mutation_features.pkl")
    metadata.to_csv(OUTPUT_DIR / "mutation_metadata.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}:")
    print(f"  mutation_features.pkl: {mut_matrix.shape}")
    print(f"  mutation_metadata.csv: {len(metadata)} samples")

    # Summary
    print(f"\nPer cancer type:")
    for ct in sorted(metadata["cancer_type"].unique()):
        n = len(metadata[metadata["cancer_type"] == ct])
        print(f"  {ct}: {n} samples")

    # Driver gene summary
    print(f"\nMost frequently mutated driver genes:")
    for gene in DRIVER_GENES[:10]:
        col = f"driver_{gene}"
        if col in feature_names:
            idx_col = feature_names.index(col)
            frac = np.mean(mut_matrix[:, idx_col])
            print(f"  {gene}: {frac*100:.1f}% of samples")

    print("\nDone!")


if __name__ == "__main__":
    main()
