#!/usr/bin/env python3
"""Step 2: Extract expression features from real TCGA RNA-seq files.

Reads STAR gene count TSVs, extracts TPM for protein-coding genes,
applies log2 transform and variance filtering.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_CSV = PROJECT_ROOT / "data" / "file_patient_mapping.csv"
DATA_DIR = PROJECT_ROOT / "data" / "production_tcga" / "expression"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_model_results"

TOP_N_GENES = 2000  # Keep top N most variable genes


def parse_star_gene_counts(filepath):
    """Parse a STAR augmented gene counts TSV, return gene_name -> TPM dict."""
    gene_tpm = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or line.startswith("N_"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            gene_id, gene_name, gene_type = parts[0], parts[1], parts[2]
            if gene_type != "protein_coding":
                continue
            try:
                tpm = float(parts[6])  # tpm_unstranded column
            except (ValueError, IndexError):
                continue
            # Use gene_name; handle duplicates by keeping max TPM
            if gene_name not in gene_tpm or tpm > gene_tpm[gene_name]:
                gene_tpm[gene_name] = tpm
    return gene_tpm


def main():
    print("=" * 60)
    print("Step 2: Extracting expression features from real TCGA data")
    print("=" * 60)

    # Load file mapping
    if not MAPPING_CSV.exists():
        print(f"ERROR: {MAPPING_CSV} not found. Run step1 first.")
        sys.exit(1)

    mapping_df = pd.read_csv(MAPPING_CSV)
    expr_map = mapping_df[
        (mapping_df["data_type"] == "expression")
        & (mapping_df["sample_type"] == "Primary Tumor")
        & (mapping_df["has_local_file"] == True)
    ].copy()

    # Deduplicate: keep one file per patient per cancer type
    expr_map = expr_map.drop_duplicates(subset=["case_submitter_id", "project_id"], keep="first")
    print(f"\nPrimary Tumor expression files: {len(expr_map)}")
    print("Per cancer type:")
    for ct in sorted(expr_map["project_id"].unique()):
        n = len(expr_map[expr_map["project_id"] == ct])
        print(f"  {ct}: {n}")

    # Parse all expression files
    print(f"\nParsing {len(expr_map)} expression files...")
    all_samples = {}
    for idx, (_, row) in enumerate(expr_map.iterrows()):
        filepath = row["local_path"]
        patient_id = row["case_submitter_id"]
        cancer_type = row["project_id"]

        gene_tpm = parse_star_gene_counts(filepath)
        if gene_tpm:
            all_samples[patient_id] = {
                "cancer_type": cancer_type,
                "genes": gene_tpm,
            }

        if (idx + 1) % 100 == 0:
            print(f"  Parsed {idx+1}/{len(expr_map)} files...")

    print(f"  Successfully parsed {len(all_samples)} samples")

    # Build gene universe: genes present in >90% of samples
    print("\nBuilding gene universe...")
    gene_counts = {}
    for sample in all_samples.values():
        for gene in sample["genes"]:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1

    min_samples = int(0.9 * len(all_samples))
    common_genes = sorted([g for g, c in gene_counts.items() if c >= min_samples])
    print(f"  Genes in >90% samples: {len(common_genes)}")

    # Build expression matrix
    print("Building expression matrix...")
    patient_ids = sorted(all_samples.keys())
    expr_matrix = np.zeros((len(patient_ids), len(common_genes)), dtype=np.float32)

    for i, pid in enumerate(patient_ids):
        genes = all_samples[pid]["genes"]
        for j, gene in enumerate(common_genes):
            expr_matrix[i, j] = genes.get(gene, 0.0)

    # Log2(TPM + 1) transform
    expr_matrix = np.log2(expr_matrix + 1)

    # Variance filter: keep top N genes
    variances = np.var(expr_matrix, axis=0)
    top_indices = np.argsort(variances)[-TOP_N_GENES:]
    top_indices = np.sort(top_indices)  # Keep sorted order

    expr_filtered = expr_matrix[:, top_indices]
    top_gene_names = [common_genes[i] for i in top_indices]

    print(f"  Final expression matrix: {expr_filtered.shape}")
    print(f"  Top variance genes kept: {TOP_N_GENES}")

    # Build metadata DataFrame
    cancer_types = [all_samples[pid]["cancer_type"] for pid in patient_ids]
    metadata = pd.DataFrame({
        "patient_id": patient_ids,
        "cancer_type": cancer_types,
    })

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    expr_df = pd.DataFrame(expr_filtered, columns=top_gene_names, index=patient_ids)
    expr_df.to_pickle(OUTPUT_DIR / "expression_features.pkl")
    metadata.to_csv(OUTPUT_DIR / "expression_metadata.csv", index=False)

    # Also save gene names for interpretability
    pd.Series(top_gene_names).to_csv(OUTPUT_DIR / "expression_gene_names.csv", index=False, header=False)

    print(f"\nSaved to {OUTPUT_DIR}:")
    print(f"  expression_features.pkl: {expr_filtered.shape}")
    print(f"  expression_metadata.csv: {len(metadata)} samples")
    print(f"  expression_gene_names.csv: {len(top_gene_names)} genes")

    # Summary stats
    print(f"\nPer cancer type in final dataset:")
    for ct in sorted(metadata["cancer_type"].unique()):
        n = len(metadata[metadata["cancer_type"] == ct])
        print(f"  {ct}: {n} samples")

    print("\nDone!")


if __name__ == "__main__":
    main()
