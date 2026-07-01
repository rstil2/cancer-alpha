"""
Methylation Feature Extraction Script

- Parses SeSAMe beta value files for each sample
- Selects top 1,000 most variable CpG sites
- Outputs samples x 1,000 methylation matrix
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

sample_matrices = []
sample_ids = []

DATA_ROOT = Path("data/raw_tcga/")  # Root directory for TCGA files
OUTPUT_DIR = Path("data/methylation_output/")
TOP_N_CPG = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Recursively find all SeSAMe beta value files (.tsv or .csv)
beta_files = []
for date_dir in DATA_ROOT.iterdir():
    if date_dir.is_dir():
        for cancer_dir in date_dir.iterdir():
            if cancer_dir.is_dir():
                files = glob.glob(str(cancer_dir / "*.beta_values.tsv"))
                files += glob.glob(str(cancer_dir / "*.beta_values.csv"))
                beta_files.extend(files)

sample_matrices = []
sample_ids = []

for file in beta_files:
    try:
        df = pd.read_csv(file, sep=None, engine='python')
        # Assume CpG sites are rows, columns are sample or vice versa
        if df.shape[0] > df.shape[1]:
            values = df.iloc[:, 1:].values  # skip first column if it's CpG name
        else:
            values = df.iloc[1:, :].values  # skip header row
        sample_matrices.append(values)
        sample_ids.append(os.path.basename(file).split('.')[0])
    except Exception as e:
        print(f"Error reading {file}: {e}")

if len(sample_matrices) == 0:
    raise RuntimeError("No methylation files found or parsed.")

all_matrix = np.vstack(sample_matrices)

# Calculate variance for each CpG site
cpg_variances = np.var(all_matrix, axis=0)

# Select top N most variable CpGs
top_cpg_idx = np.argsort(cpg_variances)[-TOP_N_CPG:]

# Extract top CpG features for each sample
top_matrix = all_matrix[:, top_cpg_idx]

# Save output
output_df = pd.DataFrame(top_matrix, index=sample_ids)
output_df.to_pickle(OUTPUT_DIR / "methylation_features.pkl")
output_df.to_csv(OUTPUT_DIR / "methylation_features.csv")

print(f"Saved methylation feature matrix: {top_matrix.shape} to {OUTPUT_DIR}")
