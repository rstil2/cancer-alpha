#!/usr/bin/env python3
"""
Quick TCGA Data Coverage Assessment
==================================

Quick check of current data coverage and progress toward 50,000+ sample target.

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import os
from pathlib import Path
from collections import defaultdict

def count_files_by_cancer_type():
    """Quick count of files by cancer type and data type."""
    base_dir = Path("data/production_tcga")
    
    data_types = ["expression", "mutations", "copy_number", "methylation", "protein", "clinical"]
    cancer_counts = defaultdict(lambda: defaultdict(int))
    total_files = 0
    
    print("Quick TCGA Data Coverage Assessment")
    print("=" * 50)
    
    for data_type in data_types:
        data_dir = base_dir / data_type
        if data_dir.exists():
            for cancer_dir in data_dir.iterdir():
                if cancer_dir.is_dir() and cancer_dir.name.startswith("TCGA-"):
                    file_count = len(list(cancer_dir.glob("*")))
                    cancer_counts[cancer_dir.name][data_type] = file_count
                    total_files += file_count
    
    # Display summary
    print(f"\nTotal files across all data types: {total_files:,}")
    print(f"\nBreakdown by cancer type:")
    print("-" * 80)
    
    for cancer_type in sorted(cancer_counts.keys()):
        total_cancer = sum(cancer_counts[cancer_type].values())
        print(f"{cancer_type}: {total_cancer:,} files")
        
        # Show breakdown by data type
        for data_type in data_types:
            count = cancer_counts[cancer_type][data_type]
            if count > 0:
                print(f"  {data_type}: {count:,} files")
    
    # Estimate unique samples (conservative estimate)
    estimated_samples = total_files // len(data_types)  # Very rough estimate
    print(f"\nEstimated samples (conservative): ~{estimated_samples:,}")
    print(f"Progress toward 50,000 target: {estimated_samples/50000*100:.1f}%")
    
    return cancer_counts, total_files

if __name__ == "__main__":
    counts, total = count_files_by_cancer_type()
