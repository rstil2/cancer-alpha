#!/usr/bin/env python3
"""
QUICK 50K DATASET SUMMARY
========================
Extract key quality insights from the 50,000 sample dataset
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 70)
    print("📊 QUICK 50K DATASET QUALITY SUMMARY")
    print("=" * 70)
    
    # Load dataset
    dataset_path = "data/ultra_permissive_50k_output/ultra_permissive_50k_plus_50000_20250822_184637.csv"
    df = pd.read_csv(dataset_path)
    
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Total features: {df.shape[1]}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Missing values
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    print(f"\n🔍 MISSING VALUES:")
    print(f"   Total missing: {total_missing}")
    print(f"   Perfect completeness: {total_missing == 0}")
    
    # Cancer type distribution
    if 'cancer_type' in df.columns:
        cancer_dist = df['cancer_type'].value_counts()
        print(f"\n🏥 CANCER TYPES:")
        print(f"   Unique types: {len(cancer_dist)}")
        print(f"   Range: {cancer_dist.min():,} - {cancer_dist.max():,} samples")
        print(f"   Top 5 cancer types:")
        for i, (cancer, count) in enumerate(cancer_dist.head(5).items(), 1):
            print(f"     {i}. {cancer}: {count:,} samples")
    
    # Multi-omics coverage
    omics_cols = ['has_expression', 'has_methylation', 'has_copy_number', 
                  'has_mutations', 'has_protein', 'has_clinical']
    
    print(f"\n🧬 MULTI-OMICS COVERAGE:")
    for col in omics_cols:
        if col in df.columns:
            coverage = df[col].sum()
            pct = (coverage / len(df)) * 100
            omics_name = col.replace('has_', '').title()
            print(f"   {omics_name}: {coverage:,} samples ({pct:.1f}%)")
    
    # Multi-omics distribution
    if 'num_data_types' in df.columns:
        multi_omics_dist = df['num_data_types'].value_counts().sort_index()
        print(f"\n📈 MULTI-OMICS DISTRIBUTION:")
        for num_types, count in multi_omics_dist.items():
            pct = (count / len(df)) * 100
            print(f"   {num_types} data types: {count:,} samples ({pct:.1f}%)")
        
        # Key quality metrics
        single_omics = (df['num_data_types'] == 1).sum()
        multi_omics_3plus = (df['num_data_types'] >= 3).sum()
        
        print(f"\n🎯 KEY QUALITY METRICS:")
        print(f"   Single-omics samples: {single_omics:,} ({single_omics/len(df)*100:.1f}%)")
        print(f"   Multi-omics (3+) samples: {multi_omics_3plus:,} ({multi_omics_3plus/len(df)*100:.1f}%)")
    
    # Data integrity checks
    print(f"\n✅ DATA INTEGRITY:")
    duplicates = df.duplicated().sum()
    duplicate_ids = df['sample_id'].duplicated().sum()
    missing_ids = df['sample_id'].isnull().sum()
    
    print(f"   Duplicate rows: {duplicates}")
    print(f"   Duplicate sample IDs: {duplicate_ids}")
    print(f"   Missing sample IDs: {missing_ids}")
    print(f"   Data integrity: {'✅ PASSED' if duplicates + duplicate_ids + missing_ids == 0 else '⚠️ ISSUES FOUND'}")
    
    # Statistical power
    cancer_types = df['cancer_type'].nunique() if 'cancer_type' in df.columns else 1
    samples_per_type = len(df) / cancer_types
    power_assessment = "EXCELLENT" if samples_per_type >= 1000 else "ADEQUATE" if samples_per_type >= 100 else "LIMITED"
    
    print(f"\n📊 STATISTICAL POWER:")
    print(f"   Samples per cancer type: {samples_per_type:.1f}")
    print(f"   Assessment: {power_assessment}")
    
    # Overall quality score
    quality_factors = []
    
    # Factor 1: Sample size (max 25 points)
    size_score = min(25, len(df) / 2000)
    quality_factors.append(("Sample Size", size_score, 25))
    
    # Factor 2: Multi-omics coverage (max 25 points)
    if 'num_data_types' in df.columns:
        avg_omics = df['num_data_types'].mean()
        omics_score = min(25, avg_omics * 5)
        quality_factors.append(("Multi-omics Coverage", omics_score, 25))
    
    # Factor 3: Cancer type diversity (max 25 points)
    diversity_score = min(25, cancer_types * 0.75)
    quality_factors.append(("Cancer Diversity", diversity_score, 25))
    
    # Factor 4: Data completeness (max 25 points)
    completeness_score = 25 if total_missing == 0 else max(0, 25 - total_missing/1000)
    quality_factors.append(("Data Completeness", completeness_score, 25))
    
    total_score = sum(score for _, score, _ in quality_factors)
    max_score = sum(max_val for _, _, max_val in quality_factors)
    
    print(f"\n🏆 OVERALL QUALITY ASSESSMENT:")
    for factor, score, max_val in quality_factors:
        print(f"   {factor}: {score:.1f}/{max_val}")
    
    overall_grade = "A+" if total_score >= 90 else "A" if total_score >= 80 else "B+" if total_score >= 70 else "B" if total_score >= 60 else "C"
    print(f"   TOTAL SCORE: {total_score:.1f}/{max_score} ({overall_grade})")
    
    print(f"\n{'🎉 EXCELLENT QUALITY DATASET!' if overall_grade.startswith('A') else '📊 GOOD QUALITY DATASET'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
