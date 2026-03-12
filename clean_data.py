#!/usr/bin/env python3
"""
Clean Real TCGA Data
===================
Remove NaN values from the processed real TCGA dataset.
"""

import pandas as pd
import numpy as np

def main():
    print('🔍 Inspecting real TCGA data for NaN values...')
    
    # Load data
    X_df = pd.read_csv('data/real_tcga_large/real_tcga_features.csv')
    y_df = pd.read_csv('data/real_tcga_large/real_tcga_labels.csv')
    
    print(f'Dataset shape: {X_df.shape}')
    
    # Check for NaN values
    nan_counts = X_df.isnull().sum()
    total_nans = nan_counts.sum()
    print(f'Total NaN values: {total_nans}')
    
    if total_nans > 0:
        print(f'Columns with NaNs: {nan_counts[nan_counts > 0].shape[0]}')
        print(f'Max NaNs in a column: {nan_counts.max()}')
        print(f'Percentage of data that is NaN: {total_nans / (X_df.shape[0] * X_df.shape[1]) * 100:.2f}%')
        
        # Clean the data by replacing NaNs with 0 (appropriate for gene expression)
        print('\n🧹 Cleaning data by replacing NaNs with 0...')
        X_cleaned = X_df.fillna(0)
        
        # Verify cleaning
        remaining_nans = X_cleaned.isnull().sum().sum()
        print(f'NaNs after cleaning: {remaining_nans}')
        
        # Save cleaned data
        X_cleaned.to_csv('data/real_tcga_large/real_tcga_features_cleaned.csv', index=False)
        print('✅ Saved cleaned features to: data/real_tcga_large/real_tcga_features_cleaned.csv')
        
        return True
        
    else:
        print('✅ No NaN values found - data is already clean!')
        # Copy to cleaned version anyway
        X_df.to_csv('data/real_tcga_large/real_tcga_features_cleaned.csv', index=False)
        return True

if __name__ == "__main__":
    main()