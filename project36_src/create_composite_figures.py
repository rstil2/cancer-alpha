#!/usr/bin/env python3
"""
Script to combine multi-panel figures into composite figures for publication.
This script takes individual figure files and combines them into publication-ready
composite figures with proper labeling (A, B, etc.).
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np
import os

def create_composite_figure(fig_paths, output_path, title, panel_labels=None, figsize=(16, 8)):
    """
    Create a composite figure from multiple individual figures.
    
    Parameters:
    fig_paths (list): List of paths to individual figure files
    output_path (str): Path for the output composite figure
    title (str): Title for the composite figure
    panel_labels (list): Labels for each panel (A, B, etc.)
    figsize (tuple): Figure size (width, height)
    """
    if panel_labels is None:
        panel_labels = [chr(65 + i) for i in range(len(fig_paths))]  # A, B, C, ...
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(fig_paths), figsize=figsize)
    if len(fig_paths) == 1:
        axes = [axes]
    
    # Load and display each image
    for i, (fig_path, ax, label) in enumerate(zip(fig_paths, axes, panel_labels)):
        if os.path.exists(fig_path):
            img = mpimg.imread(fig_path)
            ax.imshow(img)
            ax.axis('off')
            
            # Add panel label
            ax.text(0.02, 0.98, f'{label}', transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'Figure not found:\n{fig_path}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red')
            ax.axis('off')
    
    # Add main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the composite figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Created composite figure: {output_path}")

def main():
    """Main function to create all composite figures."""
    
    # Define paths
    results_dir = "manuscript_submission_package/results_figures"
    output_dir = "composite_figures"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define composite figures to create
    composite_figures = [
        {
            'files': [
                os.path.join(results_dir, 'correlation_matrix.png'),
                os.path.join(results_dir, 'feature_distributions.png')
            ],
            'output': os.path.join(output_dir, 'Figure_1_Multi_modal_Integration.png'),
            'title': 'Figure 1. Multi-modal Data Integration and Correlation Analysis',
            'labels': ['A', 'B'],
            'figsize': (20, 10)
        },
        {
            'files': [
                os.path.join(results_dir, 'pca_scatter.png'),
                os.path.join(results_dir, 'pca_analysis.png')
            ],
            'output': os.path.join(output_dir, 'Figure_2_PCA_Analysis.png'),
            'title': 'Figure 2. Principal Component Analysis of Integrated Genomic Features',
            'labels': ['A', 'B'],
            'figsize': (16, 8)
        },
        {
            'files': [
                os.path.join(results_dir, 'roc_curves.png'),
                os.path.join(results_dir, 'model_comparison.png')
            ],
            'output': os.path.join(output_dir, 'Figure_3_Model_Performance.png'),
            'title': 'Figure 3. Model Performance Comparison',
            'labels': ['A', 'B'],
            'figsize': (16, 8)
        },
        {
            'files': [
                os.path.join(results_dir, 'shap_summary.png'),
                os.path.join(results_dir, 'top_discriminative_features.png')
            ],
            'output': os.path.join(output_dir, 'Figure_4_Feature_Importance.png'),
            'title': 'Figure 4. Feature Importance Analysis',
            'labels': ['A', 'B'],
            'figsize': (20, 10)
        }
    ]
    
    # Single panel figures to copy
    single_figures = [
        {
            'file': os.path.join(results_dir, 'modality_contributions.png'),
            'output': os.path.join(output_dir, 'Figure_5_Modality_Contributions.png'),
            'title': 'Figure 5. Modality Contributions to Cancer Detection'
        },
        {
            'file': os.path.join(results_dir, 'cancer_hallmarks.png'),
            'output': os.path.join(output_dir, 'Figure_6_Cancer_Hallmarks.png'),
            'title': 'Figure 6. Cancer Hallmarks Mapping'
        }
    ]
    
    print("Creating composite figures for publication...")
    print("=" * 60)
    
    # Create composite figures
    for fig_info in composite_figures:
        create_composite_figure(
            fig_paths=fig_info['files'],
            output_path=fig_info['output'],
            title=fig_info['title'],
            panel_labels=fig_info['labels'],
            figsize=fig_info['figsize']
        )
    
    # Handle single panel figures
    print("\nProcessing single panel figures...")
    for fig_info in single_figures:
        if os.path.exists(fig_info['file']):
            # Create a figure with title for consistency
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            img = mpimg.imread(fig_info['file'])
            ax.imshow(img)
            ax.axis('off')
            fig.suptitle(fig_info['title'], fontsize=18, fontweight='bold', y=0.95)
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            plt.savefig(fig_info['output'], dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Created single figure: {fig_info['output']}")
        else:
            print(f"Warning: Source file not found: {fig_info['file']}")
    
    print("\n" + "=" * 60)
    print("Composite figure creation complete!")
    print(f"Output directory: {output_dir}")
    print("\nCreated figures:")
    
    # List all created figures
    if os.path.exists(output_dir):
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith('.png'):
                filepath = os.path.join(output_dir, filename)
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"  - {filename} ({file_size:.1f} MB)")

def create_figure_summary():
    """Create a summary document of the composite figures."""
    
    summary_content = """# Composite Figures Summary

This document summarizes the composite figures created for publication.

## Main Figures

### Figure 1: Multi-modal Data Integration and Correlation Analysis
- **File**: `Figure_1_Multi_modal_Integration.png`
- **Panel A**: Correlation matrix showing relationships between 47 genomic features across four modalities
- **Panel B**: Feature distribution analysis showing cancer vs. control differences across all features
- **Description**: Demonstrates successful integration of multi-modal genomic data and reveals inter-feature relationships

### Figure 2: Principal Component Analysis of Integrated Genomic Features
- **File**: `Figure_2_PCA_Analysis.png`
- **Panel A**: PCA scatter plot showing separation between cancer and control samples
- **Panel B**: Variance explained by each principal component
- **Description**: Shows dimensionality reduction results and sample clustering patterns

### Figure 3: Model Performance Comparison
- **File**: `Figure_3_Model_Performance.png`
- **Panel A**: ROC curves for Random Forest and Logistic Regression models
- **Panel B**: Model comparison showing accuracy, precision, recall, and F1-score
- **Description**: Demonstrates model performance and comparative analysis

### Figure 4: Feature Importance Analysis
- **File**: `Figure_4_Feature_Importance.png`
- **Panel A**: SHAP summary plot showing feature importance rankings and value distributions
- **Panel B**: Top 10 most discriminative features with importance scores
- **Description**: Reveals which features drive model predictions and their relative importance

### Figure 5: Modality Contributions to Cancer Detection
- **File**: `Figure_5_Modality_Contributions.png`
- **Description**: Pie chart showing relative importance of each genomic modality
- **Key Finding**: Methylation (43%) and CNA (41%) dominate, with fragmentomics (12%) and chromatin (4%) providing complementary information

### Figure 6: Cancer Hallmarks Mapping
- **File**: `Figure_6_Cancer_Hallmarks.png`
- **Description**: Heatmap showing how genomic features map to established cancer hallmarks
- **Key Finding**: Features map to genomic instability, epigenetic dysregulation, altered metabolism, and immune evasion

## Publication Notes

1. **Resolution**: All figures created at 300 DPI for publication quality
2. **Format**: PNG format with white background
3. **Panel Labels**: Clear A, B labeling for multi-panel figures
4. **Titles**: Descriptive titles matching manuscript text
5. **Size**: Optimized for two-column journal layout

## File Specifications

- **Figure 1**: 20" x 10" (large correlation matrix requires space)
- **Figure 2**: 16" x 8" (standard two-panel layout)
- **Figure 3**: 16" x 8" (standard two-panel layout)
- **Figure 4**: 20" x 10" (SHAP plots require space)
- **Figure 5**: 12" x 10" (single panel)
- **Figure 6**: 12" x 10" (single panel)

## Manuscript Integration

These composite figures replace the individual figure references in the manuscript:
- Original individual files remain in `results_figures/`
- Composite figures are in `composite_figures/`
- Figure legends in manuscript remain the same
- Panel references (A, B) now correspond to actual panels in composite figures
"""
    
    with open('composite_figures/COMPOSITE_FIGURES_SUMMARY.md', 'w') as f:
        f.write(summary_content)
    
    print("Created composite figures summary: composite_figures/COMPOSITE_FIGURES_SUMMARY.md")

if __name__ == "__main__":
    main()
    create_figure_summary()
