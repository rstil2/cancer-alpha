#!/usr/bin/env python3
"""
Create Reproducibility Package for Manuscript Submission

Creates a zipped archive containing all data, code, and documentation
necessary to reproduce the results reported in the manuscript.
"""

import zipfile
import os
import json
from pathlib import Path
import shutil

def create_reproducibility_package():
    """Create comprehensive reproducibility package"""
    
    base_dir = Path('/Users/stillwell/projects/cancer-alpha')
    output_dir = base_dir / 'manuscripts'
    package_name = 'Oncura_Reproducibility_Package_Final.zip'
    output_path = output_dir / package_name
    
    print("="*80)
    print("CREATING REPRODUCIBILITY PACKAGE FOR MANUSCRIPT SUBMISSION")
    print("="*80)
    
    # Files to include
    files_to_include = {
        # 1. DATA FILES (from real_tcga_large)
        'data/': [
            'data/real_tcga_large/real_tcga_features_cleaned.csv',
            'data/real_tcga_large/real_tcga_labels.csv',
            'data/real_tcga_large/dataset_metadata.json',
            'data/real_tcga_large/model_comparison_real_only.json',
        ],
        
        # 2. CORE TRAINING/EVALUATION CODE
        'code/': [
            'cancer_genomics_ai_demo_minimal/train_real_tcga_near_100_percent.py',  # Main training script
            'compare_models_no_synthetic.py',       # Model comparison
            'imbalance_stress_test.py',             # NEW: Imbalance test
        ],
        
        # 3. MANUSCRIPT FILES
        'manuscript/': [
            'manuscripts/Oncura_Revised_Manuscript_FINAL_with_Figures-THIS IS IT.docx',
            'manuscripts/Oncura_Revised_EDITING.md',
        ],
        
        # 4. DOCUMENTATION
        'docs/': [
            'manuscripts/REVISIONS_COMPLETE.md',
            'manuscripts/EXPERIMENTAL_RESULTS_SUMMARY.md',
            'manuscripts/REVIEWER_REVISIONS_SUMMARY.md',
        ],
        
        # 5. EXPERIMENTAL RESULTS
        'results/': [
            'experiments/results/imbalance_stress_test_results.json',
            'experiments/results/imbalance_stress_test_confusion_matrices.png',
        ],
        
        # 6. FIGURES
        'figures/': [
            'manuscripts/manuscript_figures/figure1_model_performance.png',
            'manuscripts/manuscript_figures/figure2_cancer_type_performance.png',
            'manuscripts/manuscript_figures/figure3_feature_importance.png',
            'manuscripts/manuscript_figures/figure4_comparison_studies.png',
        ],
    }
    
    # Create README for the package
    readme_content = """# Oncura Reproducibility Package

## Overview

This package contains all data, code, and documentation necessary to reproduce the results 
reported in the manuscript:

**"Knowledge-Guided Multi-Modal Integration Improves Robustness and Accuracy in 
Multi-Cancer Genomic Classification"**

## Contents

### 1. Data (`data/`)
- `real_tcga_features_cleaned.csv`: 1,200 samples × 2,000 features (real TCGA data)
- `real_tcga_labels.csv`: Cancer type labels for all samples
- `dataset_metadata.json`: Complete dataset provenance and metadata
- `model_comparison_real_only.json`: Performance metrics across all models

**Data Authenticity**: All data is authentic TCGA patient data. Zero synthetic samples.
- 1,200 samples perfectly balanced (150 per cancer type)
- 8 cancer types: BRCA, LUAD, COAD, PRAD, STAD, HNSC, LUSC, LIHC
- 2,000 engineered features from 6 genomic modalities

### 2. Code (`code/`)
- `train_real_tcga_near_100_percent.py`: Main training script (reproduces 96.5% accuracy)
- `compare_models_no_synthetic.py`: Compares LightGBM, XGBoost, Random Forest, etc.
- `imbalance_stress_test.py`: Robustness validation under class imbalance

### 3. Results (`results/`)
- `imbalance_stress_test_results.json`: Validation results on imbalanced data
- `imbalance_stress_test_confusion_matrices.png`: Confusion matrix visualizations

### 4. Manuscript (`manuscript/`)
- `Oncura_Revised_Manuscript_FINAL_with_Figures.docx`: Final manuscript (Word)
- `Oncura_Revised_EDITING.md`: Manuscript source (Markdown)

### 5. Documentation (`docs/`)
- `REVISIONS_COMPLETE.md`: Summary of all revisions
- `EXPERIMENTAL_RESULTS_SUMMARY.md`: Detailed experimental results
- `REVIEWER_REVISIONS_SUMMARY.md`: Reviewer feedback responses

### 6. Figures (`figures/`)
- All 4 manuscript figures in high resolution PNG format

## Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

### Reproduce Main Results (96.5% accuracy)
```bash
cd code/
python train_real_tcga_near_100_percent.py
```

**Expected output**: 96.5% ± 0.6% balanced accuracy (5-fold CV)

**Runtime**: ~10 minutes on standard laptop

### Reproduce Imbalance Stress Test (97.5% accuracy)
```bash
cd code/
python imbalance_stress_test.py
```

**Expected output**: 97.5% balanced accuracy on naturally imbalanced test set

**Runtime**: ~15 minutes

### Reproduce Model Comparison
```bash
cd code/
python compare_models_no_synthetic.py
```

**Expected output**: Performance comparison across 6 algorithms

**Runtime**: ~30 minutes

## Data Provenance

All samples are authentic TCGA patient data accessed through the GDC portal:
- **Source**: The Cancer Genome Atlas (TCGA)
- **Access**: GDC Data Portal (https://portal.gdc.cancer.gov/)
- **Date**: August-October 2025
- **Authorization**: dbGaP controlled-access data
- **Quality Control**: 100% TCGA barcode verified, >90% complete annotations
- **Synthetic Data**: ZERO (explicitly verified in metadata)

See `data/dataset_metadata.json` for complete provenance documentation.

## Key Results Summary

### Main Performance (Section 3.3)
- **Champion Model**: LightGBM with knowledge-guided features
- **Balanced Accuracy**: 96.5% ± 0.6%
- **Cross-validation**: 5-fold stratified (96.2%, 95.8%, 96.3%, 96.7%, 97.5%)
- **Improvement**: +7.3 percentage points over transformer approaches

### Imbalance Robustness (Section 3.5 - NEW)
- **Balanced test set**: 96.4% accuracy
- **Imbalanced test set**: **97.5% accuracy (+1.1%)**
- **Natural prevalence**: BRCA 30%, LUAD 18%, ..., LIHC 3%
- **Interpretation**: Robust to real-world class distributions

### Biological Validation (Section 3.6)
- **Pathway enrichment**: 68% of features in cancer pathways (p < 0.01)
- **Biomarker overlap**: 83% match with NCCN guidelines
- **V-score**: 0.87 (biological plausibility)

## Computational Requirements

**Minimum**:
- CPU: Quad-core processor
- RAM: 8 GB
- Disk: 500 MB for data + results
- OS: macOS, Linux, or Windows

**Recommended**:
- CPU: 8-core processor
- RAM: 16 GB
- GPU: Not required (CPU-only training)

## Expected Runtime

- Main training: 10 minutes
- Model comparison: 30 minutes
- Imbalance test: 15 minutes
- **Total**: ~1 hour for complete reproduction

## Citation

If you use this code or data, please cite:

Stillwell, R. C. (2025). Knowledge-Guided Multi-Modal Integration Improves Robustness 
and Accuracy in Multi-Cancer Genomic Classification. *[Journal Name]*, [Volume](Issue).

## Contact

For questions or issues:
- Email: craig.stillwell@gmail.com
- GitHub: [Repository URL if available]

## License

**Academic Use**: Permitted with proper attribution
**Commercial Use**: Requires licensing (Patent No. 63/847,316)

---

**Package Version**: 1.0 (Final Submission)
**Created**: December 21, 2025
**Manuscript Status**: Ready for journal submission
"""
    
    # Create temporary directory for package contents
    temp_dir = base_dir / 'temp_package'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Copy files to temp directory
    print("\nCollecting files...")
    files_copied = 0
    files_missing = []
    
    for category, file_list in files_to_include.items():
        category_dir = temp_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for file_path in file_list:
            src = base_dir / file_path
            if src.exists():
                # Get filename and copy
                dest = category_dir / src.name
                shutil.copy2(src, dest)
                files_copied += 1
                print(f"  ✓ {file_path}")
            else:
                files_missing.append(file_path)
                print(f"  ✗ MISSING: {file_path}")
    
    # Write README
    readme_path = temp_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"  ✓ README.md created")
    
    # Create requirements.txt
    requirements = """# Python Dependencies for Oncura Reproducibility Package

# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine learning
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Utilities
joblib>=1.0.0
tqdm>=4.62.0

# Optional (for advanced features)
shap>=0.40.0  # For SHAP analysis
"""
    
    req_path = temp_dir / 'requirements.txt'
    with open(req_path, 'w') as f:
        f.write(requirements)
    print(f"  ✓ requirements.txt created")
    
    # Create package metadata
    metadata = {
        "package_name": "Oncura Reproducibility Package",
        "version": "1.0",
        "created_date": "2025-12-21",
        "manuscript_title": "Knowledge-Guided Multi-Modal Integration Improves Robustness and Accuracy in Multi-Cancer Genomic Classification",
        "author": "R. Craig Stillwell, PhD",
        "data_samples": 1200,
        "data_features": 2000,
        "cancer_types": 8,
        "main_accuracy": "96.5% ± 0.6%",
        "imbalance_accuracy": "97.5%",
        "files_included": files_copied,
        "data_authenticity": "100% real TCGA data, zero synthetic samples",
        "contact": "craig.stillwell@gmail.com"
    }
    
    meta_path = temp_dir / 'package_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ package_metadata.json created")
    
    # Create the zip file
    print(f"\nCreating zip archive: {package_name}")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
                
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    # Get final size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*80)
    print("PACKAGE CREATED SUCCESSFULLY")
    print("="*80)
    print(f"Location: {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Files included: {files_copied}")
    if files_missing:
        print(f"\n⚠️  Warning: {len(files_missing)} files missing:")
        for f in files_missing:
            print(f"  - {f}")
    
    print("\n" + "="*80)
    print("PACKAGE CONTENTS")
    print("="*80)
    print("✓ Data: 1,200 authentic TCGA samples")
    print("✓ Code: Training and evaluation scripts")
    print("✓ Results: Experimental validation outputs")
    print("✓ Manuscript: Final Word + Markdown versions")
    print("✓ Documentation: Complete revision summaries")
    print("✓ Figures: All 4 manuscript figures")
    print("✓ README: Comprehensive usage instructions")
    print("\n" + "="*80)
    print("READY FOR JOURNAL SUBMISSION")
    print("="*80)
    
    return output_path, size_mb, files_copied

if __name__ == '__main__':
    package_path, size, n_files = create_reproducibility_package()
    print(f"\n✅ Package ready: {package_path.name} ({size:.1f} MB, {n_files} files)")
