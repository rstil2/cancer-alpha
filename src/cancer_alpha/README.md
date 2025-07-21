# Source Code Documentation

This directory contains all the Python scripts used in the cancer genomics analysis pipeline.

## Main Analysis Scripts

### 1. `comprehensive_cancer_analysis.py`
**Purpose**: Main analysis script that performs the complete multi-modal cancer detection pipeline.

**Key Functions**:
- Multi-modal data integration
- Feature engineering across 4 genomic modalities
- Machine learning model training (Random Forest, Logistic Regression)
- SHAP explainability analysis
- Visualization generation

**Usage**: 
```bash
python comprehensive_cancer_analysis.py
```

**Outputs**:
- Model performance metrics
- Feature importance rankings
- SHAP analysis results
- Visualization plots

### 2. `create_composite_figures.py`
**Purpose**: Creates publication-ready composite figures from individual analysis plots.

**Key Functions**:
- Combines multiple plots into publication-ready figures
- Adds A/B panel labeling
- Applies consistent formatting

**Usage**: 
```bash
python create_composite_figures.py
```

## Data Acquisition Scripts

### 3. `download_all_three_sources.py`
**Purpose**: Downloads and integrates data from TCGA, GEO, and ENCODE databases.

**Key Functions**:
- TCGA methylation and CNA data retrieval
- GEO fragmentomics data processing
- ENCODE chromatin accessibility data integration

### 4. `data_acquisition.py` & `data_acquisition_v2.py`
**Purpose**: Earlier versions of data acquisition scripts.

**Key Functions**:
- Database API interactions
- Data format standardization
- Quality control checks

## Analysis Pipeline Scripts

### 5. `analyze_actual_genomic_data.py`
**Purpose**: Analyzes real genomic data with comprehensive statistical methods.

**Key Functions**:
- Statistical analysis of genomic features
- Effect size calculations
- Biological interpretation

### 6. `analyze_real_integrated_data.py`
**Purpose**: Performs integrated analysis across multiple data modalities.

**Key Functions**:
- Cross-modal correlation analysis
- Integrated feature selection
- Multi-modal visualization

### 7. `real_data_analysis.py`
**Purpose**: Comprehensive analysis of real-world genomic datasets.

**Key Functions**:
- Real data processing pipelines
- Validation against synthetic controls
- Performance benchmarking

## Model Development Scripts

### 8. `multimodal_model.py`
**Purpose**: Develops and trains multi-modal machine learning models.

**Key Functions**:
- Model architecture design
- Cross-validation implementation
- Hyperparameter optimization

### 9. `cancer_genomics_model.py`
**Purpose**: Specialized models for cancer genomics applications.

**Key Functions**:
- Cancer-specific feature engineering
- Biological constraint incorporation
- Clinical interpretation frameworks

### 10. `simplified_model.py`
**Purpose**: Simplified version of the analysis for demonstration purposes.

**Key Functions**:
- Streamlined analysis pipeline
- Essential feature extraction
- Basic visualization

## Utility Scripts

### 11. `preprocess_data.py`
**Purpose**: Data preprocessing and cleaning utilities.

**Key Functions**:
- Data quality control
- Missing value handling
- Feature scaling and normalization

### 12. `feature_extraction.py`
**Purpose**: Advanced feature engineering methods.

**Key Functions**:
- Multi-modal feature creation
- Biological feature mapping
- Dimensionality reduction

### 13. `final_analysis.py`
**Purpose**: Final analysis pipeline combining all components.

**Key Functions**:
- Complete analysis workflow
- Result compilation
- Report generation

## Data Download Scripts

### 14. `download_actual_genomic_data.py`
**Purpose**: Downloads actual genomic data from public repositories.

### 15. `download_real_multimodal_data.py`
**Purpose**: Downloads multi-modal datasets for integrated analysis.

## Script Dependencies

Most scripts depend on:
- pandas, numpy for data manipulation
- scikit-learn for machine learning
- matplotlib, seaborn for visualization
- shap for explainability analysis
- requests for API calls

## Execution Order

For full pipeline execution:
1. `download_all_three_sources.py` - Download data
2. `preprocess_data.py` - Clean and prepare data
3. `comprehensive_cancer_analysis.py` - Main analysis
4. `create_composite_figures.py` - Generate publication figures

## Output Files

Scripts generate various output files:
- CSV files with analysis results
- PNG files with visualizations
- JSON files with model parameters
- Pickle files with trained models

## Error Handling

All scripts include error handling for:
- Missing data files
- API connection failures
- Memory limitations
- Invalid parameter settings

## Performance Considerations

- Scripts are optimized for datasets up to 10,000 samples
- Memory usage is monitored for large datasets
- Parallel processing is implemented where applicable
- Progress tracking is included for long-running analyses

## Troubleshooting

Common issues and solutions:
1. **Import errors**: Check `requirements.txt` and install missing packages
2. **Data not found**: Verify data files are in correct directories
3. **Memory errors**: Reduce dataset size or increase system memory
4. **API failures**: Check internet connection and API key validity

## Contributing

When adding new scripts:
1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Add error handling
4. Update this README with new script information
