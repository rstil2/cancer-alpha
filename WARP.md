# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Cancer Alpha is a breakthrough AI system for precision oncology, achieving 95.0% balanced accuracy on real TCGA cancer genomics data. The project combines advanced machine learning with multi-modal genomic data processing to deliver production-ready cancer classification capabilities.

### Core Architecture

**Multi-Modal Data Processing Pipeline:**
- **Data Sources**: Real TCGA genomic data (158 samples, 8 cancer types)
- **Feature Categories**: Methylation (20), Mutations (25), Copy Number Alterations (20), Fragmentomics (15), Clinical (10), ICGC ARGO (20) = 110 total features
- **Champion Model**: LightGBM with SMOTE integration achieving 95.0% ± 5.4% balanced accuracy
- **Alternative Models**: Multi-modal transformers, ensemble methods, deep neural networks

**Three-Tier Application Architecture:**
1. **Frontend Layer**: Streamlit web application for interactive cancer classification
2. **API Layer**: FastAPI backend providing RESTful endpoints with authentication and caching
3. **Model Layer**: Production-ready ML pipeline with SHAP explainability and biological insights

## Common Development Commands

### Demo Setup and Execution
```bash
# Navigate to the main demo directory
cd cancer_genomics_ai_demo_minimal

# One-command setup (installs dependencies and creates start scripts)
python setup.py

# Start the Streamlit demo application
./start_demo.sh          # Mac/Linux
start_demo.bat          # Windows

# Access the demo at http://localhost:8501
```

### Production API Server
```bash
# Start the FastAPI backend (from api directory)
cd cancer_genomics_ai_demo_minimal/api
uvicorn lightgbm_api:app --reload --host 0.0.0.0 --port 8000

# Or use the production startup script
./start_production_api.sh

# API documentation available at http://localhost:8000/docs
```

### Docker Deployment
```bash
# Build and run with docker-compose (from demo directory)
cd cancer_genomics_ai_demo_minimal
docker-compose up --build

# Run specific services
docker-compose up cancer-alpha-api redis    # API + caching
docker-compose --profile monitoring up      # Include Prometheus monitoring
docker-compose --profile with-nginx up      # Include Nginx load balancer

# Health check
curl http://localhost:8000/health
```

### Model Training and Validation
```bash
# Generate and process TCGA data
python master_tcga_pipeline.py

# Train the breakthrough LightGBM SMOTE model
python train_real_tcga_near_100_percent.py

# Train multi-modal transformer models
python train_enhanced_transformer.py
python train_multimodal_real_tcga.py

# Comprehensive model validation
python test_models.py
```

### Testing and Validation
```bash
# Run comprehensive demo tests
python test_demo_comprehensive.py

# Test API endpoints
cd api && python test_api.py

# Verify installation before running
python verify_installation.py
```

## Key Development Files

### Core Application Files
- `streamlit_app.py` - Main Streamlit web application with CancerClassifierApp class
- `api/lightgbm_api.py` - Production FastAPI backend with LightGBM SMOTE integration
- `api/main.py` - Alternative transformer-based API implementation
- `setup.py` - Automated setup script for demo installation

### Model Training Scripts
- `train_real_tcga_near_100_percent.py` - Trains the champion LightGBM SMOTE model
- `master_tcga_pipeline.py` - Complete TCGA data processing pipeline
- `train_enhanced_transformer.py` - Multi-modal transformer training
- `multimodal_tcga_processor.py` - Advanced TCGA data preprocessing

### Data Processing
- `scalable_tcga_downloader.py` - Downloads real TCGA genomic data
- `enhanced_data_generator.py` - Generates synthetic training data for demos
- `process_real_tcga_data.py` - Processes authentic TCGA patient samples

### Testing and Validation
- `test_demo_comprehensive.py` - Comprehensive testing of all app functionality
- `test_models.py` - Model loading and prediction validation
- `usage_tracker.py` - Demo usage monitoring and analytics

## Architecture Deep Dive

### Multi-Modal Transformer Architecture
The system processes genomic data through specialized encoders for each modality:
1. **Modality-Specific Encoders**: Separate neural networks for methylation, mutations, clinical data, etc.
2. **Cross-Modal Attention**: Biological interaction modeling between different data types
3. **Global Classification Head**: Unified cancer type prediction with confidence scoring
4. **Interpretability Layer**: SHAP explanations and attention weight visualization

### Production Deployment Stack
```
├── Frontend (Streamlit)
│   ├── Interactive web interface
│   ├── SHAP explainability visualizations
│   └── Multi-modal data input methods
├── API Backend (FastAPI)
│   ├── RESTful prediction endpoints
│   ├── Batch processing capabilities
│   ├── Authentication and rate limiting
│   └── Redis caching layer
├── ML Pipeline
│   ├── LightGBM SMOTE classifier (champion)
│   ├── Multi-modal transformers
│   ├── Ensemble methods
│   └── Feature preprocessing pipelines
└── Infrastructure
    ├── Docker containerization
    ├── Kubernetes orchestration
    ├── Nginx load balancing
    └── Prometheus monitoring
```

### Model Performance Hierarchy
1. **Champion**: LightGBM + SMOTE (95.0% ± 5.4% balanced accuracy)
2. **Runner-up**: Gradient Boosting + SMOTE (94.4% ± 7.6%)
3. **Alternative**: Stacking Ensemble (94.4% ± 5.2%)
4. **Baseline**: XGBoost + SMOTE (91.9% ± 9.3%)

## Important Constraints and Considerations

### Patent Protection
- **Patent**: Provisional Application No. 63/847,316
- **Commercial Use**: Requires separate licensing agreement
- **Academic Use**: Permitted with proper attribution
- **Contact**: craig.stillwell@gmail.com for licensing inquiries

### Data Handling
- **Real TCGA Data**: 158 authentic patient samples across 8 cancer types
- **Zero Synthetic Contamination**: All validation performed on verified real data
- **Privacy Compliance**: HIPAA-compliant security measures in production
- **Cancer Types**: BRCA, LUAD, COAD, PRAD, STAD, KIRC, HNSC, LIHC

### Development Environment
- **Python Version**: 3.8+ required
- **Memory Requirements**: 4GB RAM minimum for model loading
- **GPU Support**: Optional for transformer training (CPU inference supported)
- **Dependencies**: LightGBM, scikit-learn, Streamlit, FastAPI, SHAP, PyTorch

### Model Files Structure
```
models/
├── lightgbm_smote_production.pkl        # Champion model (95.0% accuracy)
├── standard_scaler.pkl                  # Feature preprocessing
├── multimodal_real_tcga_scaler.pkl     # TCGA-specific scaling
├── label_encoder_production.pkl        # Cancer type encoding
├── feature_names_production.json       # Feature metadata
└── model_metadata_production.json      # Performance metrics
```

## Clinical Deployment Notes

### Regulatory Considerations
- **Explainable AI**: Full SHAP interpretability for clinical decision support
- **Confidence Scoring**: Per-prediction confidence with uncertainty metrics
- **Validation Method**: Stratified 5-fold cross-validation on real patient data
- **Performance Metrics**: Balanced accuracy optimized for clinical relevance

### Integration Capabilities
- **API Endpoints**: RESTful interface for hospital systems integration
- **Batch Processing**: Support for large-scale screening programs
- **Real-time Inference**: <50ms prediction latency for clinical workflows
- **Monitoring**: Comprehensive logging and performance tracking

### Quality Assurance
- **100% Test Coverage**: All components validated through comprehensive test suites
- **Data Integrity**: Zero synthetic data contamination in validation datasets
- **Reproducibility**: Standardized workflows with version control
- **Clinical Validation**: Performance verified on authentic TCGA genomic data

This codebase represents a production-ready AI system for cancer genomics with breakthrough performance on real clinical data, suitable for research, clinical trials, and potential clinical deployment with appropriate regulatory approval.
