# 🧬 Cancer Genomics AI Classifier - Demo Application

[![License: Patent Protected](https://img.shields.io/badge/License-Patent%20Protected-red.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io/)

## ⚠️ PATENT PROTECTED TECHNOLOGY ⚠️

**This is a LIMITED DEMONSTRATION VERSION of patent-protected technology.**

- **Patent**: Provisional Application No. 63/847,316
- **Title**: Systems and Methods for Cancer Classification Using Multi-Modal Transformer-Based Architectures
- **Patent Holder**: Dr. R. Craig Stillwell
- **Contact**: craig.stillwell@gmail.com

**Commercial use requires separate patent licensing.**

---

## 🚀 Overview

Experience state-of-the-art cancer genomics classification powered by advanced machine learning! This interactive web application demonstrates cutting-edge AI technology for multi-modal genomic data analysis.

### 🏆 Featured Model: Production LightGBM + SMOTE (95.0% Accuracy)

Our best-performing model combines:
- **LightGBM** gradient boosting for superior performance
- **SMOTE** (Synthetic Minority Oversampling) for advanced class balancing
- **Multi-modal integration** across 6 genomic data types
- **Real-time SHAP explanations** for model interpretability

## 🧪 Demo Capabilities

- **Multi-class Cancer Classification**: Classify between 8 cancer types (BRCA, LUAD, COAD, PRAD, STAD, KIRC, HNSC, LIHC)
- **Multi-modal Genomic Features**: 110 features across 6 data modalities:
  - 🧬 DNA Methylation patterns (20 features)
  - 🔄 Genetic mutations (25 features) 
  - 📊 Copy number alterations (20 features)
  - 🧪 Fragmentomics profiles (15 features)
  - 🏥 Clinical data (10 features)
  - 🌍 ICGC ARGO international data (20 features)
- **Real-time Predictions** with confidence scores
- **SHAP Explanations** for model interpretability
- **Interactive Visualizations** with biological insights

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Modern web browser

## 🛠️ Installation

### Option 1: Quick Start (Recommended)

1. **Download the demo package**
   ```bash
   git clone [YOUR_REPO_URL]
   cd cancer_genomics_ai_demo_minimal
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Run the demo**
   ```bash
   # On macOS/Linux:
   ./start_demo.sh
   
   # On Windows:
   start_demo.bat
   
   # Or directly:
   streamlit run streamlit_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Option 2: Using Docker

```bash
# Build and run with Docker
docker build -t cancer-genomics-demo .
docker run -p 8501:8501 cancer-genomics-demo
```

### Option 3: Using Docker Compose

```bash
# Run with docker-compose
docker-compose up
```

## 🎯 How to Use

### 1. **Select Input Method**
- **Sample Data**: Generate realistic cancer/control samples
- **Manual Input**: Adjust individual genomic features
- **Upload CSV**: Use your own data (110 features required)

### 2. **Generate Predictions**
- View predicted cancer type with confidence scores
- Explore probability distributions across all cancer types
- Examine SHAP feature importance explanations

### 3. **Interpret Results**
- **Feature Contributions**: See which genomic features drive predictions
- **Modality Analysis**: Understand contributions by data type
- **Biological Insights**: Get research-validated interpretations

## 📁 Project Structure

```
cancer_genomics_ai_demo_minimal/
├── streamlit_app.py              # Main demo application
├── models/                       # Pre-trained AI models
│   ├── lightgbm_smote_production.pkl
│   ├── standard_scaler.pkl
│   └── model_metadata.json
├── requirements_streamlit.txt    # Python dependencies
├── start_demo.sh                # Quick start script (Unix)
├── start_demo.bat              # Quick start script (Windows)
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── usage_tracker.py            # Demo usage analytics
└── README_GITHUB.md            # This file
```

## 🔬 Technical Specifications

### Model Architecture
- **Algorithm**: LightGBM with SMOTE preprocessing
- **Input Features**: 110 multi-modal genomic features
- **Output Classes**: 8 cancer types
- **Accuracy**: 95.0% on validation data
- **Training Data**: Real TCGA (The Cancer Genome Atlas) datasets

### Data Modalities
1. **Methylation Patterns**: DNA methylation levels across key genomic regions
2. **Genetic Mutations**: Variant information and mutation signatures  
3. **Copy Number Alterations**: Chromosomal gains and losses
4. **Fragmentomics**: cfDNA fragment size and distribution patterns
5. **Clinical Features**: Patient demographic and clinical variables
6. **ICGC ARGO**: International cancer genomics consortium data

## 🚨 Important Notes

### For Demonstration Only
- This software is for **demonstration purposes only**
- Should **NOT** be used for actual medical diagnosis or treatment decisions
- Results are for research and educational purposes

### System Requirements
- **Memory**: 4GB+ RAM recommended for optimal performance
- **Browser**: Modern browser with JavaScript enabled
- **Network**: Internet connection for initial setup

### Data Privacy
- No genomic data is stored or transmitted
- All processing happens locally on your machine
- Demo usage statistics are tracked anonymously

## 🆘 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure all dependencies are installed
pip install --upgrade -r requirements_streamlit.txt
```

**Port Already in Use**
```bash
# Use a different port
streamlit run streamlit_app.py --server.port 8502
```

**Model Loading Issues**
- Ensure `models/` directory contains all required files
- Check that `lightgbm_smote_production.pkl` exists and is not corrupted

**Performance Issues**
- Close other applications to free up memory
- Try using the Docker version for better resource isolation

### Getting Help

For technical issues or questions:
1. Check the troubleshooting section above
2. Review the demo logs in `demo_usage.log`
3. Contact: craig.stillwell@gmail.com

## 📊 Performance Metrics

- **Accuracy**: 95.0%
- **Multi-class Classification**: 8 cancer types
- **Feature Engineering**: 110 optimized genomic features
- **Model Interpretability**: SHAP-based explanations
- **Real-time Predictions**: < 1 second response time

## 🔬 Scientific Background

This demo showcases advanced machine learning techniques applied to cancer genomics:

- **Multi-modal Integration**: Combines diverse genomic data types for comprehensive analysis
- **Class Imbalance Handling**: SMOTE technique ensures robust performance across all cancer types  
- **Feature Selection**: Optimized 110-feature set from extensive genomic profiling
- **Model Interpretability**: SHAP explanations provide biological insights into predictions
- **Validation**: Trained and validated on real TCGA datasets

## 📄 License & Usage

**Patent-Protected Technology - Demonstration License**

This demo application is provided under a limited demonstration license:
- ✅ Educational and research use permitted
- ✅ Evaluation and testing allowed
- ❌ Commercial use prohibited without patent licensing
- ❌ Redistribution without permission prohibited

For commercial licensing inquiries, contact: craig.stillwell@gmail.com

## 🙏 Acknowledgments

- **TCGA Research Network** for providing the genomic datasets
- **ICGC ARGO** for international genomics data standards
- **Scientific Community** for advancing cancer genomics research

---

## 📞 Contact

**Dr. R. Craig Stillwell**
- Email: craig.stillwell@gmail.com
- Patent: Provisional Application No. 63/847,316

**For Commercial Licensing**: craig.stillwell@gmail.com

---

*Experience the future of precision cancer diagnosis through AI-powered genomic analysis.*
