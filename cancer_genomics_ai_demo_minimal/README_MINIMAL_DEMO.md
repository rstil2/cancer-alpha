# Oncura - Minimal Demo Package

**⚡ Ultra-lightweight demo package that generates models on first run**

This minimal demo package (under 4MB) contains only the essential code and generates the necessary data and models automatically when you first run it.

## 🚀 Quick Start

1. **Unzip the demo package:**
   ```bash
   unzip cancer_genomics_ai_demo_minimal.zip
   cd cancer_genomics_ai_demo_minimal
   ```

2. **Run the demo:**
   ```bash
   chmod +x start_demo.sh
   ./start_demo.sh
   ```

3. **Open your browser to:** http://localhost:8501

## 📦 What's Included

- **Essential Code Only**: Streamlit app, model training scripts, requirements
- **Auto-Generated Data**: Creates 270-feature synthetic genomic data on first run
- **Auto-Generated Models**: Trains Random Forest and Logistic Regression models
- **Zero External Dependencies**: No large model files or data downloads required

## 🔬 Generated Content

On first run, the demo automatically creates:
- **Synthetic genomic data**: 1,000 samples with 270 features across 8 cancer types
- **Random Forest model**: Trained on the synthetic data
- **Logistic Regression model**: With proper scaling and preprocessing
- **Feature metadata**: Names and descriptions for all genomic features

## ⚙️ System Requirements

- **Python 3.8+**
- **Internet connection** (for installing Python packages)
- **~100MB free space** (after model generation)

## 🎯 Demo Features

- **Multi-class cancer classification** (8 cancer types)
- **Interactive feature input** (manual or CSV upload)
- **Real-time predictions** with confidence scores
- **SHAP explainability** visualizations
- **Sample data generation** for testing

## 📋 Cancer Types Supported

- BRCA (Breast Invasive Carcinoma)
- LUAD (Lung Adenocarcinoma)  
- COAD (Colon Adenocarcinoma)
- PRAD (Prostate Adenocarcinoma)
- STAD (Stomach Adenocarcinoma)
- KIRC (Kidney Renal Clear Cell Carcinoma)
- HNSC (Head and Neck Squamous Cell Carcinoma)
- LIHC (Liver Hepatocellular Carcinoma)

## ⚠️ Demo Limitations

- **Synthetic data only** - not for clinical use
- **Educational purposes** - demonstrates AI methodology
- **Limited accuracy** - real production models achieve 95%+ accuracy
- **No real patient data** - complies with privacy regulations

## 🛠️ Troubleshooting

If you encounter issues:

1. **Python not found**: Install Python 3.8+ from https://python.org
2. **Permission denied**: Run `chmod +x start_demo.sh` first
3. **Port 8501 busy**: Stop other Streamlit instances or use a different port
4. **Package installation fails**: Try `pip3 install --user -r requirements_streamlit.txt`

## 🔗 Full System

For the complete Oncura system with production models and real data validation:
- Visit: https://github.com/stillwell/cancer-alpha
- Accuracy: 95.33% on real clinical data
- Features: 270 multi-modal genomic features
- Architecture: Advanced multi-modal transformers

## 📄 License & Patents

This demo is provided under patent protection. See LICENSE file for details.

---

**🚀 Ready to start? Run `./start_demo.sh` and open http://localhost:8501**
