# Cancer Genomics AI Classifier - Streamlit Web App

A sophisticated interactive web application that uses trained machine learning models to classify cancer from multi-modal genomic data. The app provides predictions with confidence scores and SHAP-based explanations for model interpretability.

## 🎯 Features

- **Model Selection**: Choose from multiple pre-trained models (Random Forest, Gradient Boosting, Deep Neural Network, Ensemble)
- **Multi-Modal Input**: Accepts genomic data across 6 modalities:
  - Methylation patterns (20 features)
  - Mutations (25 features) 
  - Copy number alterations (20 features)
  - Fragmentomics profiles (15 features)
  - Clinical information (10 features)
  - ICGC ARGO data (20 features)
- **Interactive Interface**: Three input methods - sample data generation, manual input, or CSV upload
- **Real-Time Predictions**: Instant cancer classification with confidence scores
- **SHAP Explainability**: Comprehensive model explanations including:
  - Feature importance rankings
  - Modality contribution analysis
  - Biological insights generation
- **Professional Visualizations**: Interactive plots using Plotly

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Cancer Alpha project models (located in `/Users/stillwell/projects/cancer-alpha/models/phase2_models/`)

### Installation

1. **Navigate to the web app directory:**
   ```bash
   cd /Users/stillwell/projects/cancer-alpha/src/phase4_systemization_and_tool_deployment/web_app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Test model loading (optional but recommended):**
   ```bash
   python3 test_models.py
   ```

4. **Launch the app:**
   ```bash
   streamlit run streamlit_app.py
   ```
   
   Or use the convenient startup script:
   ```bash
   ./start_app.sh
   ```

The app will open automatically in your web browser at `http://localhost:8501`

## 🔬 Using the App

### Model Selection
- Choose from available pre-trained models in the sidebar
- Models include Random Forest, Gradient Boosting, Deep Neural Network, and Ensemble approaches

### Input Methods

#### 1. Sample Data
- Generate representative cancer or control samples
- Pre-configured with realistic genomic patterns
- Perfect for testing and demonstration

#### 2. Manual Input
- Manually adjust all 110 genomic features
- Features are categorized by biological modality
- Tooltips provide descriptions for each feature

#### 3. CSV Upload
- Upload your own genomic data
- CSV should contain exactly 110 features matching the trained model structure
- First row will be used for prediction

### Understanding Results

#### Prediction Display
- **Classification**: Cancer detected or not detected
- **Cancer Probability**: Likelihood of cancer (0-100%)
- **Confidence Score**: Model confidence in the prediction

#### SHAP Explanations
- **Feature Importance**: Top contributing features with SHAP values
- **Modality Breakdown**: Contribution by genomic data type
- **Biological Insights**: Automated interpretation of results

### Biological Insights
The app provides context-aware insights based on:
- Which genomic modalities drive the prediction
- Confidence levels and their clinical significance
- Biological interpretation of feature patterns

## 📊 Data Structure

The app expects 110 features organized as:

| Modality | Features | Description |
|----------|----------|-------------|
| Methylation | 20 | DNA methylation patterns |
| Mutations | 25 | Genetic variant information |
| Copy Number Alterations | 20 | Chromosomal gains/losses |
| Fragmentomics | 15 | cfDNA fragment characteristics |
| Clinical | 10 | Patient clinical information |
| ICGC ARGO | 20 | International cancer genomics data |

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit with Plotly visualizations
- **Backend**: Scikit-learn models with SHAP explainability
- **Data Processing**: Standardized feature scaling
- **Deployment**: Local hosting with option for cloud deployment

### Model Support
- Tree-based models (Random Forest, Gradient Boosting)
- Neural networks (MLPClassifier)
- Ensemble methods
- SHAP compatibility for all model types

### Performance
- Real-time predictions (< 1 second)
- Interactive visualizations
- Responsive design for different screen sizes

## 🔍 Troubleshooting

### Common Issues

1. **Models not loading**
   - Verify model files exist in `/Users/stillwell/projects/cancer-alpha/models/phase2_models/`
   - Run `python3 test_models.py` to diagnose issues

2. **SHAP explanations not working**
   - Some models may have compatibility issues
   - The app will gracefully fall back to basic predictions

3. **Feature count mismatch**
   - Ensure uploaded CSV has exactly 110 columns
   - Use the sample data generator to see expected format

### Getting Help
- Check the model test script output for diagnostic information
- Review feature descriptions in the manual input mode
- Use sample data to verify app functionality

## 📈 Future Enhancements

- [ ] Support for batch predictions
- [ ] Additional visualization types
- [ ] Export functionality for reports
- [ ] Integration with clinical databases
- [ ] Advanced filtering and search capabilities
- [ ] Multi-user authentication and sessions

## 📝 Citation

If you use this application in research, please cite:

```
Cancer Alpha Research Team (2025). 
Cancer Genomics AI Classifier: An Interactive Web Application for 
Multi-Modal Cancer Classification with Explainable AI. 
Phase 4 Tool Deployment.
```

## 📄 License

This application is part of the Cancer Alpha research project. See project documentation for licensing details.

---

**Author**: Cancer Alpha Research Team  
**Date**: July 26, 2025  
**Version**: 1.0.0
