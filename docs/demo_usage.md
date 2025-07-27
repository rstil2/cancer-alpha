# Demo Usage Guide

This guide explains how to use the Cancer Alpha interactive demo for cancer genomics classification.

## Getting Started

### Download and Setup
1. Start the demo server: `python3 serve_demo.py`
2. Download the demo package from http://localhost:8000/cancer_genomics_ai_demo.zip
3. Extract the ZIP file to your desired location
4. Run the demo:
   - **Windows**: Double-click `start_demo.bat`
   - **Mac/Linux**: Run `./start_demo.sh` in terminal

### Accessing the Interface
- Open your browser to http://localhost:8501
- The Streamlit interface will load with the Cancer Alpha demo

## Using the Demo

### Input Methods
The demo supports three ways to input genomic data:

1. **Synthetic Sample Data**: Pre-loaded examples for testing
2. **Manual Input**: Enter individual feature values
3. **CSV Upload**: Upload a file with genomic features

### Features Available
- **Cancer Classification**: 8 cancer types supported
- **SHAP Explainability**: Visual explanations for predictions
- **Confidence Scoring**: Prediction confidence levels
- **Multi-Modal Analysis**: Integration of different genomic data types

### Understanding Results
- **Prediction**: Shows the predicted cancer type
- **Confidence**: Indicates model certainty (High/Medium/Low)
- **SHAP Values**: Feature importance visualization
- **Biological Insights**: Interpretation of key features

## Demo Limitations

⚠️ **Important**: This is a demonstration version:
- Uses simplified models and synthetic data
- ~70% accuracy (vs 99.5% in production system)
- Limited to 8 basic cancer types
- Educational purposes only

## Getting Help

- Check the README files in the demo package
- Review the main project documentation
- Contact craig.stillwell@gmail.com for licensing questions

---

**Note**: This demo is for educational purposes only and should not be used for actual medical diagnosis.
