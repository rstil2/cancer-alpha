# Cancer Genomics AI Demo

A simplified demonstration of cancer classification using machine learning on genomic data.

## Quick Start

1. **Navigate to the demo directory:**
   ```bash
   cd cancer_genomics_ai_demo
   ```

2. **Run the demo:**
   ```bash
   ./start_demo.sh
   ```

3. **Open your browser** to http://localhost:8501

## What This Demo Does

- Classifies cancer samples into 8 different cancer types (BRCA, LUAD, COAD, STAD, BLCA, LIHC, CESC, KIRP)
- Uses a simple logistic regression model trained on synthetic genomic features
- Provides prediction confidence scores
- Shows probability distribution across all cancer types
- Generates models automatically on first run

## Features

- **Self-contained**: No external model files needed
- **Automatic setup**: Models are generated on first run
- **Interactive**: Adjust feature values or generate random samples
- **Simple interface**: Easy-to-use web interface
- **Lightweight**: Minimal dependencies

## Dependencies

- streamlit
- pandas
- numpy 
- scikit-learn

## Troubleshooting

If you encounter any issues:

1. Make sure you have Python 3.7+ installed
2. Try running: `pip install -r requirements.txt` manually
3. Then run: `streamlit run streamlit_app.py`

## Note

This is a simplified demonstration version. The model generates synthetic training data and is meant for educational purposes only.
