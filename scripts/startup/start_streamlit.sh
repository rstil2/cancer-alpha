#!/bin/bash

# Oncura - Streamlit Demo Launcher
# This is a demonstration version using simplified models and synthetic data
# For the full production system, contact craig.stillwell@gmail.com for licensing

echo "🧬 Oncura - Streamlit Demo Interface"
echo "=========================================="
echo ""
echo "⚠️  DEMO VERSION NOTICE:"
echo "This is a demonstration using simplified models and synthetic data."
echo "The full production system achieves 99.5% accuracy with real genomic data."
echo ""

# Check if we're in the right directory
if [ ! -d "DEMO_PACKAGE/cancer_genomics_ai_demo" ]; then
    echo "❌ Error: DEMO_PACKAGE not found."
    echo "Please make sure you're running this script from the cancer-alpha root directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "🚀 Starting Oncura Streamlit Demo..."
echo "📦 Using demo package with synthetic data"
echo ""

# Navigate to demo directory and start
cd DEMO_PACKAGE/cancer_genomics_ai_demo

# Check if requirements are installed
if [ ! -f "requirements_installed.flag" ]; then
    echo "📦 Installing demo requirements..."
    pip install -r requirements_streamlit.txt
    touch requirements_installed.flag
fi

echo "🌐 Starting Streamlit interface..."
echo "📍 Open your browser to: http://localhost:8501"
echo "🔍 Features: Random Forest model, SHAP explainability, synthetic data"
echo ""
echo "Press Ctrl+C to stop the demo"
echo ""

# Start the Streamlit app
python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
