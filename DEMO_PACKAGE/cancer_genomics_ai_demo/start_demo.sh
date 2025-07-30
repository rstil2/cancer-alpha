#!/bin/bash

# Cancer Genomics AI Classifier - Minimal Demo
# ============================================

echo "🧬 Cancer Alpha - Minimal Streamlit Demo"
echo "======================================="
echo ""

echo "📦 MINIMAL DEMO PACKAGE:"
echo "This demo generates models and data on first run:"
echo "- Synthetic genomic data (270 features)"
echo "- Random Forest and Logistic Regression models"
echo ""

# Check Python installation
if ! command -v python3 > /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "   Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -r requirements_streamlit.txt

# Generate data and models if they don't exist
if [ ! -f "models/scaler.pkl" ] || [ ! -f "data/tcga_processed_data.npz" ]; then
    echo "🔬 Generating demo data and models..."
    python3 generate_demo_data.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate demo data. Please check the error above."
        exit 1
    fi
else
    echo "✅ Demo data and models already exist"
fi

echo ""
echo "🚀 Starting Streamlit demo..."
echo "   Demo will open in your browser at http://localhost:8501"
echo "   Press Ctrl+C to stop the demo"
echo ""

# Start Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.headless false
