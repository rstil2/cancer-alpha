#!/bin/bash

# Cancer Genomics AI Classifier - Startup Script
# ==============================================

echo "🧬 Cancer Genomics AI Classifier"
echo "=================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:/Users/stillwell/projects/cancer-alpha/src"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if we need to install requirements
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "📦 Installing required packages..."
    pip3 install -r requirements_streamlit.txt
fi

# Test model loading
echo "🔍 Testing model loading..."
python3 test_models.py

if [ $? -ne 0 ]; then
    echo "❌ Model testing failed. Please check model files."
    exit 1
fi

# Start Streamlit app
echo ""
echo "🚀 Starting Streamlit app..."
echo "📱 The app will open in your default web browser"
echo "🌐 URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address localhost
