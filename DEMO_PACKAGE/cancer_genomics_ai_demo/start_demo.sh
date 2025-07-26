#!/bin/bash

# Cancer Genomics AI Classifier - Standalone Demo
# ===============================================

echo "🧬 Cancer Genomics AI Classifier - Demo"
echo "======================================="
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "   Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -r requirements_streamlit.txt

# Download models if needed
if [ ! -d "models" ] || [ ! -f "models/random_forest_model.pkl" ]; then
    echo "🤖 Downloading AI models (this may take a moment)..."
    python3 -c "
import requests
import os
from pathlib import Path

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# This would download from a public repository in real deployment
print('Models will be generated on first run...')
"
fi

# Test the setup
echo "🧪 Testing setup..."
python3 test_models.py

if [ $? -eq 0 ]; then
    echo
    echo "🚀 Starting Cancer Genomics AI Classifier..."
    echo "🌐 Open your browser to: http://localhost:8501"
    echo "⏹️  Press Ctrl+C to stop"
    echo
    
    streamlit run streamlit_app.py --server.port 8501 --server.address localhost
else
    echo "❌ Setup test failed. Please check the error messages above."
    exit 1
fi
