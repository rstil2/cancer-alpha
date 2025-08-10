#!/bin/bash

# Cancer Genomics AI Classifier - Production Demo
# ==============================================

echo "🧬 Cancer Alpha - Production Demo (95.0% Accuracy)"
echo "================================================" 
echo ""

echo "🏆 PRODUCTION DEMO PACKAGE:"
echo "Features our breakthrough LightGBM + SMOTE model:"
echo "- Production LightGBM + SMOTE (95.0% accuracy)"
echo "- Multi-modal genomic data (110 features)"
echo "- Full SHAP explanations"
echo "- 8 cancer types classification"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "   Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check for required model files
echo "🔍 Checking for production models..."
if [ ! -f "models/lightgbm_smote_production.pkl" ]; then
    echo "❌ Production model not found: models/lightgbm_smote_production.pkl"
    echo "   Please ensure you have the complete demo package."
    exit 1
fi

if [ ! -f "models/standard_scaler.pkl" ]; then
    echo "❌ Scaler not found: models/standard_scaler.pkl"
    echo "   Please ensure you have the complete demo package."
    exit 1
fi

echo "✅ Production models found"

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -r requirements_streamlit.txt

echo ""
# Clear any existing processes on port 8501
echo "🔄 Clearing port 8501..."
PORT_PID=$(lsof -ti :8501)
if [ ! -z "$PORT_PID" ]; then
    echo "   Killing existing process on port 8501 (PID: $PORT_PID)"
    kill -9 $PORT_PID 2>/dev/null || true
    sleep 2
fi
echo "✅ Port 8501 cleared"

echo "🚀 Starting Cancer Genomics AI Classifier..."
echo "🌐 Open your browser to: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop"
echo "⚠️  Note: This is demonstration software only - not for medical diagnosis"
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address localhost
