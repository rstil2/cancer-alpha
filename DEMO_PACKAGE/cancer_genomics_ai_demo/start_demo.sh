#!/bin/bash

echo "🧬 Starting Cancer Genomics AI Demo..."
echo "======================================"

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found in current directory"
    echo "Please run this script from the demo directory"
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Starting Streamlit app..."
echo "The demo will open in your web browser at http://localhost:8501"
echo "Press Ctrl+C to stop the demo"
echo ""

streamlit run streamlit_app.py
