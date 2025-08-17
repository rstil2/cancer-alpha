#!/bin/bash

# Oncura Demo Server Startup Script
# This script starts the demo server to serve the downloadable demo package

echo "🌟 Oncura Demo Server"
echo "=================================="

# Check if demo package exists
if [ ! -f "cancer_genomics_ai_demo.zip" ]; then
    echo "❌ Error: cancer_genomics_ai_demo.zip not found!"
    echo "Please ensure you're running this from the cancer-alpha directory."
    exit 1
fi

# Check if server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ Demo server is already running on port 8000"
    echo "🔗 Download link: http://localhost:8000/cancer_genomics_ai_demo.zip"
    echo "📊 File size: $(du -h cancer_genomics_ai_demo.zip | cut -f1)"
else
    echo "🚀 Starting demo server..."
    echo "📦 Serving: cancer_genomics_ai_demo.zip"
    echo "📊 File size: $(du -h cancer_genomics_ai_demo.zip | cut -f1)"
    echo "🌐 Server will be available at: http://localhost:8000/"
    echo "🔗 Download link: http://localhost:8000/cancer_genomics_ai_demo.zip"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "=================================="
    
    # Start the server
    python3 serve_demo.py
fi
