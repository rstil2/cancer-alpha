#!/bin/bash

# Quick Fix Script - Deploy Demo Locally
# This creates a local server to serve the demo file

echo "🚀 Cancer Alpha Demo - Local Deployment Fix"
echo "=========================================="

# Check if demo file exists
if [ ! -f "cancer_genomics_ai_demo.zip" ]; then
    echo "❌ Demo file not found!"
    exit 1
fi

# Start a simple Python HTTP server
echo "📦 Starting local demo server..."
echo "Demo will be available at: http://localhost:8080/cancer_genomics_ai_demo.zip"
echo ""
echo "📝 Update your README download link to:"
echo "   http://localhost:8080/cancer_genomics_ai_demo.zip"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

# Start server on port 8080
python3 -m http.server 8080
