#!/bin/bash

# --- Configuration ---
PORT=8501

echo "🧬 Starting Cancer Genomics AI Demo..."
echo "======================================"

# Check for running processes on the Streamlit port
if lsof -i :$PORT -t >/dev/null; then
    echo "⚠️ Port $PORT is currently in use. Attempting to free it..."
    # Politely ask the process to terminate
    kill -9 $(lsof -ti :$PORT)
    sleep 2 # Give it a moment to shut down
    
    # Check again and force kill if necessary
    if lsof -i :$PORT -t >/dev/null; then
        echo "❌ Could not free port $PORT. Force killing process..."
        kill -9 $(lsof -ti :$PORT)
    fi
    echo "✅ Port $PORT is now free."
fi

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found in current directory"
    echo "Please run this script from the demo directory"
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Starting Streamlit app..."
echo "The demo will open in your web browser at http://localhost:$PORT"
echo "Press Ctrl+C to stop the demo"
echo ""

streamlit run streamlit_app.py --server.port $PORT
