#!/bin/bash

# Cancer Genomics AI Classifier API Startup Script
# =================================================

echo "🚀 Starting Cancer Genomics AI Classifier API..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
export API_ENV="production"
export API_HOST="0.0.0.0"
export API_PORT="8000"

# Create logs directory
mkdir -p logs

# Function to check if required files exist
check_requirements() {
    echo "📋 Checking requirements..."
    
    # Check for model files
    if [ ! -f "../models/optimized_multimodal_transformer.pth" ]; then
        echo "❌ Error: optimized_multimodal_transformer.pth not found!"
        echo "   Please ensure the model file is in the models directory."
        exit 1
    fi
    
    if [ ! -f "../models/scalers.pkl" ]; then
        echo "❌ Error: scalers.pkl not found!"
        echo "   Please ensure the scalers file is in the models directory."
        exit 1
    fi
    
    # Check for transformer module
    if [ ! -f "../models/multimodal_transformer.py" ]; then
        echo "❌ Error: multimodal_transformer.py not found!"
        echo "   Please ensure the transformer module is available."
        exit 1
    fi
    
    echo "✅ All required files found."
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing dependencies..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    pip install -r requirements.txt
    
    echo "✅ Dependencies installed."
}

# Function to check Redis connection (optional)
check_redis() {
    echo "🔍 Checking Redis connection..."
    
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping > /dev/null 2>&1; then
            echo "✅ Redis is running and accessible."
        else
            echo "⚠️  Redis is not running. Caching will be disabled."
        fi
    else
        echo "⚠️  Redis CLI not found. Install Redis for caching support."
    fi
}

# Function to start the API
start_api() {
    echo "🌐 Starting FastAPI server..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start server with production settings
    uvicorn main:app \
        --host $API_HOST \
        --port $API_PORT \
        --workers 4 \
        --loop uvloop \
        --log-level info \
        --access-log \
        --log-config logging.yaml 2>/dev/null || \
    uvicorn main:app \
        --host $API_HOST \
        --port $API_PORT \
        --workers 4 \
        --log-level info \
        --access-log
}

# Function for development mode
start_dev() {
    echo "🔧 Starting in development mode..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start server with reload
    uvicorn main:app \
        --host $API_HOST \
        --port $API_PORT \
        --reload \
        --log-level debug
}

# Parse command line arguments
case "$1" in
    "dev")
        echo "Development mode selected"
        check_requirements
        install_dependencies
        check_redis
        start_dev
        ;;
    "install")
        echo "Installing dependencies only"
        install_dependencies
        ;;
    "check")
        echo "Checking requirements only"
        check_requirements
        check_redis
        ;;
    *)
        echo "Production mode (default)"
        check_requirements
        install_dependencies
        check_redis
        start_api
        ;;
esac
