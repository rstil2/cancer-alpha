#!/bin/bash
# Oncura - LightGBM SMOTE API Production Startup
# ===================================================

echo "🚀 Starting Oncura LightGBM SMOTE API (Production Mode)..."

# Set production environment
export ENVIRONMENT=production
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=info

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing production dependencies..."
pip install --upgrade pip
pip install -r requirements_lightgbm.txt

# Verify LightGBM installation
echo "✅ Verifying LightGBM installation..."
python3 -c "import lightgbm; print(f'LightGBM version: {lightgbm.__version__}')"

# Check if models directory exists
MODELS_DIR="../../models"
if [ ! -d "$MODELS_DIR" ]; then
    echo "❌ Models directory not found at $MODELS_DIR"
    exit 1
fi

# Check for production model files
if [ ! -f "$MODELS_DIR/lightgbm_smote_production.pkl" ]; then
    echo "🔧 Production model not found, generating..."
    cd $MODELS_DIR
    python3 lightgbm_smote_production.py
    cd -
fi

echo "🔍 Production model files verified:"
ls -la $MODELS_DIR/*production*

# Start the API server
echo "🌟 Starting Oncura LightGBM SMOTE API..."
echo "   📡 Endpoint: http://localhost:$API_PORT"
echo "   📋 Documentation: http://localhost:$API_PORT/docs"
echo "   🔑 API Keys: cancer-alpha-prod-key-2025, demo-key-123"
echo ""
echo "🎯 Model: LightGBM with SMOTE - 95.0% target accuracy"
echo "🧬 Cancer Types: BRCA, LUAD, COAD, PRAD, STAD, KIRC, HNSC, LIHC"
echo ""

# Run the API
python3 -m uvicorn lightgbm_api:app \
    --host $API_HOST \
    --port $API_PORT \
    --log-level $LOG_LEVEL \
    --access-log \
    --workers 1
