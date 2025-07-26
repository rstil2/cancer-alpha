#!/bin/bash

# 🧬 Cancer Alpha - Clean API Startup Script
# This script ensures a clean startup by killing any existing processes on port 8001
# and then starts the real Cancer Alpha API with trained models.

set -e  # Exit on any error

# Configuration
PORT=8001
API_SCRIPT="real_cancer_alpha_api.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}🧬${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

echo "🧬 Cancer Alpha - Clean API Startup"
echo "===================================="

# Step 1: Find and kill any processes using port 8001
print_status "Checking for existing processes on port $PORT..."

PID=$(lsof -ti:$PORT 2>/dev/null || echo "")

if [ -n "$PID" ]; then
    print_warning "Found process $PID using port $PORT"
    print_status "Killing process $PID..."
    kill -9 $PID 2>/dev/null || {
        print_error "Failed to kill process $PID"
        print_status "You may need to run: sudo kill -9 $PID"
        exit 1
    }
    
    # Wait a moment for the process to be fully terminated
    sleep 2
    
    # Verify the process is gone
    NEW_PID=$(lsof -ti:$PORT 2>/dev/null || echo "")
    if [ -n "$NEW_PID" ]; then
        print_error "Process still running on port $PORT"
        exit 1
    else
        print_success "Port $PORT is now available"
    fi
else
    print_success "Port $PORT is already available"
fi

# Step 2: Check if the API script exists
if [ ! -f "$API_SCRIPT" ]; then
    print_error "API script $API_SCRIPT not found in current directory"
    print_status "Current directory: $(pwd)"
    print_status "Make sure you're in the cancer-alpha root directory"
    exit 1
fi

# Step 3: Quick dependency check
print_status "Checking Python dependencies..."
python3 -c "import fastapi, uvicorn, numpy, sklearn" 2>/dev/null || {
    print_error "Missing required Python packages"
    print_status "Install with: pip3 install fastapi uvicorn numpy scikit-learn"
    exit 1
}
print_success "Dependencies check passed"

# Step 4: Start the API
print_status "Starting Cancer Alpha API with real trained models..."
print_status "API will be available at: http://localhost:$PORT"
print_status "API documentation at: http://localhost:$PORT/docs"
echo "===================================="
echo ""

# Start the API
python3 "$API_SCRIPT"
