#!/bin/bash

# 🧬 Cancer Alpha - Clean Web App Startup Script
# This script ensures a clean startup by killing any existing processes on port 3000
# and then starts the React web application.

set -e  # Exit on any error

# Configuration
PORT=3000
WEBAPP_DIR="src/phase4_systemization_and_tool_deployment/web_app"

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

echo "🧬 Cancer Alpha - Clean Web App Startup"
echo "======================================"

# Step 1: Find and kill any processes using port 3000
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

# Step 2: Check if the web app directory exists
if [ ! -d "$WEBAPP_DIR" ]; then
    print_error "Web app directory $WEBAPP_DIR not found"
    print_status "Current directory: $(pwd)"
    print_status "Make sure you're in the cancer-alpha root directory"
    exit 1
fi

# Step 3: Navigate to web app directory
print_status "Navigating to web app directory..."
cd "$WEBAPP_DIR"

# Step 4: Check if package.json exists
if [ ! -f "package.json" ]; then
    print_error "package.json not found in $WEBAPP_DIR"
    print_status "This doesn't appear to be a valid React project directory"
    exit 1
fi

# Step 5: Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    print_status "Installing dependencies..."
    npm install || {
        print_error "Failed to install dependencies"
        exit 1
    }
    print_success "Dependencies installed"
else
    print_success "Dependencies already installed"
fi

# Step 6: Start the React app
print_status "Starting Cancer Alpha React Web App..."
print_status "Web app will be available at: http://localhost:$PORT"
print_status "API should be running at: http://localhost:8001"
echo "======================================"
echo ""

# Start the React development server
npm start
