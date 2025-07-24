#!/bin/bash

# 🧬 Cancer Alpha API Startup Script
# This script handles port conflicts and starts the API cleanly
# Works on macOS, Linux, and Windows (WSL)

set -e  # Exit on any error

# Configuration
DEFAULT_PORT=8001
API_SCRIPT="real_cancer_alpha_api.py"
DEMO_API_SCRIPT="simple_cancer_api.py"

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

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to find process using port (cross-platform)
find_port_process() {
    local port=$1
    local os=$(detect_os)
    
    case $os in
        "macos"|"linux")
            lsof -ti :$port 2>/dev/null || echo ""
            ;;
        "windows")
            netstat -ano | grep ":$port " | awk '{print $5}' | head -1 || echo ""
            ;;
        *)
            print_warning "Unknown OS, trying lsof..."
            lsof -ti :$port 2>/dev/null || echo ""
            ;;
    esac
}

# Function to kill process by PID (cross-platform)
kill_process() {
    local pid=$1
    local os=$(detect_os)
    
    if [ -z "$pid" ]; then
        return 0
    fi
    
    case $os in
        "macos"|"linux")
            kill -9 $pid 2>/dev/null || true
            ;;
        "windows")
            taskkill /PID $pid /F 2>/dev/null || true
            ;;
        *)
            kill -9 $pid 2>/dev/null || true
            ;;
    esac
}

# Function to check if port is available
is_port_available() {
    local port=$1
    local pid=$(find_port_process $port)
    [ -z "$pid" ]
}

# Function to cleanup port
cleanup_port() {
    local port=$1
    print_status "Checking port $port..."
    
    local pid=$(find_port_process $port)
    if [ -n "$pid" ]; then
        print_warning "Found process $pid using port $port"
        print_status "Terminating process $pid..."
        kill_process $pid
        sleep 2
        
        # Verify it's really gone
        local still_running=$(find_port_process $port)
        if [ -n "$still_running" ]; then
            print_error "Failed to terminate process on port $port"
            return 1
        else
            print_success "Port $port is now available"
        fi
    else
        print_success "Port $port is already available"
    fi
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if API script exists
    if [ ! -f "$API_SCRIPT" ]; then
        print_error "API script $API_SCRIPT not found in current directory"
        print_status "Current directory: $(pwd)"
        print_status "Available Python files:"
        ls -la *.py 2>/dev/null || echo "No Python files found"
        exit 1
    fi
    
    # Check Python packages
    python3 -c "import fastapi, uvicorn, numpy, sklearn" 2>/dev/null || {
        print_error "Missing required Python packages"
        print_status "Install with: pip3 install fastapi uvicorn numpy scikit-learn"
        exit 1
    }
    
    print_success "All dependencies check passed"
}

# Function to show usage
show_usage() {
    echo "🧬 Cancer Alpha API Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --port PORT     Use custom port (default: $DEFAULT_PORT)"
    echo "  --demo          Start demo API instead of real models API"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start real models API on port $DEFAULT_PORT"
    echo "  $0 --port 8002       # Start on custom port"
    echo "  $0 --demo            # Start demo API with mock predictions"
}

# Parse command line arguments
PORT=$DEFAULT_PORT
USE_DEMO=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --demo)
            USE_DEMO=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "🧬 Cancer Alpha API Startup Script"
    echo "========================================"
    
    # Set API script based on mode
    if [ "$USE_DEMO" = true ]; then
        API_SCRIPT="$DEMO_API_SCRIPT"
        print_status "Starting DEMO API with mock predictions"
    else
        print_status "Starting REAL API with trained models"
    fi
    
    # Check dependencies
    check_dependencies
    
    # Cleanup port
    cleanup_port $PORT
    
    # Start the API
    print_status "Starting Cancer Alpha API on port $PORT..."
    echo "========================================"
    
    # Update port in API script if needed (for custom ports)
    if [ $PORT -ne $DEFAULT_PORT ]; then
        print_status "Using custom port $PORT"
        # Note: This assumes the API script uses uvicorn.run with configurable port
        # Most implementations should handle this via environment variables or command line
        export CANCER_ALPHA_PORT=$PORT
    fi
    
    # Start the API
    python3 "$API_SCRIPT"
}

# Run main function
main "$@"
