#!/bin/bash

# Cancer Alpha - Quick Status Check
# ================================
# 
# Quick utility to check the status of Cancer Alpha system components
# Usage: ./scripts/quick_status.sh

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# API URL
API_URL="http://localhost:8001"

echo -e "${BLUE}🧬 Cancer Alpha - Quick Status Check${NC}"
echo "========================================"
echo

# Function to check if API is running
check_api_status() {
    echo -e "${BLUE}🔍 Checking API Status...${NC}"
    
    if curl -s -f "${API_URL}/health" > /dev/null 2>&1; then
        # Get API health data
        HEALTH_DATA=$(curl -s "${API_URL}/health")
        API_VERSION=$(echo "$HEALTH_DATA" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('version', 'unknown'))" 2>/dev/null || echo "unknown")
        MODELS_LOADED=$(echo "$HEALTH_DATA" | python3 -c "import sys, json; data=json.load(sys.stdin); print('true' if data.get('models_loaded', False) else 'false')" 2>/dev/null || echo "false")
        
        echo -e "   ${GREEN}✅ API is running${NC} (${API_VERSION})"
        if [ "$MODELS_LOADED" = "true" ]; then
            echo -e "   ${GREEN}✅ Models are loaded${NC}"
        else
            echo -e "   ${RED}❌ Models not loaded${NC}"
        fi
        echo -e "   🌐 URL: ${API_URL}"
        echo -e "   📚 Docs: ${API_URL}/docs"
        return 0
    else
        echo -e "   ${RED}❌ API not accessible${NC}"
        echo -e "   💡 Try: python3 real_cancer_alpha_api.py"
        return 1
    fi
}

# Function to check model files
check_model_files() {
    echo -e "${BLUE}🗂 Checking Model Files...${NC}"
    
    MODEL_DIR="results/phase2"
    EXPECTED_FILES=(
        "random_forest_model.pkl"
        "gradient_boosting_model.pkl"
        "deep_neural_network_model.pkl"
        "ensemble_model.pkl"
        "scaler.pkl"
    )
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo -e "   ${RED}❌ Model directory not found: $MODEL_DIR${NC}"
        echo -e "   💡 Try: python3 src/phase2_model_training_and_validation/phase2_fixed_model_training.py"
        return 1
    fi
    
    MISSING_FILES=()
    TOTAL_SIZE=0
    
    for file in "${EXPECTED_FILES[@]}"; do
        if [ -f "$MODEL_DIR/$file" ]; then
            SIZE=$(stat -f%z "$MODEL_DIR/$file" 2>/dev/null || stat -c%s "$MODEL_DIR/$file" 2>/dev/null || echo 0)
            SIZE_MB=$((SIZE / 1024 / 1024))
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE_MB))
            echo -e "   ${GREEN}✅${NC} $file (${SIZE_MB}MB)"
        else
            MISSING_FILES+=("$file")
            echo -e "   ${RED}❌${NC} $file"
        fi
    done
    
    echo -e "   📊 Total size: ${TOTAL_SIZE}MB"
    
    if [ ${#MISSING_FILES[@]} -eq 0 ]; then
        echo -e "   ${GREEN}✅ All model files present${NC}"
        return 0
    else
        echo -e "   ${YELLOW}⚠️ Missing ${#MISSING_FILES[@]} model files${NC}"
        return 1
    fi
}

# Function to check web app
check_web_app() {
    echo -e "${BLUE}🌐 Checking Web Application...${NC}"
    
    WEB_APP_DIR="src/phase4_systemization_and_tool_deployment/web_app"
    
    if [ ! -d "$WEB_APP_DIR" ]; then
        echo -e "   ${RED}❌ Web app directory not found${NC}"
        return 1
    fi
    
    if [ -f "$WEB_APP_DIR/package.json" ]; then
        echo -e "   ${GREEN}✅${NC} package.json found"
    else
        echo -e "   ${RED}❌${NC} package.json missing"
    fi
    
    if [ -d "$WEB_APP_DIR/node_modules" ]; then
        echo -e "   ${GREEN}✅${NC} node_modules installed"
    else
        echo -e "   ${YELLOW}⚠️${NC} node_modules not found"
        echo -e "   💡 Try: cd $WEB_APP_DIR && npm install"
    fi
    
    if [ -d "$WEB_APP_DIR/build" ]; then
        echo -e "   ${GREEN}✅${NC} Build directory exists"
    else
        echo -e "   ${YELLOW}⚠️${NC} No build directory"
        echo -e "   💡 Try: cd $WEB_APP_DIR && npm run build"
    fi
    
    # Check if web app is running (common ports)
    WEB_PORTS=(3000 3001 8080)
    WEB_RUNNING=false
    
    for port in "${WEB_PORTS[@]}"; do
        if curl -s -f "http://localhost:$port" > /dev/null 2>&1; then
            echo -e "   ${GREEN}✅ Web app running on port $port${NC}"
            echo -e "   🌐 URL: http://localhost:$port"
            WEB_RUNNING=true
            break
        fi
    done
    
    if [ "$WEB_RUNNING" = false ]; then
        echo -e "   ${YELLOW}⚠️ Web app not running${NC}"
        echo -e "   💡 Try: cd $WEB_APP_DIR && npm start"
    fi
}

# Function to check documentation
check_documentation() {
    echo -e "${BLUE}📚 Checking Documentation...${NC}"
    
    DOCS=(
        "README.md"
        "MASTER_INSTALLATION_GUIDE.md"
        "docs/API_REFERENCE_GUIDE.md"
        "docs/UPDATED_PROJECT_ROADMAP_2025.md"
    )
    
    for doc in "${DOCS[@]}"; do
        if [ -f "$doc" ]; then
            echo -e "   ${GREEN}✅${NC} $doc"
        else
            echo -e "   ${RED}❌${NC} $doc"
        fi
    done
}

# Function to check Python dependencies
check_python_deps() {
    echo -e "${BLUE}🐍 Checking Python Dependencies...${NC}"
    
    REQUIRED_PACKAGES=(
        "fastapi"
        "uvicorn"
        "pandas"
        "numpy"
        "scikit-learn"
        "psutil"
        "requests"
    )
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            echo -e "   ${GREEN}✅${NC} $package"
        else
            echo -e "   ${RED}❌${NC} $package"
        fi
    done
}

# Function to show quick commands
show_quick_commands() {
    echo -e "${BLUE}⚡ Quick Commands${NC}"
    echo "================="
    echo -e "${YELLOW}Start API:${NC}        python3 real_cancer_alpha_api.py"
    echo -e "${YELLOW}Start Web App:${NC}    cd src/phase4_systemization_and_tool_deployment/web_app && npm start"
    echo -e "${YELLOW}Train Models:${NC}     python3 src/phase2_model_training_and_validation/phase2_fixed_model_training.py"
    echo -e "${YELLOW}System Monitor:${NC}   python3 utils/system_monitor.py"
    echo -e "${YELLOW}API Docs:${NC}         open http://localhost:8001/docs"
    echo -e "${YELLOW}Test API:${NC}         curl http://localhost:8001/health"
    echo
}

# Main execution
main() {
    # Run all checks
    check_api_status
    echo
    
    check_model_files
    echo
    
    check_web_app
    echo
    
    check_documentation
    echo
    
    check_python_deps
    echo
    
    show_quick_commands
    
    echo "========================================"
    echo -e "${BLUE}🎉 Status check complete!${NC}"
    echo -e "💡 For detailed monitoring: ${YELLOW}python3 utils/system_monitor.py${NC}"
}

# Run main function
main
