#!/bin/bash

# Cancer Alpha - Deployment Setup Script
# This script prepares the environment for deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Cancer Alpha - Deployment Setup${NC}"
echo -e "${BLUE}===================================${NC}"

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
print_status "Project root: $PROJECT_ROOT"

# 1. Create models/phase2_models directory and copy models
print_status "Setting up models directory..."

if [ ! -d "$PROJECT_ROOT/models/phase2_models" ]; then
    mkdir -p "$PROJECT_ROOT/models/phase2_models"
    print_status "Created models/phase2_models directory"
fi

# Copy models from results/phase2_optimized to models/phase2_models
if [ -d "$PROJECT_ROOT/results/phase2_optimized" ]; then
    cp "$PROJECT_ROOT/results/phase2_optimized"/*.pkl "$PROJECT_ROOT/models/phase2_models/" 2>/dev/null || true
    model_count=$(find "$PROJECT_ROOT/models/phase2_models" -name "*.pkl" | wc -l)
    print_status "Copied $model_count model files to models/phase2_models"
else
    print_warning "results/phase2_optimized directory not found"
    print_warning "You may need to run model training first"
fi

# 2. Check and start Docker Desktop if not running
print_status "Checking Docker status..."

if ! docker info >/dev/null 2>&1; then
    print_warning "Docker is not running"
    print_status "Please start Docker Desktop and wait for it to be ready"
    echo "Press any key to continue once Docker Desktop is running..."
    read -n 1 -r
fi

# 3. Verify required tools
print_status "Verifying required tools..."

# Check Docker
if command -v docker &> /dev/null; then
    print_status "✅ Docker is installed: $(docker --version)"
else
    print_error "❌ Docker is not installed"
    exit 1
fi

# Check docker-compose
if command -v docker-compose &> /dev/null; then
    print_status "✅ Docker Compose is installed: $(docker-compose --version)"
else
    print_error "❌ Docker Compose is not installed"
    print_status "Installing Docker Compose..."
    brew install docker-compose
fi

# Check kubectl
if command -v kubectl &> /dev/null; then
    print_status "✅ kubectl is installed: $(kubectl version --client --short 2>/dev/null)"
else
    print_warning "⚠️ kubectl is not installed (only needed for Kubernetes deployment)"
fi

# 4. Create environment file if not exists
print_status "Setting up environment configuration..."

ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << EOF
# Cancer Alpha Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=info
API_HOST=0.0.0.0
API_PORT=8000
WEB_PORT=3000

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# Model Configuration
MODEL_PATH=/app/models/phase2_models

# Security
JWT_SECRET=your-secret-key-here-change-in-production
API_KEY=your-api-key-here-change-in-production
EOF
    print_status "Created .env file with default configuration"
else
    print_status "Environment file already exists"
fi

# 5. Check port availability
print_status "Checking port availability..."

check_port() {
    local port=$1
    local service=$2
    if command -v lsof &> /dev/null; then
        if lsof -i ":$port" >/dev/null 2>&1; then
            print_warning "Port $port is already in use (required for $service)"
            lsof -i ":$port"
            return 1
        else
            print_status "✅ Port $port is available for $service"
            return 0
        fi
    else
        print_warning "lsof not available, skipping port check"
        return 0
    fi
}

check_port 8000 "API"
check_port 3000 "Web App"
check_port 6379 "Redis"
check_port 9090 "Prometheus"
check_port 3001 "Grafana"

# 6. Clean up Docker resources if requested
if [ "${1:-}" = "--clean" ]; then
    print_status "Cleaning up Docker resources..."
    docker system prune -f
    print_status "Docker cleanup completed"
fi

# 7. Make deployment scripts executable
print_status "Making deployment scripts executable..."
chmod +x "$PROJECT_ROOT/src/phase4_systemization_and_tool_deployment/deployment/scripts"/*.sh

print_status "🎉 Setup completed successfully!"
echo
print_status "📋 Next Steps:"
echo "1. To deploy with Docker Compose (recommended):"
echo "   cd src/phase4_systemization_and_tool_deployment"
echo "   ./deployment/scripts/deploy-docker.sh"
echo
echo "2. To deploy with Kubernetes:"
echo "   cd src/phase4_systemization_and_tool_deployment"
echo "   ./deployment/scripts/deploy-kubernetes.sh"
echo
echo "3. To validate your setup:"
echo "   cd src/phase4_systemization_and_tool_deployment"
echo "   ./deployment/scripts/validate-deployment.sh"
echo
print_status "📚 For detailed instructions, see: COMPREHENSIVE_DEPLOYMENT_GUIDE.md"
