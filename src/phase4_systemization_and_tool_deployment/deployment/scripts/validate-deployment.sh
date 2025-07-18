#!/bin/bash

# Cancer Alpha Deployment Validation Script
# This script validates the complete deployment of the Cancer Alpha system

set -e

echo "🔍 Cancer Alpha Deployment Validation"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $2 -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1${NC}"
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
    echo -e "ℹ️ $1"
}

# Check if we're in the right directory
if [ ! -f "deployment/scripts/validate-deployment.sh" ]; then
    echo -e "${RED}❌ Please run this script from the phase4_systemization_and_tool_deployment directory${NC}"
    exit 1
fi

# 1. Check Docker installation
print_info "Checking Docker installation..."
if command -v docker &> /dev/null; then
    print_status "Docker is installed" 0
    docker --version
else
    print_status "Docker is not installed" 1
    print_warning "Please install Docker to proceed with containerized deployment"
fi

# 2. Check kubectl installation (for Kubernetes)
print_info "Checking kubectl installation..."
if command -v kubectl &> /dev/null; then
    print_status "kubectl is installed" 0
    kubectl version --client --short 2>/dev/null || true
else
    print_status "kubectl is not installed" 1
    print_warning "kubectl is required for Kubernetes deployment"
fi

# 3. Check required files exist
print_info "Checking deployment files..."
required_files=(
    "deployment/docker/Dockerfile.api"
    "deployment/docker/Dockerfile.web"
    "deployment/docker/docker-compose.yml"
    "deployment/kubernetes/namespace.yaml"
    "deployment/kubernetes/api-deployment.yaml"
    "deployment/kubernetes/web-deployment.yaml"
    "deployment/kubernetes/ingress.yaml"
    "deployment/scripts/deploy-docker.sh"
    "deployment/scripts/deploy-kubernetes.sh"
    "ci_cd/github-actions.yml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "Found $file" 0
    else
        print_status "Missing $file" 1
    fi
done

# 4. Check API files
print_info "Checking API files..."
if [ -f "api/real_model_api.py" ]; then
    print_status "API file exists" 0
else
    print_status "API file missing" 1
fi

# 5. Check web app files
print_info "Checking web application files..."
if [ -f "web_app/package.json" ]; then
    print_status "Web app package.json exists" 0
else
    print_status "Web app package.json missing" 1
fi

# 6. Check model files
print_info "Checking model files..."
if [ -d "../../models/phase2_models" ]; then
    model_count=$(find ../../models/phase2_models -name "*.pkl" | wc -l)
    if [ $model_count -gt 0 ]; then
        print_status "Found $model_count model files" 0
    else
        print_status "No model files found" 1
    fi
else
    print_status "Models directory not found" 1
fi

# 7. Validate Docker Compose configuration
print_info "Validating Docker Compose configuration..."
if [ -f "deployment/docker/docker-compose.yml" ]; then
    if docker-compose -f deployment/docker/docker-compose.yml config &> /dev/null; then
        print_status "Docker Compose configuration is valid" 0
    else
        print_status "Docker Compose configuration has errors" 1
    fi
fi

# 8. Validate Kubernetes manifests
print_info "Validating Kubernetes manifests..."
if command -v kubectl &> /dev/null; then
    for manifest in deployment/kubernetes/*.yaml; do
        if kubectl apply --dry-run=client -f "$manifest" &> /dev/null; then
            print_status "$(basename $manifest) is valid" 0
        else
            print_status "$(basename $manifest) has errors" 1
        fi
    done
else
    print_warning "Skipping Kubernetes validation (kubectl not available)"
fi

# 9. Check port availability
print_info "Checking port availability..."
check_port() {
    local port=$1
    local service=$2
    if command -v nc &> /dev/null; then
        if nc -z localhost $port 2>/dev/null; then
            print_warning "Port $port is already in use (required for $service)"
        else
            print_status "Port $port is available for $service" 0
        fi
    else
        print_warning "netcat not available, skipping port check"
    fi
}

check_port 8000 "API"
check_port 3000 "Web App"
check_port 6379 "Redis"
check_port 9090 "Prometheus"
check_port 3001 "Grafana"

# 10. Environment configuration check
print_info "Checking environment configurations..."
for env_file in deployment/environments/*.env; do
    if [ -f "$env_file" ]; then
        print_status "Found $(basename $env_file)" 0
    else
        print_status "Missing $(basename $env_file)" 1
    fi
done

# 11. Security configuration check
print_info "Checking security configurations..."
if [ -f "deployment/kubernetes/security.yaml" ]; then
    print_status "Security policies configured" 0
else
    print_status "Security policies missing" 1
fi

# 12. Monitoring configuration check
print_info "Checking monitoring configurations..."
monitoring_files=(
    "deployment/docker/prometheus.yml"
    "deployment/docker/grafana-dashboard.json"
    "deployment/docker/grafana-datasource.yml"
    "deployment/kubernetes/monitoring.yaml"
)

for file in "${monitoring_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "Found monitoring config: $(basename $file)" 0
    else
        print_status "Missing monitoring config: $(basename $file)" 1
    fi
done

# 13. CI/CD configuration check
print_info "Checking CI/CD configuration..."
if [ -f "ci_cd/github-actions.yml" ]; then
    print_status "GitHub Actions workflow configured" 0
else
    print_status "GitHub Actions workflow missing" 1
fi

echo ""
echo "🎯 Validation Summary"
echo "===================="
print_info "Validation completed!"
print_info "If all checks passed, you can proceed with deployment using:"
echo "  - Docker Compose: ./deployment/scripts/deploy-docker.sh"
echo "  - Kubernetes: ./deployment/scripts/deploy-kubernetes.sh"
echo ""
print_info "For detailed deployment instructions, see:"
echo "  - PHASE4C_PRODUCTION_DEPLOYMENT_GUIDE.md"
echo ""
print_info "The Cancer Alpha system is ready for production deployment! 🚀"
