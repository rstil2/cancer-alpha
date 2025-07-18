#!/bin/bash
# Cancer Alpha - Kubernetes Deployment Script
# This script deploys the Cancer Alpha system to Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
K8S_DIR="$PROJECT_ROOT/src/phase4_systemization_and_tool_deployment/deployment/kubernetes"
DOCKER_DIR="$PROJECT_ROOT/src/phase4_systemization_and_tool_deployment/deployment/docker"

echo -e "${BLUE}🚀 Cancer Alpha - Kubernetes Deployment${NC}"
echo -e "${BLUE}======================================${NC}"

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

# Check if kubectl is installed and configured
check_kubectl() {
    print_status "Checking kubectl installation..."
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        print_error "kubectl is not configured or cluster is not accessible."
        exit 1
    fi
    
    print_status "kubectl is ready!"
    kubectl cluster-info
}

# Build and push Docker images
build_and_push_images() {
    print_status "Building and pushing Docker images..."
    
    local registry="${DOCKER_REGISTRY:-cancer-alpha}"
    local tag="${IMAGE_TAG:-latest}"
    
    cd "$DOCKER_DIR"
    
    # Build API image
    print_status "Building Cancer Alpha API image..."
    docker build -f Dockerfile.api -t "$registry/api:$tag" "$PROJECT_ROOT"
    
    # Build Web image
    print_status "Building Cancer Alpha Web image..."
    docker build -f Dockerfile.web -t "$registry/web:$tag" "$PROJECT_ROOT"
    
    # Push images if registry is specified
    if [[ "$registry" != "cancer-alpha" ]]; then
        print_status "Pushing images to registry..."
        docker push "$registry/api:$tag"
        docker push "$registry/web:$tag"
    fi
    
    print_status "Docker images ready!"
}

# Create namespace
create_namespace() {
    print_status "Creating namespace..."
    
    kubectl apply -f "$K8S_DIR/namespace.yaml"
    
    print_status "Namespace created!"
}

# Deploy applications
deploy_applications() {
    print_status "Deploying applications..."
    
    # Deploy API
    print_status "Deploying Cancer Alpha API..."
    kubectl apply -f "$K8S_DIR/api-deployment.yaml"
    
    # Deploy Web
    print_status "Deploying Cancer Alpha Web..."
    kubectl apply -f "$K8S_DIR/web-deployment.yaml"
    
    # Deploy Ingress
    print_status "Deploying Ingress..."
    kubectl apply -f "$K8S_DIR/ingress.yaml"
    
    print_status "Applications deployed!"
}

# Wait for deployments to be ready
wait_for_deployments() {
    print_status "Waiting for deployments to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s deployment/cancer-alpha-api -n cancer-alpha
    kubectl wait --for=condition=available --timeout=300s deployment/cancer-alpha-web -n cancer-alpha
    
    print_status "✅ All deployments are ready!"
}

# Show deployment status
show_status() {
    print_status "Deployment Status:"
    echo
    
    kubectl get all -n cancer-alpha
    echo
    
    print_status "🌐 Access Information:"
    
    # Get LoadBalancer IPs
    local web_ip=$(kubectl get svc cancer-alpha-web-loadbalancer -n cancer-alpha -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    local api_ip=$(kubectl get svc cancer-alpha-api-loadbalancer -n cancer-alpha -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    echo "  • Web Application: http://$web_ip (if LoadBalancer IP is available)"
    echo "  • API: http://$api_ip (if LoadBalancer IP is available)"
    echo
    
    print_status "📊 Useful Commands:"
    echo "  • View pods: kubectl get pods -n cancer-alpha"
    echo "  • View services: kubectl get svc -n cancer-alpha"
    echo "  • View logs: kubectl logs -f deployment/cancer-alpha-api -n cancer-alpha"
    echo "  • Port forward API: kubectl port-forward svc/cancer-alpha-api-service 8000:8000 -n cancer-alpha"
    echo "  • Port forward Web: kubectl port-forward svc/cancer-alpha-web-service 3000:3000 -n cancer-alpha"
    echo "  • Delete deployment: kubectl delete namespace cancer-alpha"
    echo
}

# Setup port forwarding for local access
setup_port_forwarding() {
    print_status "Setting up port forwarding for local access..."
    
    print_status "Starting port forwarding in background..."
    
    # Kill existing port forwards
    pkill -f "kubectl port-forward.*cancer-alpha" || true
    
    # Start new port forwards
    kubectl port-forward svc/cancer-alpha-api-service 8000:8000 -n cancer-alpha &
    kubectl port-forward svc/cancer-alpha-web-service 3000:3000 -n cancer-alpha &
    
    sleep 5
    
    print_status "🌐 Local Access URLs:"
    echo "  • Web Application: http://localhost:3000"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • API Health Check: http://localhost:8000/health"
    echo
    print_status "Port forwarding is running in background. Use 'pkill -f kubectl' to stop."
}

# Main deployment function
main() {
    print_status "Starting Cancer Alpha Kubernetes deployment..."
    
    check_kubectl
    build_and_push_images
    create_namespace
    deploy_applications
    wait_for_deployments
    show_status
    
    print_status "🎉 Cancer Alpha deployment completed successfully!"
    print_status "The system is now running on Kubernetes."
    
    # Ask if user wants port forwarding
    echo
    read -p "Do you want to setup port forwarding for local access? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_port_forwarding
    fi
}

# Parse command line arguments
case "${1:-}" in
    "build")
        build_and_push_images
        ;;
    "deploy")
        check_kubectl
        create_namespace
        deploy_applications
        wait_for_deployments
        show_status
        ;;
    "status")
        show_status
        ;;
    "logs")
        kubectl logs -f deployment/cancer-alpha-api -n cancer-alpha
        ;;
    "port-forward")
        setup_port_forwarding
        ;;
    "delete")
        print_status "Deleting Cancer Alpha deployment..."
        kubectl delete namespace cancer-alpha
        print_status "Cancer Alpha deployment deleted."
        ;;
    "scale")
        local replicas=${2:-3}
        print_status "Scaling deployments to $replicas replicas..."
        kubectl scale deployment cancer-alpha-api --replicas=$replicas -n cancer-alpha
        kubectl scale deployment cancer-alpha-web --replicas=$replicas -n cancer-alpha
        print_status "Deployments scaled to $replicas replicas."
        ;;
    *)
        main
        ;;
esac
