#!/bin/bash
# Cancer Alpha - Docker Deployment Script
# This script builds and deploys the Cancer Alpha system using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/../docker"

echo -e "${BLUE}🚀 Cancer Alpha - Docker Deployment${NC}"
echo -e "${BLUE}====================================${NC}"

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

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are ready!"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    cd "$DOCKER_DIR"
    
    # Build API image
    print_status "Building Cancer Alpha API image..."
    docker build -f Dockerfile.api -t cancer-alpha/api:latest "$PROJECT_ROOT"
    
    # Build Web image
    print_status "Building Cancer Alpha Web image..."
    docker build -f Dockerfile.web -t cancer-alpha/web:latest "$PROJECT_ROOT"
    
    print_status "Docker images built successfully!"
}

# Deploy with Docker Compose
deploy_compose() {
    print_status "Deploying with Docker Compose..."
    
    cd "$DOCKER_DIR"
    
    # Stop existing containers if running
    docker-compose down --remove-orphans
    
    # Start services
    docker-compose up -d
    
    print_status "Deployment completed!"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for API
    local api_ready=false
    local web_ready=false
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if ! $api_ready && curl -f http://localhost:8000/health &> /dev/null; then
            print_status "✅ API service is ready!"
            api_ready=true
        fi
        
        if ! $web_ready && curl -f http://localhost:3000/health &> /dev/null; then
            print_status "✅ Web service is ready!"
            web_ready=true
        fi
        
        if $api_ready && $web_ready; then
            break
        fi
        
        ((attempt++))
        print_status "Waiting for services... ($attempt/$max_attempts)"
        sleep 5
    done
    
    if ! $api_ready || ! $web_ready; then
        print_warning "Some services may not be ready yet. Check logs with: docker-compose logs"
    fi
}

# Show deployment status
show_status() {
    print_status "Deployment Status:"
    echo
    docker-compose ps
    echo
    
    print_status "🌐 Access URLs:"
    echo "  • Web Application: http://localhost:3000"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • API Health Check: http://localhost:8000/health"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3001 (admin/admin123)"
    echo
    
    print_status "📊 Useful Commands:"
    echo "  • View logs: docker-compose logs -f"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart services: docker-compose restart"
    echo "  • Update and restart: $0"
    echo
}

# Main deployment function
main() {
    print_status "Starting Cancer Alpha deployment..."
    
    check_docker
    build_images
    deploy_compose
    wait_for_services
    show_status
    
    print_status "🎉 Cancer Alpha deployment completed successfully!"
    print_status "The system is now running in production mode with Docker."
}

# Parse command line arguments
case "${1:-}" in
    "build")
        check_docker
        build_images
        ;;
    "deploy")
        check_docker
        deploy_compose
        wait_for_services
        show_status
        ;;
    "status")
        show_status
        ;;
    "logs")
        cd "$DOCKER_DIR"
        docker-compose logs -f
        ;;
    "stop")
        cd "$DOCKER_DIR"
        docker-compose down
        print_status "Cancer Alpha services stopped."
        ;;
    "restart")
        cd "$DOCKER_DIR"
        docker-compose restart
        print_status "Cancer Alpha services restarted."
        ;;
    "clean")
        cd "$DOCKER_DIR"
        docker-compose down --volumes --remove-orphans
        docker system prune -f
        print_status "Cleaned up Docker resources."
        ;;
    *)
        main
        ;;
esac
