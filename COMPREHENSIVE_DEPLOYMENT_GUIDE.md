# 🚀 Cancer Alpha - Comprehensive Deployment Guide

## Overview

This comprehensive guide will help you successfully deploy and run the Cancer Alpha system using either Docker or Kubernetes. This guide addresses common deployment issues and provides step-by-step instructions for users at all levels.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Quick Start - Docker Compose (Recommended)](#quick-start---docker-compose-recommended)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Troubleshooting Common Issues](#troubleshooting-common-issues)
6. [Verification and Testing](#verification-and-testing)
7. [Production Configuration](#production-configuration)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Prerequisites

### 1. Required Software Installation

#### macOS (your current system)
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Docker Desktop
brew install --cask docker

# Install kubectl for Kubernetes
brew install kubectl

# Install Docker Compose (if not included with Docker Desktop)
brew install docker-compose

# Verify installations
docker --version
kubectl version --client
docker-compose --version
```

#### Linux (Ubuntu/Debian)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

#### Windows (PowerShell as Administrator)
```powershell
# Install Docker Desktop
# Download from: https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe

# Install kubectl using chocolatey
choco install kubernetes-cli

# Or download directly
curl -LO "https://dl.k8s.io/release/v1.28.0/bin/windows/amd64/kubectl.exe"
```

### 2. Docker Desktop Configuration

**Important for macOS users:**
1. Open Docker Desktop
2. Go to Settings → Advanced
3. Set CPU allocation: **4 CPUs minimum** (8 recommended)
4. Set Memory allocation: **8GB minimum** (16GB recommended)
5. Enable Kubernetes in Docker Desktop if using Kubernetes locally

---

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 20GB free space
- **Network**: Internet connection for image downloads

### Recommended Requirements
- **CPU**: 8+ cores
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ free space
- **Network**: High-speed internet for faster deployment

---

## Quick Start - Docker Compose (Recommended)

This is the easiest way to get Cancer Alpha running locally.

### Step 1: Prepare the Environment

```bash
# Navigate to the project directory
cd /Users/stillwell/projects/cancer-alpha

# Navigate to the deployment directory
cd src/phase4_systemization_and_tool_deployment
```

### Step 2: Validate Your Setup

```bash
# Run the validation script to check prerequisites
chmod +x deployment/scripts/validate-deployment.sh
./deployment/scripts/validate-deployment.sh
```

### Step 3: Fix Common Issues Before Deployment

#### Issue: Docker not running
```bash
# Start Docker Desktop application
# On macOS: Applications → Docker Desktop
# Wait for Docker Desktop to fully start (whale icon in menu bar should be stable)

# Verify Docker is running
docker info
```

#### Issue: Insufficient resources
```bash
# Check Docker resource allocation
docker system df
docker system info | grep -E "CPUs|Total Memory"

# If resources are low, adjust in Docker Desktop Settings
```

### Step 4: Deploy with Docker Compose

```bash
# Make the deploy script executable
chmod +x deployment/scripts/deploy-docker.sh

# Run the deployment script
./deployment/scripts/deploy-docker.sh

# If you encounter permission issues:
sudo chmod +x deployment/scripts/deploy-docker.sh
sudo ./deployment/scripts/deploy-docker.sh
```

### Step 5: Monitor the Deployment

```bash
# Watch the deployment progress
docker-compose -f deployment/docker/docker-compose.yml logs -f

# In another terminal, check service status
docker-compose -f deployment/docker/docker-compose.yml ps
```

### Expected Output

When successful, you should see:
```
✅ API service is ready!
✅ Web service is ready!
🌐 Access URLs:
  • Web Application: http://localhost:3000
  • API Documentation: http://localhost:8000/docs
  • API Health Check: http://localhost:8000/health
  • Prometheus: http://localhost:9090
  • Grafana: http://localhost:3001 (admin/admin123)
```

---

## Kubernetes Deployment

For production or if you prefer Kubernetes:

### Step 1: Setup Kubernetes Cluster

#### Option A: Local Kubernetes (Docker Desktop)
1. Open Docker Desktop
2. Go to Settings → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"
5. Wait for Kubernetes to start (green indicator)

#### Option B: Minikube
```bash
# Install minikube
brew install minikube

# Start minikube cluster
minikube start --driver=docker --cpus=4 --memory=8192

# Verify cluster is running
kubectl cluster-info
```

#### Option C: Cloud Kubernetes (AWS EKS, Google GKE, Azure AKS)
```bash
# AWS EKS example
aws eks update-kubeconfig --region us-west-2 --name your-cluster-name

# Google GKE example
gcloud container clusters get-credentials your-cluster-name --zone us-central1-a

# Azure AKS example
az aks get-credentials --resource-group your-rg --name your-cluster-name
```

### Step 2: Deploy to Kubernetes

```bash
# Make the Kubernetes deploy script executable
chmod +x deployment/scripts/deploy-kubernetes.sh

# Deploy to Kubernetes
./deployment/scripts/deploy-kubernetes.sh

# For custom registry (if pushing to DockerHub/ECR/GCR):
DOCKER_REGISTRY=your-registry.com ./deployment/scripts/deploy-kubernetes.sh
```

### Step 3: Access the Application

```bash
# Setup port forwarding for local access
kubectl port-forward svc/cancer-alpha-api-service 8000:8000 -n cancer-alpha &
kubectl port-forward svc/cancer-alpha-web-service 3000:3000 -n cancer-alpha &

# Access the application
# Web App: http://localhost:3000
# API: http://localhost:8000/docs
```

---

## Troubleshooting Common Issues

### 1. Docker Issues

#### "Docker daemon not running"
```bash
# macOS: Start Docker Desktop application
# Linux: Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

#### "Permission denied while trying to connect to Docker daemon"
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo temporarily
sudo docker ps
```

#### "Port already in use"
```bash
# Find process using the port
lsof -i :8000  # For API port
lsof -i :3000  # For web port

# Kill the process
kill -9 <PID>

# Or use different ports
API_PORT=8001 WEB_PORT=3001 ./deployment/scripts/deploy-docker.sh
```

### 2. Memory/Resource Issues

#### "Out of memory" or slow performance
```bash
# Increase Docker memory allocation in Docker Desktop settings
# Or prune unused Docker resources
docker system prune -a --volumes

# Check current resource usage
docker stats

# Reduce service replicas in docker-compose.yml
# Change replicas: 3 to replicas: 1
```

### 3. Image Build Issues

#### "Failed to build Docker image"
```bash
# Clean Docker cache
docker system prune -a

# Build images individually for debugging
cd deployment/docker
docker build -f Dockerfile.api -t cancer-alpha/api:latest ../../../
docker build -f Dockerfile.web -t cancer-alpha/web:latest ../../../

# Check for specific error messages in build logs
```

### 4. Model Loading Issues

#### "Models not found"
```bash
# Check if model files exist
ls -la ../../../models/phase2_models/

# If models are missing, run training first
cd ../../../
python src/phase2_fixed_model_training.py

# Verify models are created
ls -la models/phase2_models/
```

### 5. Network Issues

#### "Service unreachable" or "Connection refused"
```bash
# Check service status
docker-compose ps

# Restart specific services
docker-compose restart cancer-alpha-api
docker-compose restart cancer-alpha-web

# Check service logs for errors
docker-compose logs cancer-alpha-api
docker-compose logs cancer-alpha-web
```

### 6. Kubernetes-Specific Issues

#### "kubectl command not found"
```bash
# Install kubectl
brew install kubectl  # macOS
sudo snap install kubectl --classic  # Linux
```

#### "Unable to connect to cluster"
```bash
# Check cluster status
kubectl cluster-info

# If using minikube
minikube status
minikube start

# If using Docker Desktop Kubernetes
# Restart Docker Desktop and re-enable Kubernetes
```

#### "ImagePullBackOff" errors
```bash
# Build and tag images locally
docker build -t cancer-alpha/api:latest -f deployment/docker/Dockerfile.api .
docker build -t cancer-alpha/web:latest -f deployment/docker/Dockerfile.web .

# For minikube, use local Docker daemon
eval $(minikube docker-env)
# Then rebuild images
```

---

## Verification and Testing

### 1. Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check web application
curl http://localhost:3000/

# Check all services
curl -s http://localhost:8000/health | jq .
```

### 2. Functional Testing

```bash
# Test API prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "TEST_001",
    "age": 65,
    "gender": "M",
    "features": {
      "gene_1": 0.5,
      "gene_2": 0.8,
      "gene_3": 0.3
    },
    "model_type": "ensemble"
  }'
```

### 3. Load Testing (Optional)

```bash
# Install load testing tools
pip install locust

# Run load test
locust -f deployment/testing/load_test.py --host http://localhost:8000
```

---

## Production Configuration

### 1. Environment Variables

Create environment-specific configuration files:

```bash
# Production environment
cp deployment/environments/prod.env .env

# Edit production settings
nano .env
```

### 2. Security Hardening

```bash
# Generate secure secrets
openssl rand -base64 32  # For JWT secret

# Create Kubernetes secrets
kubectl create secret generic cancer-alpha-secrets \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  --from-literal=api-key=$(openssl rand -base64 32) \
  -n cancer-alpha
```

### 3. SSL/TLS Configuration

```bash
# For production, use proper SSL certificates
# Update nginx configuration with SSL settings
# Configure ingress with TLS termination
```

### 4. Resource Optimization

```bash
# Adjust resource limits in production
# Edit docker-compose.yml or kubernetes manifests
# Set appropriate CPU/Memory limits
```

---

## Monitoring and Maintenance

### 1. Viewing Logs

```bash
# Docker Compose logs
docker-compose -f deployment/docker/docker-compose.yml logs -f

# Kubernetes logs
kubectl logs -f deployment/cancer-alpha-api -n cancer-alpha

# Specific service logs
docker logs cancer-alpha-api
```

### 2. Monitoring Dashboards

- **Grafana**: http://localhost:3001 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **API Metrics**: http://localhost:8000/metrics

### 3. Backup and Recovery

```bash
# Backup persistent data
docker-compose -f deployment/docker/docker-compose.yml exec postgres pg_dump -U cancer_alpha > backup.sql

# Backup Kubernetes persistent volumes
kubectl get pv -n cancer-alpha
```

### 4. Updates and Maintenance

```bash
# Update images
docker-compose -f deployment/docker/docker-compose.yml pull
docker-compose -f deployment/docker/docker-compose.yml up -d

# Rolling updates in Kubernetes
kubectl rollout restart deployment/cancer-alpha-api -n cancer-alpha
```

---

## Quick Command Reference

### Docker Compose Commands
```bash
# Start services
./deployment/scripts/deploy-docker.sh

# View status
docker-compose -f deployment/docker/docker-compose.yml ps

# View logs
docker-compose -f deployment/docker/docker-compose.yml logs -f

# Stop services
docker-compose -f deployment/docker/docker-compose.yml down

# Clean up everything
docker-compose -f deployment/docker/docker-compose.yml down --volumes --remove-orphans
```

### Kubernetes Commands
```bash
# Deploy
./deployment/scripts/deploy-kubernetes.sh

# Check status
kubectl get all -n cancer-alpha

# Port forward
kubectl port-forward svc/cancer-alpha-api-service 8000:8000 -n cancer-alpha

# Delete deployment
kubectl delete namespace cancer-alpha
```

---

## Support and Troubleshooting

If you continue to experience issues:

1. **Check logs**: Always start by checking service logs for error messages
2. **Resource constraints**: Ensure sufficient CPU/Memory allocation
3. **Port conflicts**: Make sure required ports are available
4. **Docker daemon**: Ensure Docker is running and accessible
5. **Model files**: Verify that trained models exist in the expected location

For additional support, please check:
- Docker Desktop documentation
- Kubernetes documentation
- Project-specific documentation in the `docs/` directory

---

## 🎉 Success!

Once deployed successfully, you'll have:
- ✅ Full-featured Cancer Alpha web application
- ✅ REST API with automatic documentation
- ✅ Real-time prediction capabilities
- ✅ Monitoring and observability
- ✅ Production-ready configuration

Access your deployment at:
- **Web Application**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

The Cancer Alpha system is now ready for clinical evaluation and research use!
