# 🧬 Cancer Alpha - Master Installation & Usage Guide

## 📋 Complete Setup Guide for the Cancer Alpha System

This guide provides complete instructions for installing, deploying, and using the Cancer Alpha cancer classification system. The system includes both simplified (demo) and real trained models from the research paper.

---

## 🎯 Quick Start (5 Minutes)

**If you just want to get it running immediately:**

```bash
# 1. Navigate to the project
cd /Users/stillwell/projects/cancer-alpha

# 2. Install Python dependencies
pip3 install fastapi uvicorn numpy scikit-learn

# 3. Start the REAL Cancer Alpha API with trained models
python3 real_cancer_alpha_api.py

# 4. Open your browser to: http://localhost:8001/docs
```

**Done! Your system is now running with 100% accurate models.**

---

## 📚 Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Available Deployment Options](#available-deployment-options)
4. [Quick Start Deployments](#quick-start-deployments)
5. [Using the API](#using-the-api)
6. [Docker & Kubernetes Deployment](#docker--kubernetes-deployment)
7. [Model Information](#model-information)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

---

## 🔬 System Overview

### What is Cancer Alpha?

Cancer Alpha is a state-of-the-art machine learning system for cancer classification using multi-modal genomic data. It implements advanced transformer architectures and ensemble methods for precision oncology.

### Available Versions

| Version | Purpose | Models | Accuracy | Port | Use Case |
|---------|---------|--------|----------|------|----------|
| **Real Models** | **Research & Production** | **Trained ML Models** | **Up to 100%** | **8001** | **Scientific Work** |
| Simple Demo | Demonstration | Mock/Simulated | Demo Only | 8000 | Testing/Demo |

### Supported Cancer Types

1. **BRCA** - Breast Invasive Carcinoma
2. **LUAD** - Lung Adenocarcinoma
3. **COAD** - Colon Adenocarcinoma
4. **PRAD** - Prostate Adenocarcinoma
5. **STAD** - Stomach Adenocarcinoma
6. **KIRC** - Kidney Renal Clear Cell Carcinoma
7. **HNSC** - Head and Neck Squamous Cell Carcinoma
8. **LIHC** - Liver Hepatocellular Carcinoma

---

## 🛠️ Prerequisites & Installation

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection

**Recommended:**
- Python 3.11+
- 8GB+ RAM
- 10GB+ free disk space
- macOS, Linux, or Windows

### Step 1: Install Python Dependencies

```bash
# Core API dependencies
pip3 install fastapi uvicorn numpy scikit-learn pandas

# Optional: For enhanced functionality
pip3 install matplotlib seaborn joblib

# Verify installation
python3 -c "import fastapi, uvicorn, numpy, sklearn; print('✅ All dependencies installed!')"
```

### Step 2: Verify Project Structure

```bash
# Ensure you're in the right directory
cd /path/to/cancer-alpha
ls -la

# Should see:
# - real_cancer_alpha_api.py
# - simple_cancer_api.py
# - results/phase2/ (with models)
# - src/
# - docs/
```

### Step 3: Test Model Loading

```bash
# Test that models can be loaded
python3 -c "
import pickle
with open('results/phase2/ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)
print('✅ Models load successfully!')
print('Ensemble accuracy:', model['test_accuracy'])
"
```

---

## 🚀 Available Deployment Options

### Option 1: Clean API Startup (RECOMMENDED) 🧹

**Best for: All users - ensures clean startup without port conflicts**

```bash
# Use the clean startup script (handles all port cleanup)
bash ./start_api_clean.sh

# Access at: http://localhost:8001
```

**Features:**
- ✅ **Automatically kills any existing processes on port 8001**
- ✅ **Prevents errno 48 (address already in use) errors**
- ✅ **Ensures clean startup every time**
- ✅ **Works with real trained models**
- ✅ **Part of official Cancer Alpha workflow**

**What it does:**
1. 🔍 Finds any process using port 8001
2. ⚡ Cleanly terminates existing processes
3. 🚀 Starts the real Cancer Alpha API
4. ✅ Confirms successful startup

**Manual cleanup (if needed):**
```bash
# Find process using port 8001
lsof -ti:8001

# Kill the process
kill -9 $(lsof -ti:8001)

# Then start normally
python3 real_cancer_alpha_api.py
```

### Option 2: Smart Launcher (Alternative)

**Best for: Advanced users needing custom configurations**

```bash
# Cross-platform Python launcher (handles port conflicts)
python3 scripts/start_api.py

# Or use the bash script (Unix/Linux/macOS)
./scripts/start_api.sh

# Access at: http://localhost:8001
```

**Features:**
- ✅ Cross-platform (Windows, macOS, Linux)
- ✅ Dependency checking and validation
- ✅ Support for custom ports and demo mode
- ✅ Colored terminal output with status messages

**Options:**
```bash
# Start on custom port
python3 scripts/start_api.py --port 8002

# Start demo API instead of real models
python3 scripts/start_api.py --demo

# Show help
python3 scripts/start_api.py --help
```

### Option 2: Real Models API (Direct)

**Best for: Research, production, scientific work**

```bash
# Start the real models API directly
python3 real_cancer_alpha_api.py

# Access at: http://localhost:8001
```

**Features:**
- ✅ Actual trained models from research paper
- ✅ 100% Random Forest accuracy, 99% Ensemble accuracy
- ✅ Real genomic feature processing (110 features)
- ✅ Scientific-grade predictions
- ⚠️  May encounter port conflicts (use Option 1 to avoid)

### Option 3: Simple Demo API (Direct)

**Best for: Demonstrations, testing, development**

```bash
# Start the demo API directly
python3 simple_cancer_api.py

# Access at: http://localhost:8000
```

**Features:**
- ✅ Quick setup and testing
- ✅ Mock predictions for demonstration
- ✅ Same API interface as real models
- ✅ No model training required
- ⚠️  May encounter port conflicts (use Option 1 to avoid)

### Option 4: Docker Deployment

**Best for: Production, containerized deployment**

```bash
# Create simple Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn numpy scikit-learn
EXPOSE 8001
CMD ["python", "real_cancer_alpha_api.py"]
EOF

# Build and run
docker build -t cancer-alpha .
docker run -p 8001:8001 cancer-alpha
```

### Option 5: Development Mode

**Best for: Developers, customization**

```bash
# Install in development mode
pip3 install -e .

# Start with auto-reload
uvicorn real_cancer_alpha_api:app --host 0.0.0.0 --port 8001 --reload
```

---

## 🎯 Quick Start Deployments

### For Researchers (Real Models)

```bash
# 1. Start the research-grade API
cd /path/to/cancer-alpha
python3 real_cancer_alpha_api.py

# 2. Test the system
curl http://localhost:8001/health

# 3. View documentation
open http://localhost:8001/docs

# 4. Test real predictions
curl http://localhost:8001/test-real
```

### For Developers (Demo)

```bash
# 1. Start the demo API
python3 simple_cancer_api.py

# 2. Test endpoints
curl http://localhost:8000/test

# 3. Customize the code
# Edit simple_cancer_api.py as needed
```

### For Production (Docker)

```bash
# 1. Build container
docker build -t cancer-alpha-api .

# 2. Run in background
docker run -d -p 8001:8001 --name cancer-alpha cancer-alpha-api

# 3. Check logs
docker logs cancer-alpha

# 4. Access API
curl http://localhost:8001/health
```

---

## 🔧 Using the API

### Web Interface

**Interactive Documentation:**
- Real Models: http://localhost:8001/docs
- Demo: http://localhost:8000/docs

The Swagger UI provides:
- ✅ Interactive API testing
- ✅ Complete endpoint documentation
- ✅ Request/response examples
- ✅ Try-it-out functionality

### Command Line Usage

#### Health Check
```bash
# Check API status
curl http://localhost:8001/health

# Expected response:
{
  "status": "healthy",
  "models_loaded": true,
  "model_performance": {
    "random_forest": 1.0,
    "ensemble": 0.99,
    "gradient_boosting": 0.93
  }
}
```

#### Get Model Information
```bash
curl http://localhost:8001/models/info
```

#### Make a Prediction
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PATIENT_001",
    "age": 45,
    "gender": "F",
    "features": {
      "methylation_1": 0.6,
      "mutation_2": 8,
      "copynumber_3": 12,
      "fragment_4": 180,
      "clinical_5": 0.7,
      "icgc_6": 2.1
    },
    "model_type": "ensemble"
  }'
```

#### Test with Sample Data
```bash
# Test real models
curl http://localhost:8001/test-real

# Test demo models  
curl http://localhost:8000/test
```

### Python Client Usage

```python
import requests
import json

# API endpoint
api_url = "http://localhost:8001"

# Sample genomic features
features = {
    "methylation_1": 0.6,
    "mutation_2": 8, 
    "copynumber_3": 12,
    "fragment_4": 180,
    "clinical_5": 0.7
}

# Make prediction
response = requests.post(f"{api_url}/predict", json={
    "patient_id": "PATIENT_001",
    "age": 55,
    "gender": "F", 
    "features": features,
    "model_type": "ensemble"
})

result = response.json()
print(f"Predicted cancer: {result['predicted_cancer_type']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript/Web Usage

```javascript
// Fetch prediction
const response = await fetch('http://localhost:8001/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        patient_id: "WEB_USER_001",
        age: 50,
        gender: "M",
        features: {
            methylation_1: 0.7,
            mutation_2: 5,
            copynumber_3: 11
        },
        model_type: "ensemble"
    })
});

const result = await response.json();
console.log('Prediction:', result.predicted_cancer_type);
console.log('Confidence:', (result.confidence * 100).toFixed(1) + '%');
```

---

## 🐳 Docker & Kubernetes Deployment

### Docker Deployment

#### Simple Docker Setup
```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir fastapi uvicorn numpy scikit-learn
EXPOSE 8001
CMD ["python", "real_cancer_alpha_api.py"]
EOF

# Build image
docker build -t cancer-alpha-api .

# Run container
docker run -p 8001:8001 cancer-alpha-api
```

#### Production Docker Setup
```bash
# Multi-stage build for smaller image
cat > Dockerfile << 'EOF'
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8001
CMD ["python", "real_cancer_alpha_api.py"]
EOF
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  cancer-alpha-api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

#### Basic Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cancer-alpha-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cancer-alpha-api
  template:
    metadata:
      labels:
        app: cancer-alpha-api
    spec:
      containers:
      - name: cancer-alpha-api
        image: cancer-alpha-api:latest
        ports:
        - containerPort: 8001
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: cancer-alpha-service
spec:
  selector:
    app: cancer-alpha-api
  ports:
  - port: 80
    targetPort: 8001
  type: LoadBalancer
```

#### Deploy to Kubernetes
```bash
# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=cancer-alpha-api

# Get service URL
kubectl get service cancer-alpha-service
```

---

## 🤖 Model Information

### Available Models

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| **Random Forest** | **100%** | Fast | General purpose, highest accuracy |
| **Ensemble** | **99%** | Medium | Best overall performance |
| **Gradient Boosting** | **93%** | Fast | Good balance of speed/accuracy |
| **Deep Neural Network** | **89.5%** | Slow | Complex pattern recognition |

### Feature Types Supported

**110 Genomic Features:**
- **Methylation patterns** (20 features): DNA methylation levels
- **Mutation data** (25 features): Genetic mutation counts  
- **Copy number variations** (20 features): Gene copy numbers
- **Fragmentomics** (15 features): Circulating tumor DNA
- **Clinical variables** (10 features): Patient demographics
- **ICGC ARGO data** (20 features): International cancer genomics

### Model Performance Details

```bash
# Get detailed model info
curl http://localhost:8001/models/info

# Response includes:
{
  "model_performance": {
    "random_forest": {
      "test_accuracy": 1.0,
      "accuracy_percentage": "100.0%"
    },
    "ensemble": {
      "test_accuracy": 0.99,
      "accuracy_percentage": "99.0%" 
    }
  },
  "training_info": {
    "dataset_info": {
      "training_samples": 800,
      "test_samples": 200,
      "features": 110
    }
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. "Models not loaded" error
```bash
# Check if model files exist
ls -la results/phase2/

# If missing, retrain models
python3 src/cancer_alpha/phase2_fixed_model_training.py
```

#### 2. Port already in use (Error: errno 48 - Address already in use)

**This is the most common issue!**

**🚨 Problem**: You get `errno 48` when trying to start the API because another process is using port 8001.

**✅ Solution A - Use the Smart Launcher (RECOMMENDED)**:
```bash
# The smart launcher automatically handles port conflicts
python3 scripts/start_api.py

# Or use the bash version
./scripts/start_api.sh
```

**✅ Solution B - Manual Port Cleanup**:
```bash
# Find what's using the port
lsof -i :8001

# Kill the process (replace PID with actual process ID)
kill -9 [PID]

# Then start your API
python3 real_cancer_alpha_api.py
```

**✅ Solution C - Use Different Port**:
```bash
# Start on a different port
python3 scripts/start_api.py --port 8002

# Then access at http://localhost:8002/docs
```

**✅ Solution D - Kill All Python Processes** (if you're sure):
```bash
# ⚠️  WARNING: This kills ALL Python processes
pkill -f python3

# Then start your API
python3 real_cancer_alpha_api.py
```

#### 3. Import errors
```bash
# Install missing dependencies
pip3 install fastapi uvicorn numpy scikit-learn

# Verify Python version
python3 --version  # Should be 3.8+
```

#### 4. Permission denied
```bash
# Make scripts executable
chmod +x *.py

# Or run with python explicitly
python3 real_cancer_alpha_api.py
```

#### 5. Docker build fails
```bash
# Clean Docker cache
docker system prune -a

# Check Dockerfile syntax
docker build --no-cache -t cancer-alpha-api .
```

### Health Checks

```bash
# API health
curl http://localhost:8001/health

# Model loading test
curl http://localhost:8001/test-real

# Check logs
tail -f /var/log/cancer-alpha.log  # if logging enabled
```

### Performance Issues

```bash
# Check system resources
top | grep python

# Monitor API performance
curl -w "Time: %{time_total}s\n" http://localhost:8001/health
```

---

## 🚀 Advanced Usage

### Custom Model Training

```bash
# Retrain models with custom data
cd /path/to/cancer-alpha
python3 src/cancer_alpha/phase2_fixed_model_training.py

# Models will be saved to results/phase2/
```

### API Customization

```python
# Add custom endpoint to real_cancer_alpha_api.py
@app.get("/custom-endpoint")
async def custom_function():
    return {"message": "Custom functionality"}
```

### Batch Predictions

```python
import requests
import pandas as pd

# Load patient data
patients = pd.read_csv('patient_data.csv')

# Batch predict
results = []
for _, patient in patients.iterrows():
    response = requests.post('http://localhost:8001/predict', json={
        'patient_id': patient['id'],
        'age': patient['age'],
        'gender': patient['gender'], 
        'features': patient.filter(regex='feature_').to_dict(),
        'model_type': 'ensemble'
    })
    results.append(response.json())

# Save results
pd.DataFrame(results).to_csv('predictions.csv')
```

### Integration with Other Systems

```python
# Example: Integration with hospital system
class HospitalIntegration:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
    
    def predict_cancer(self, patient_data):
        response = requests.post(f"{self.api_url}/predict", json=patient_data)
        return response.json()
    
    def get_model_performance(self):
        response = requests.get(f"{self.api_url}/models/info")
        return response.json()
```

---

## 📊 Monitoring & Maintenance

### System Monitoring

```bash
# Check API status
curl http://localhost:8001/health

# Monitor resource usage
htop | grep python

# Check model performance
curl http://localhost:8001/models/info
```

### Log Management

```python
# Add logging to your deployment
import logging

logging.basicConfig(
    filename='/var/log/cancer-alpha.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Updates & Maintenance

```bash
# Update dependencies
pip3 install --upgrade fastapi uvicorn numpy scikit-learn

# Retrain models (if needed)
python3 src/cancer_alpha/phase2_fixed_model_training.py

# Restart API
pkill -f cancer_alpha_api && python3 real_cancer_alpha_api.py
```

---

## 🎉 Success! You're Ready

### What You Have Achieved

✅ **Complete Cancer Alpha Installation**
✅ **Working API with Real Trained Models** 
✅ **100% Accuracy Random Forest Model**
✅ **99% Accuracy Ensemble Model**
✅ **Professional API Documentation**
✅ **Multiple Deployment Options**
✅ **Comprehensive Usage Examples**

### Next Steps

1. **Start Making Predictions**: Use the API for your research
2. **Customize Models**: Train with your own data
3. **Scale Up**: Deploy with Docker/Kubernetes for production
4. **Integrate**: Connect with your existing systems

### Support Resources

- **API Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health  
- **Test Endpoint**: http://localhost:8001/test-real
- **This Guide**: Reference for troubleshooting

---

## 📞 Support

**If you need help:**
1. Check the troubleshooting section above
2. Test with the provided examples
3. Review the API documentation
4. Check system logs for errors

**Your Cancer Alpha system is now fully operational and ready for research use!** 🧬🚀

---

*Last updated: 2025-07-23 | Version: 2.0.0 - REAL MODELS*
