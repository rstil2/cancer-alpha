#!/usr/bin/env python3
"""
Cancer Alpha Real Model API - Phase 4A Implementation
====================================================

This version integrates the actual trained models from Phase 2 into the API.
We load the real models and use them for predictions instead of mock data.

To run this API:
1. Install dependencies: pip install fastapi uvicorn scikit-learn pandas numpy
2. Run the server: python real_model_api.py
3. Open your browser to: http://localhost:8000/docs

Author: Cancer Alpha Research Team
Date: July 18, 2025
"""

import os
import sys
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Add the project root to the path so we can import from src
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Print debug info
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Models directory: {project_root / 'results' / 'phase2'}")

# Model loading utilities
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_importance = None
        self.model_metadata = None
        
    def load_models(self):
        """Load all trained models from Phase 2"""
        models_dir = project_root / "results" / "phase2"
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
            
        print(f"Loading models from: {models_dir}")
        
        try:
            # Load individual models
            model_files = {
                'deep_neural_network': 'deep_neural_network_model.pkl',
                'gradient_boosting': 'gradient_boosting_model.pkl', 
                'random_forest': 'random_forest_model.pkl',
                'ensemble': 'ensemble_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = models_dir / filename
                if model_path.exists():
                    print(f"Loading {model_name}...")
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✓ {model_name} loaded successfully")
                else:
                    print(f"⚠ Warning: {filename} not found")
            
            # Load scaler
            scaler_path = models_dir / 'scaler.pkl'
            if scaler_path.exists():
                print("Loading scaler...")
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✓ Scaler loaded successfully")
            
            # Load feature importance
            feature_importance_path = models_dir / 'feature_importance.json'
            if feature_importance_path.exists():
                print("Loading feature importance...")
                with open(feature_importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
                print("✓ Feature importance loaded successfully")
            
            # Load model metadata
            report_path = models_dir / 'phase2_report.json'
            if report_path.exists():
                print("Loading model metadata...")
                with open(report_path, 'r') as f:
                    self.model_metadata = json.load(f)
                print("✓ Model metadata loaded successfully")
            
            print(f"\nSuccessfully loaded {len(self.models)} models!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'loaded_models': list(self.models.keys()),
            'scaler_loaded': self.scaler is not None,
            'feature_importance_loaded': self.feature_importance is not None,
            'metadata_loaded': self.model_metadata is not None
        }
        
        if self.model_metadata:
            info['model_performance'] = {
                name: {
                    'test_accuracy': self.model_metadata['results'][name].get('test_accuracy', 'N/A'),
                    'cv_mean': self.model_metadata['results'][name].get('cv_mean', 'N/A')
                }
                for name in self.models.keys() if name in self.model_metadata['results']
            }
        
        return info

# Initialize model loader
model_loader = ModelLoader()

# API Models
class PredictionRequest(BaseModel):
    """Request model for cancer prediction"""
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=150, description="Patient age")
    gender: str = Field(..., description="Patient gender (M/F)")
    features: Dict[str, float] = Field(..., description="Genomic features (110 features expected)")
    model_type: Optional[str] = Field("ensemble", description="Model type: ensemble, random_forest, gradient_boosting, deep_neural_network")

class PredictionResponse(BaseModel):
    """Response model for cancer prediction"""
    patient_id: str
    predicted_cancer_type: str
    predicted_cancer_name: str
    confidence: float
    probability_distribution: Dict[str, float]
    model_used: str
    timestamp: str
    processing_time_ms: float

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    loaded_models: List[str]
    scaler_loaded: bool
    feature_importance_loaded: bool
    metadata_loaded: bool
    model_performance: Optional[Dict[str, Dict[str, Any]]] = None
    feature_count: Optional[int] = None

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    models_loaded: bool
    message: str

# Cancer type mappings
CANCER_TYPES = {
    0: {"code": "BRCA", "name": "Breast Invasive Carcinoma"},
    1: {"code": "COAD", "name": "Colon Adenocarcinoma"},
    2: {"code": "HNSC", "name": "Head and Neck Squamous Cell Carcinoma"},
    3: {"code": "KIRC", "name": "Kidney Renal Clear Cell Carcinoma"},
    4: {"code": "LIHC", "name": "Liver Hepatocellular Carcinoma"},
    5: {"code": "LUAD", "name": "Lung Adenocarcinoma"},
    6: {"code": "PRAD", "name": "Prostate Adenocarcinoma"},
    7: {"code": "STAD", "name": "Stomach Adenocarcinoma"}
}

# Lifespan event manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n" + "="*60)
    print("🚀 Cancer Alpha API - Phase 4A Starting Up")
    print("="*60)
    
    try:
        model_loader.load_models()
        print("\n✅ All models loaded successfully!")
        print("📊 Model Performance Summary:")
        
        if model_loader.model_metadata:
            for model_name, results in model_loader.model_metadata['results'].items():
                if model_name in model_loader.models:
                    test_acc = results.get('test_accuracy', 'N/A')
                    cv_mean = results.get('cv_mean', 'N/A')
                    print(f"   • {model_name}: Test Accuracy = {test_acc}, CV Mean = {cv_mean}")
        
        print("\n🌐 API is ready to serve predictions!")
        print("   • Swagger UI: http://localhost:8000/docs")
        print("   • ReDoc: http://localhost:8000/redoc")
        
    except Exception as e:
        print(f"\n❌ Failed to load models: {str(e)}")
        print("⚠️  API will start but predictions will fail")
    
    print("="*60)
    
    yield
    
    # Shutdown
    print("\n🔄 API shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Cancer Alpha API - Real Models (Phase 4A)",
    description="Cancer classification API using real trained models from Phase 2",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for model loader
def get_model_loader():
    return model_loader

# Prediction utilities
def prepare_features(features: Dict[str, float]) -> np.ndarray:
    """Prepare features for model prediction"""
    # We expect 110 features (feature_0 to feature_109)
    expected_features = [f"feature_{i}" for i in range(110)]
    
    # Create feature array
    feature_array = np.zeros(110)
    
    for i, feature_name in enumerate(expected_features):
        if feature_name in features:
            feature_array[i] = features[feature_name]
        else:
            # Fill missing features with 0 or could use mean imputation
            feature_array[i] = 0.0
    
    return feature_array.reshape(1, -1)

def make_prediction(features: Dict[str, float], model_type: str) -> tuple:
    """Make prediction using specified model"""
    start_time = datetime.now()
    
    # Prepare features
    X = prepare_features(features)
    
    # Scale features if scaler is available
    if model_loader.scaler is not None:
        X = model_loader.scaler.transform(X)
    
    # Get model
    if model_type not in model_loader.models:
        raise ValueError(f"Model {model_type} not available. Available models: {list(model_loader.models.keys())}")
    
    model = model_loader.models[model_type]
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
    else:
        # For models without predict_proba, create mock probabilities
        probabilities = np.zeros(len(CANCER_TYPES))
        probabilities[prediction] = 0.85  # High confidence for predicted class
        # Distribute remaining probability among other classes
        remaining_prob = 0.15
        for i in range(len(CANCER_TYPES)):
            if i != prediction:
                probabilities[i] = remaining_prob / (len(CANCER_TYPES) - 1)
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return prediction, probabilities, processing_time

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cancer Alpha API - Real Models (Phase 4A)",
        "version": "2.0.0",
        "description": "Cancer classification using real trained models from Phase 2",
        "endpoints": {
            "prediction": "/predict",
            "health": "/health", 
            "models": "/models/info",
            "cancer_types": "/cancer-types",
            "documentation": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = len(model_loader.models) > 0
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        message="API is running with real models" if models_loaded else "Models not loaded"
    )

@app.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded models"""
    info = model_loader.get_model_info()
    
    return ModelInfoResponse(
        loaded_models=info['loaded_models'],
        scaler_loaded=info['scaler_loaded'],
        feature_importance_loaded=info['feature_importance_loaded'],
        metadata_loaded=info['metadata_loaded'],
        model_performance=info.get('model_performance'),
        feature_count=110 if model_loader.feature_importance else None
    )

@app.get("/cancer-types", response_model=dict)
async def get_cancer_types():
    """Get available cancer types"""
    return {
        "cancer_types": [info["code"] for info in CANCER_TYPES.values()],
        "descriptions": {info["code"]: info["name"] for info in CANCER_TYPES.values()},
        "total_types": len(CANCER_TYPES)
    }

@app.get("/models/feature-importance", response_model=dict)
async def get_feature_importance():
    """Get feature importance from trained models"""
    if not model_loader.feature_importance:
        raise HTTPException(status_code=404, detail="Feature importance not available")
    
    # Get top 20 most important features
    sorted_features = sorted(
        model_loader.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    
    return {
        "top_features": {feature: importance for feature, importance in sorted_features},
        "total_features": len(model_loader.feature_importance),
        "importance_scale": "0.0 to 1.0 (higher is more important)"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_cancer(request: PredictionRequest):
    """Make cancer prediction using real trained models"""
    
    # Validate model type
    if request.model_type not in model_loader.models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model_type}' not available. Available models: {list(model_loader.models.keys())}"
        )
    
    # Check if we have enough features
    if len(request.features) == 0:
        raise HTTPException(status_code=400, detail="No features provided")
    
    try:
        # Make prediction
        prediction, probabilities, processing_time = make_prediction(
            request.features, request.model_type
        )
        
        # Get cancer type info
        cancer_info = CANCER_TYPES[prediction]
        
        # Create probability distribution
        prob_dist = {
            CANCER_TYPES[i]["code"]: float(probabilities[i]) 
            for i in range(len(CANCER_TYPES))
        }
        
        # Get confidence (max probability)
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            patient_id=request.patient_id,
            predicted_cancer_type=cancer_info["code"],
            predicted_cancer_name=cancer_info["name"],
            confidence=confidence,
            probability_distribution=prob_dist,
            model_used=request.model_type,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    print("Starting Cancer Alpha API with Real Models...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )

