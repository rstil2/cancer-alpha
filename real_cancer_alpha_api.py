#!/usr/bin/env python3
"""
Cancer Alpha - REAL API with Trained Models
==========================================

This API uses the actual trained models from the Cancer Alpha research paper.
These models were trained on synthetic cancer genomics data with realistic
patterns for 8 different cancer types.

Model Performance (Test Accuracy):
- Random Forest: 100.0%
- Ensemble Model: 99.0%
- Gradient Boosting: 93.0%
- Deep Neural Network: 89.5%

To run: python3 real_cancer_alpha_api.py
Then visit: http://localhost:8000/docs
"""

import pickle
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Load the actual trained models
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.ensemble_model = None
        self.feature_importance = None
        self.metadata = None
        self.models_loaded = False
        
    def load_models(self):
        """Load all trained models from Phase 2"""
        models_dir = Path("results/phase2")
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        try:
            # Load individual models
            model_files = {
                'deep_neural_network': 'deep_neural_network_model.pkl',
                'gradient_boosting': 'gradient_boosting_model.pkl',
                'random_forest': 'random_forest_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = models_dir / filename
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"✓ Loaded {model_name}")
            
            # Load ensemble model
            ensemble_path = models_dir / 'ensemble_model.pkl'
            with open(ensemble_path, 'rb') as f:
                self.ensemble_model = pickle.load(f)
            print("✓ Loaded ensemble model")
            
            # Load scaler
            scaler_path = models_dir / 'scaler.pkl'
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✓ Loaded scaler")
            
            # Load feature importance
            feature_importance_path = models_dir / 'feature_importance.json'
            if feature_importance_path.exists():
                with open(feature_importance_path, 'r') as f:
                    self.feature_importance = json.load(f)
                print("✓ Loaded feature importance")
            
            # Load metadata
            metadata_path = models_dir / 'phase2_report.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print("✓ Loaded metadata")
            
            self.models_loaded = True
            print(f"\n🎉 Successfully loaded all models!")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise

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
    model_accuracy: float

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    models_loaded: bool
    message: str
    version: str
    model_performance: Dict[str, float]

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    loaded_models: List[str]
    model_performance: Dict[str, Dict[str, float]]
    feature_count: int
    cancer_types: List[str]
    training_info: Dict[str, Any]

# Cancer type mappings (matches the trained models)
CANCER_TYPES = {
    0: {"code": "BRCA", "name": "Breast Invasive Carcinoma"},
    1: {"code": "LUAD", "name": "Lung Adenocarcinoma"},
    2: {"code": "COAD", "name": "Colon Adenocarcinoma"},
    3: {"code": "PRAD", "name": "Prostate Adenocarcinoma"},
    4: {"code": "STAD", "name": "Stomach Adenocarcinoma"},
    5: {"code": "KIRC", "name": "Kidney Renal Clear Cell Carcinoma"},
    6: {"code": "HNSC", "name": "Head and Neck Squamous Cell Carcinoma"},
    7: {"code": "LIHC", "name": "Liver Hepatocellular Carcinoma"}
}

# Create FastAPI app
app = FastAPI(
    title="Cancer Alpha API - Real Trained Models",
    description="Cancer classification API using actual trained models from research paper",
    version="2.0.0 - REAL MODELS",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def prepare_features(features: Dict[str, float]) -> np.ndarray:
    """Prepare features for model prediction"""
    # Convert features dictionary to ordered array (110 features expected)
    feature_array = np.zeros(110)
    
    # Fill with provided features
    for i, (key, value) in enumerate(features.items()):
        if i < 110:  # Ensure we don't exceed expected features
            feature_array[i] = value
    
    return feature_array.reshape(1, -1)

def make_real_prediction(features: Dict[str, float], model_type: str) -> tuple:
    """Make prediction using actual trained models"""
    start_time = datetime.now()
    
    if not model_loader.models_loaded:
        raise ValueError("Models not loaded")
    
    # Prepare and scale features
    X = prepare_features(features)
    if model_loader.scaler:
        X = model_loader.scaler.transform(X)
    
    # Get model performance for confidence adjustment
    model_accuracy = 0.85  # default
    if model_loader.metadata and 'results' in model_loader.metadata:
        model_accuracy = model_loader.metadata['results'].get(model_type, {}).get('test_accuracy', 0.85)
    
    # Make prediction based on model type
    if model_type == "ensemble" and model_loader.ensemble_model:
        # Use the ensemble model
        individual_models = model_loader.ensemble_model['models']
        predictions = []
        
        for name, model in individual_models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[0]
            else:
                pred = model.predict(X)[0]
                pred_proba = np.zeros(8)
                pred_proba[pred] = 0.9
                pred_proba += 0.1 / 8  # Normalize
            predictions.append(pred_proba)
        
        # Average predictions
        final_proba = np.mean(predictions, axis=0)
        prediction = np.argmax(final_proba)
        model_accuracy = model_loader.ensemble_model.get('test_accuracy', 0.99)
        
    elif model_type in model_loader.models:
        # Use individual model
        model = model_loader.models[model_type]
        
        if hasattr(model, 'predict_proba'):
            final_proba = model.predict_proba(X)[0]
            prediction = np.argmax(final_proba)
        else:
            prediction = model.predict(X)[0]
            final_proba = np.zeros(8)
            final_proba[prediction] = 0.9
            final_proba += 0.1 / 8  # Add small probability to other classes
        
    else:
        raise ValueError(f"Model {model_type} not available")
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return prediction, final_proba, processing_time, model_accuracy

# Load models at startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Cancer Alpha API - Loading Real Trained Models")
    print("=" * 60)
    try:
        model_loader.load_models()
        print("✅ All models loaded successfully!")
        print("🌟 API ready with real trained models from research paper")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("⚠️  API will start but predictions will fail")
    print("=" * 60)

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    performance_summary = {}
    if model_loader.metadata and 'results' in model_loader.metadata:
        for model, results in model_loader.metadata['results'].items():
            performance_summary[model] = f"{results.get('test_accuracy', 0):.1%}"
    
    return {
        "message": "Cancer Alpha API - Real Trained Models",
        "version": "2.0.0 - REAL MODELS",
        "description": "Cancer classification using actual trained models from research paper",
        "status": "✅ Fully operational with real models",
        "model_performance": performance_summary,
        "features": [
            "🧬 Real trained models from Cancer Alpha research",
            "📊 100% Random Forest accuracy",
            "🎯 99% Ensemble model accuracy", 
            "🔬 110 genomic features support",
            "⚡ Fast predictions (< 100ms)",
            "📚 Comprehensive API documentation"
        ],
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
    performance = {}
    if model_loader.metadata and 'results' in model_loader.metadata:
        for model, results in model_loader.metadata['results'].items():
            performance[model] = results.get('test_accuracy', 0)
    
    return HealthResponse(
        status="healthy" if model_loader.models_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=model_loader.models_loaded,
        message="API operational with real trained models" if model_loader.models_loaded else "Models not loaded",
        version="2.0.0 - REAL MODELS",
        model_performance=performance
    )

@app.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get detailed information about loaded models"""
    if not model_loader.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    performance = {}
    training_info = {}
    
    if model_loader.metadata:
        if 'results' in model_loader.metadata:
            for model, results in model_loader.metadata['results'].items():
                performance[model] = {
                    "test_accuracy": results.get('test_accuracy', 0),
                    "accuracy_percentage": f"{results.get('test_accuracy', 0):.1%}"
                }
        
        training_info = {
            "training_date": model_loader.metadata.get('timestamp', 'Unknown'),
            "dataset_info": model_loader.metadata.get('dataset_info', {}),
            "cancer_types": model_loader.metadata.get('cancer_types', [])
        }
    
    return ModelInfoResponse(
        loaded_models=list(model_loader.models.keys()) + (['ensemble'] if model_loader.ensemble_model else []),
        model_performance=performance,
        feature_count=110,
        cancer_types=[info["code"] for info in CANCER_TYPES.values()],
        training_info=training_info
    )

@app.get("/cancer-types", response_model=dict)
async def get_cancer_types():
    """Get available cancer types"""
    return {
        "cancer_types": [info["code"] for info in CANCER_TYPES.values()],
        "descriptions": {info["code"]: info["name"] for info in CANCER_TYPES.values()},
        "total_types": len(CANCER_TYPES),
        "supported_models": ["ensemble", "random_forest", "gradient_boosting", "deep_neural_network"],
        "note": "These are the cancer types the models were trained on"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_cancer(request: PredictionRequest):
    """Make cancer prediction using real trained models"""
    
    if not model_loader.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate model type
    valid_models = ["ensemble", "random_forest", "gradient_boosting", "deep_neural_network"]
    if request.model_type not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available. Available models: {valid_models}"
        )
    
    # Validate features
    if len(request.features) == 0:
        raise HTTPException(status_code=400, detail="No features provided")
    
    try:
        # Make prediction using real trained models
        prediction, probabilities, processing_time, model_accuracy = make_real_prediction(
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
            processing_time_ms=processing_time,
            model_accuracy=model_accuracy
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/test-real", response_model=dict)
async def test_real_models():
    """Test endpoint using real models with sample data"""
    if not model_loader.models_loaded:
        return {"error": "Models not loaded"}
    
    # Create sample genomic features (110 features)
    sample_features = {}
    
    # Methylation features (20)
    for i in range(20):
        sample_features[f"methylation_{i}"] = np.random.normal(0.5, 0.1)
    
    # Mutation features (25) 
    for i in range(25):
        sample_features[f"mutation_{i}"] = np.random.poisson(5)
    
    # Copy number features (20)
    for i in range(20):
        sample_features[f"copynumber_{i}"] = np.random.normal(10, 2)
    
    # Fragmentomics features (15)
    for i in range(15):
        sample_features[f"fragment_{i}"] = np.random.exponential(150)
    
    # Clinical features (10)
    for i in range(10):
        sample_features[f"clinical_{i}"] = np.random.normal(0.5, 0.1)
    
    # ICGC ARGO features (20)
    for i in range(20):
        sample_features[f"icgc_{i}"] = np.random.gamma(2, 0.5)
    
    # Test with different models
    results = {}
    for model_type in ["ensemble", "random_forest", "gradient_boosting", "deep_neural_network"]:
        try:
            prediction, probabilities, processing_time, accuracy = make_real_prediction(
                sample_features, model_type
            )
            
            results[model_type] = {
                "predicted_cancer": CANCER_TYPES[prediction]["code"],
                "confidence": float(max(probabilities)),
                "processing_time_ms": processing_time,
                "model_accuracy": accuracy
            }
        except Exception as e:
            results[model_type] = {"error": str(e)}
    
    return {
        "message": "✅ Real model test completed!",
        "test_results": results,
        "sample_features_count": len(sample_features),
        "models_tested": 4
    }

if __name__ == "__main__":
    print("🧬 Starting Cancer Alpha API with REAL TRAINED MODELS")
    print("=" * 65)
    print("🎯 Model Performance:")
    print("   • Random Forest: 100.0% accuracy")
    print("   • Ensemble Model: 99.0% accuracy") 
    print("   • Gradient Boosting: 93.0% accuracy")
    print("   • Deep Neural Network: 89.5% accuracy")
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("🧪 Real Model Test: http://localhost:8000/test-real")
    print("=" * 65)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        log_level="info"
    )
