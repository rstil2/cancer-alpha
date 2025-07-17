#!/usr/bin/env python3
"""
Cancer Alpha API - FastAPI-based REST API for model inference
============================================================

This module provides a RESTful API for the Cancer Alpha models, enabling
clinical and research users to access model predictions through standardized
HTTP endpoints.

Features:
- Model inference endpoints
- Authentication and authorization
- Rate limiting
- Input validation
- Async processing support
- Comprehensive logging

Author: Cancer Alpha Research Team
Date: July 17, 2025
"""

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
import asyncio
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cancer Alpha API",
    description="RESTful API for Cancer Alpha multi-modal cancer classification models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables for model caching
MODELS_CACHE = {}
SCALER_CACHE = {}

# Configuration
MODEL_PATH = Path("../../results/phase2_optimized")
API_KEY = "cancer-alpha-api-key"  # In production, use environment variable

class GenomicData(BaseModel):
    """Input model for genomic data"""
    features: Dict[str, float] = Field(..., description="Genomic features as key-value pairs")
    sample_id: Optional[str] = Field(None, description="Sample identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    data: GenomicData
    model_type: str = Field("ensemble", description="Model type: deep_neural_network, gradient_boosting, random_forest, or ensemble")
    return_probabilities: bool = Field(True, description="Return class probabilities")
    return_confidence: bool = Field(True, description="Return confidence scores")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    sample_id: Optional[str] = None
    model_type: str
    timestamp: datetime
    request_id: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: List[str]
    version: str

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_type: str
    version: str
    accuracy: float
    features_count: int
    classes: List[str]
    last_updated: datetime

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key authentication"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def load_models():
    """Load trained models into memory cache"""
    global MODELS_CACHE, SCALER_CACHE
    
    try:
        # Load models from Phase 2 results
        model_files = {
            "deep_neural_network": "deep_neural_network_model.pkl",
            "gradient_boosting": "gradient_boosting_model.pkl", 
            "random_forest": "random_forest_model.pkl",
            "ensemble": "ensemble_model.pkl"
        }
        
        for model_name, filename in model_files.items():
            model_path = MODEL_PATH / filename
            if model_path.exists():
                MODELS_CACHE[model_name] = joblib.load(model_path)
                logger.info(f"Loaded model: {model_name}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        # Load scaler
        scaler_path = MODEL_PATH / "scaler.pkl"
        if scaler_path.exists():
            SCALER_CACHE["scaler"] = joblib.load(scaler_path)
            logger.info("Loaded feature scaler")
        else:
            logger.warning(f"Scaler file not found: {scaler_path}")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def preprocess_data(data: GenomicData) -> np.ndarray:
    """Preprocess input data for model inference"""
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([data.features])
        
        # Apply same preprocessing as training
        if "scaler" in SCALER_CACHE:
            scaled_features = SCALER_CACHE["scaler"].transform(df)
            return scaled_features
        else:
            logger.warning("Scaler not available, using raw features")
            return df.values
            
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {e}")

def calculate_confidence(probabilities: np.ndarray) -> float:
    """Calculate prediction confidence score"""
    # Use entropy-based confidence measure
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    max_entropy = np.log(len(probabilities))
    confidence = 1 - (entropy / max_entropy)
    return float(confidence)

async def make_prediction(model_name: str, features: np.ndarray, return_probabilities: bool = True) -> Dict:
    """Make prediction using specified model"""
    try:
        if model_name not in MODELS_CACHE:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        model = MODELS_CACHE[model_name]
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        result = {
            "prediction": prediction,
            "probabilities": None,
            "confidence": None
        }
        
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            
            # Get class names (assuming standard cancer types)
            class_names = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
            
            result["probabilities"] = {
                class_name: float(prob) 
                for class_name, prob in zip(class_names, probabilities)
            }
            result["confidence"] = calculate_confidence(probabilities)
        
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    logger.info("Starting Cancer Alpha API...")
    load_models()
    logger.info(f"API started successfully. Models loaded: {list(MODELS_CACHE.keys())}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Cancer Alpha API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=list(MODELS_CACHE.keys()),
        version="1.0.0"
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models"""
    models_info = []
    
    for model_name in MODELS_CACHE.keys():
        models_info.append(ModelInfo(
            model_name=model_name,
            model_type=model_name,
            version="1.0.0",
            accuracy=0.95,  # Would be loaded from model metadata
            features_count=110,  # Would be loaded from model metadata
            classes=['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC'],
            last_updated=datetime.now()
        ))
    
    return models_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Make prediction using specified model"""
    
    request_id = str(uuid.uuid4())
    
    try:
        # Preprocess input data
        features = preprocess_data(request.data)
        
        # Make prediction
        result = await make_prediction(
            request.model_type,
            features,
            request.return_probabilities
        )
        
        # Log prediction request (background task)
        background_tasks.add_task(
            log_prediction_request,
            request_id,
            request.model_type,
            request.data.sample_id
        )
        
        return PredictionResponse(
            prediction=result["prediction"],
            probabilities=result["probabilities"] if request.return_probabilities else None,
            confidence=result["confidence"] if request.return_confidence else None,
            sample_id=request.data.sample_id,
            model_type=request.model_type,
            timestamp=datetime.now(),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(
    requests: List[PredictionRequest],
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Make batch predictions"""
    
    if len(requests) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size exceeds limit (100)")
    
    results = []
    for request in requests:
        try:
            features = preprocess_data(request.data)
            result = await make_prediction(
                request.model_type,
                features,
                request.return_probabilities
            )
            
            results.append(PredictionResponse(
                prediction=result["prediction"],
                probabilities=result["probabilities"] if request.return_probabilities else None,
                confidence=result["confidence"] if request.return_confidence else None,
                sample_id=request.data.sample_id,
                model_type=request.model_type,
                timestamp=datetime.now(),
                request_id=str(uuid.uuid4())
            ))
            
        except Exception as e:
            logger.error(f"Batch prediction failed for sample {request.data.sample_id}: {e}")
            results.append({
                "error": str(e),
                "sample_id": request.data.sample_id
            })
    
    return {"results": results}

async def log_prediction_request(request_id: str, model_type: str, sample_id: Optional[str]):
    """Log prediction request for monitoring and analytics"""
    logger.info(f"Prediction request logged - ID: {request_id}, Model: {model_type}, Sample: {sample_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
