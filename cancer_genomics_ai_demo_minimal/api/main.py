#!/usr/bin/env python3
"""
Production API for Cancer Genomics AI Classifier
================================================

FastAPI backend providing scalable access to the optimized multi-modal transformer
for cancer genomics classification with comprehensive authentication and monitoring.

Author: Cancer Alpha Research Team
Date: July 28, 2025
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import numpy as np
import torch
import joblib
import logging
import time
import uuid
from pathlib import Path
import sys
import os
from datetime import datetime
import redis
from contextlib import asynccontextmanager

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_transformer import MultiModalTransformer, MultiModalConfig
from interpretability.transformer_explainer import TransformerExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global variables for model and scalers
model = None
scalers = None
explainer = None
redis_client = None

class PredictionRequest(BaseModel):
    """Request model for cancer prediction"""
    features: List[float] = Field(..., min_items=110, max_items=110, 
                                 description="110 genomic features in order: methylation(20), mutation(25), cna(20), fragmentomics(15), clinical(10), icgc(20)")
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    include_explanations: bool = Field(False, description="Include SHAP explanations and attention weights")
    include_biological_insights: bool = Field(True, description="Include biological insights")

class PredictionResponse(BaseModel):
    """Response model for cancer prediction"""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    predicted_cancer_type: str = Field(..., description="Predicted cancer type")
    predicted_class: int = Field(..., description="Predicted class index")
    confidence_score: float = Field(..., description="Prediction confidence (0-1)")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for all cancer types")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    explanations: Optional[Dict] = Field(None, description="Model explanations if requested")
    biological_insights: Optional[List[str]] = Field(None, description="Biological insights")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    scalers_loaded: bool
    redis_connected: bool
    uptime_seconds: float
    version: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    samples: List[List[float]] = Field(..., description="List of sample feature arrays")
    patient_ids: Optional[List[str]] = Field(None, description="Optional patient identifiers")
    include_explanations: bool = Field(False, description="Include explanations for all samples")

# Application startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

app = FastAPI(
    title="Cancer Genomics AI Classifier API",
    description="Production API for multi-modal cancer genomics classification using advanced transformer models",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # Add your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.canceralphasolutions.com"]
)

# Startup time for uptime calculation
startup_time = time.time()

async def startup_event():
    """Load models and initialize services on startup"""
    global model, scalers, explainer, redis_client
    
    logger.info("Starting Cancer Genomics AI API...")
    
    try:
        # Load transformer model
        models_dir = Path(__file__).parent.parent / "models"
        checkpoint = torch.load(models_dir / 'optimized_multimodal_transformer.pth', weights_only=False)
        config = checkpoint.get('config', MultiModalConfig())
        model = MultiModalTransformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("✅ Transformer model loaded successfully")
        
        # Load scalers
        scalers = joblib.load(models_dir / 'scalers.pkl')
        logger.info("✅ Data scalers loaded successfully")
        
        # Initialize explainer
        feature_names = [f'feature_{i}' for i in range(110)]  # Simplified for API
        explainer = TransformerExplainer(model, scalers, feature_names)
        logger.info("✅ Model explainer initialized")
        
        # Initialize Redis (optional for caching)
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
            logger.info("✅ Redis connected for caching")
        except:
            logger.warning("⚠️ Redis not available, caching disabled")
            redis_client = None
        
        logger.info("🚀 API startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Cancer Genomics AI API...")
    if redis_client:
        redis_client.close()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key authentication"""
    # In production, validate against a proper authentication system
    valid_keys = ["demo-key-123", "test-key-456"]  # Replace with real authentication
    
    if credentials.credentials not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def preprocess_features(features: List[float]) -> np.ndarray:
    """Preprocess input features using loaded scalers"""
    input_array = np.array(features).reshape(1, -1)
    
    # Apply modality-specific scalers
    methylation = scalers['methylation'].transform(input_array[:, :20])
    mutation = scalers['mutation'].transform(input_array[:, 20:45])
    cna = scalers['cna'].transform(input_array[:, 45:65])
    fragmentomics = scalers['fragmentomics'].transform(input_array[:, 65:80])
    clinical = scalers['clinical'].transform(input_array[:, 80:90])
    icgc = scalers['icgc'].transform(input_array[:, 90:110])
    
    return np.concatenate([methylation, mutation, cna, fragmentomics, clinical, icgc], axis=1)

def make_prediction(processed_features: np.ndarray) -> Dict:
    """Make prediction using the transformer model"""
    with torch.no_grad():
        # Create data dictionary for transformer
        data_dict = {
            'methylation': torch.FloatTensor(processed_features[:, :20]),
            'mutation': torch.FloatTensor(processed_features[:, 20:45]),
            'cna': torch.FloatTensor(processed_features[:, 45:65]),
            'fragmentomics': torch.FloatTensor(processed_features[:, 65:80]),
            'clinical': torch.FloatTensor(processed_features[:, 80:90]),
            'icgc': torch.FloatTensor(processed_features[:, 90:110])
        }
        
        # Forward pass
        outputs = model(data_dict)
        probabilities = outputs['probabilities'].numpy()[0]
        
        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
        predicted_cancer_type = cancer_types[prediction]
        
        class_probabilities = {
            cancer_type: float(prob) for cancer_type, prob in zip(cancer_types, probabilities)
        }
        
        return {
            'predicted_class': prediction,
            'predicted_cancer_type': predicted_cancer_type,
            'confidence_score': confidence,
            'class_probabilities': class_probabilities
        }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        scalers_loaded=scalers is not None,
        redis_connected=redis_client is not None,
        uptime_seconds=uptime,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_cancer(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict cancer type from genomic features
    
    Requires valid API key authentication.
    """
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    try:
        # Validate model is loaded
        if model is None or scalers is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Preprocess features
        processed_features = preprocess_features(request.features)
        
        # Make prediction
        prediction_result = make_prediction(processed_features)
        
        # Generate explanations if requested
        explanations = None
        if request.include_explanations and explainer:
            try:
                # Get attention weights
                attention_weights = explainer.extract_attention_weights(np.array(request.features).reshape(1, -1))
                
                # Get gradient attributions
                attributions = explainer.compute_gradient_attributions(np.array(request.features).reshape(1, -1))
                
                # Modality importance
                modality_importance = explainer.modality_importance_analysis(attributions)
                
                explanations = {
                    'attention_weights': {k: v.tolist() for k, v in attention_weights.items()},
                    'feature_attributions': {k: v.tolist() for k, v in attributions.items()},
                    'modality_importance': modality_importance
                }
            except Exception as e:
                logger.warning(f"Could not generate explanations: {str(e)}")
        
        # Generate biological insights
        biological_insights = None
        if request.include_biological_insights and explainer:
            try:
                attributions = explainer.compute_gradient_attributions(np.array(request.features).reshape(1, -1))
                biological_insights = explainer.generate_biological_insights(
                    attributions, 
                    prediction_result['predicted_class'],
                    prediction_result['confidence_score']
                )
            except Exception as e:
                logger.warning(f"Could not generate biological insights: {str(e)}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Cache result if Redis is available
        if redis_client:
            try:
                cache_key = f"prediction:{prediction_id}"
                redis_client.setex(cache_key, 3600, str(prediction_result))  # Cache for 1 hour
            except:
                pass  # Continue without caching if Redis fails
        
        # Log prediction
        logger.info(f"Prediction {prediction_id}: {prediction_result['predicted_cancer_type']} "
                   f"(confidence: {prediction_result['confidence_score']:.3f}, "
                   f"time: {processing_time:.1f}ms)")
        
        return PredictionResponse(
            prediction_id=prediction_id,
            predicted_cancer_type=prediction_result['predicted_cancer_type'],
            predicted_class=prediction_result['predicted_class'],
            confidence_score=prediction_result['confidence_score'],
            class_probabilities=prediction_result['class_probabilities'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            model_version="optimized_transformer_v1.0",
            explanations=explanations,
            biological_insights=biological_insights
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Batch prediction endpoint for multiple samples
    
    Requires valid API key authentication.
    """
    start_time = time.time()
    
    try:
        if model is None or scalers is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        if len(request.samples) > 100:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size too large (max 100 samples)"
            )
        
        results = []
        
        for i, sample_features in enumerate(request.samples):
            if len(sample_features) != 110:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sample {i} has {len(sample_features)} features, expected 110"
                )
            
            # Process individual sample
            processed_features = preprocess_features(sample_features)
            prediction_result = make_prediction(processed_features)
            
            sample_result = {
                'sample_index': i,
                'patient_id': request.patient_ids[i] if request.patient_ids and i < len(request.patient_ids) else None,
                **prediction_result
            }
            
            results.append(sample_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batch prediction completed: {len(results)} samples in {processing_time:.1f}ms")
        
        return {
            'batch_id': str(uuid.uuid4()),
            'samples_processed': len(results),
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        'model_type': 'MultiModalTransformer',
        'architecture': 'Cross-Modal Attention Transformer',
        'version': 'optimized_v1.0',
        'cancer_types': ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC'],
        'feature_count': 110,
        'modalities': ['methylation', 'mutation', 'cna', 'fragmentomics', 'clinical', 'icgc'],
        'validation_accuracy': 0.725,
        'training_date': '2025-07-28',
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'config': {
            'd_model': model.config.d_model,
            'n_heads': model.config.n_heads,
            'n_layers': model.config.n_layers,
            'dropout': model.config.dropout,
            'use_cross_modal_attention': model.config.use_cross_modal_attention
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        'name': 'Cancer Genomics AI Classifier API',
        'version': '1.0.0',
        'description': 'Production API for multi-modal cancer genomics classification',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'batch_predict': '/predict/batch',
            'model_info': '/model/info',
            'docs': '/docs'
        },
        'documentation': '/docs',
        'status': 'operational'
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
