#!/usr/bin/env python3
"""
Simple Cancer Alpha API - A beginner-friendly version
====================================================

This is a simplified version of the Cancer Alpha API that's easier to understand
and test. We'll start with this and gradually add more features.

To run this API:
1. Install dependencies: pip install fastapi uvicorn
2. Run the server: python simple_api.py
3. Open your browser to: http://localhost:8000/docs

Author: Cancer Alpha Research Team
Date: July 17, 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
from datetime import datetime

# Create the FastAPI app
app = FastAPI(
    title="Cancer Alpha API - Simple Version",
    description="A simple API for cancer classification",
    version="1.0.0"
)

# Data models (these define what data looks like)
class PredictionRequest(BaseModel):
    """What the user sends to us for prediction"""
    patient_id: str
    age: int
    gender: str
    features: Dict[str, float]  # genomic features

class PredictionResponse(BaseModel):
    """What we send back to the user"""
    patient_id: str
    predicted_cancer_type: str
    confidence: float
    timestamp: str

# Simple mock data for testing
MOCK_PREDICTIONS = {
    "BRCA": 0.85,
    "LUAD": 0.65,
    "COAD": 0.45,
    "PRAD": 0.75
}

@app.get("/")
async def welcome():
    """Welcome message - this is what shows when you go to the root URL"""
    return {
        "message": "Welcome to Cancer Alpha API!",
        "version": "1.0.0",
        "documentation": "Go to /docs to see all available endpoints"
    }

@app.get("/health")
async def health_check():
    """Check if the API is working"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "API is running normally"
    }

@app.get("/cancer-types")
async def get_cancer_types():
    """Get list of cancer types we can predict"""
    return {
        "cancer_types": ["BRCA", "LUAD", "COAD", "PRAD", "STAD", "KIRC", "HNSC", "LIHC"],
        "descriptions": {
            "BRCA": "Breast Invasive Carcinoma",
            "LUAD": "Lung Adenocarcinoma", 
            "COAD": "Colon Adenocarcinoma",
            "PRAD": "Prostate Adenocarcinoma",
            "STAD": "Stomach Adenocarcinoma",
            "KIRC": "Kidney Renal Clear Cell Carcinoma",
            "HNSC": "Head and Neck Squamous Cell Carcinoma",
            "LIHC": "Liver Hepatocellular Carcinoma"
        }
    }

@app.post("/predict")
async def predict_cancer(request: PredictionRequest):
    """Make a cancer prediction - this is our main function"""
    
    # Basic validation
    if not request.patient_id:
        raise HTTPException(status_code=400, detail="Patient ID is required")
    
    if request.age < 0 or request.age > 150:
        raise HTTPException(status_code=400, detail="Invalid age")
    
    if not request.features:
        raise HTTPException(status_code=400, detail="Genomic features are required")
    
    # For now, we'll use a simple rule-based prediction
    # In a real system, this would use our trained models
    
    # Simple logic: if certain features are high, predict certain cancers
    feature_sum = sum(request.features.values())
    
    if feature_sum > 10:
        predicted_type = "BRCA"
        confidence = 0.85
    elif feature_sum > 5:
        predicted_type = "LUAD"
        confidence = 0.72
    elif feature_sum > 2:
        predicted_type = "COAD"
        confidence = 0.68
    else:
        predicted_type = "PRAD"
        confidence = 0.61
    
    # Create response
    response = PredictionResponse(
        patient_id=request.patient_id,
        predicted_cancer_type=predicted_type,
        confidence=confidence,
        timestamp=datetime.now().isoformat()
    )
    
    return response

@app.get("/stats")
async def get_stats():
    """Get some basic statistics about the API"""
    return {
        "total_predictions_made": 42,  # Mock data
        "most_common_prediction": "BRCA",
        "average_confidence": 0.73,
        "api_uptime": "24 hours"
    }

# This is what runs when you execute: python simple_api.py
if __name__ == "__main__":
    import uvicorn
    print("Starting Cancer Alpha API...")
    print("Once started, go to: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
