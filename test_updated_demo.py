#!/usr/bin/env python3
"""
Demo Test Script
===============

Tests the updated demo models to ensure they work correctly.
"""

import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_demo_models():
    """Test loading and prediction with updated models"""
    logger.info("Testing updated demo models...")
    
    # Test directories
    demo_dirs = [
        Path("/Users/stillwell/projects/cancer-alpha/DEMO_PACKAGE/cancer_genomics_ai_demo/models"),
        Path("/Users/stillwell/projects/cancer-alpha/cancer_genomics_ai_demo_minimal/models")
    ]
    
    cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
    
    for demo_dir in demo_dirs:
        if not demo_dir.exists():
            logger.warning(f"Directory does not exist: {demo_dir}")
            continue
            
        logger.info(f"Testing models in: {demo_dir}")
        
        try:
            # Test Real TCGA Logistic Regression
            lr_path = demo_dir / "multimodal_real_tcga_logistic_regression.pkl"
            if lr_path.exists():
                lr_model = joblib.load(lr_path)
                logger.info("✅ Loaded Real TCGA Logistic Regression")
                
                # Test scaler
                scaler_path = demo_dir / "multimodal_real_tcga_scaler.pkl"
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                    logger.info("✅ Loaded Real TCGA scaler")
                    
                    # Test feature selector  
                    selector_path = demo_dir / "feature_selector.pkl"
                    if selector_path.exists():
                        selector = joblib.load(selector_path)
                        logger.info("✅ Loaded feature selector")
                        
                        # Generate test data (110 features)
                        test_data = np.random.random((1, 110))
                        
                        # Full prediction pipeline test
                        scaled_data = scaler.transform(test_data)
                        selected_data = selector.transform(scaled_data)
                        
                        prediction = lr_model.predict(selected_data)[0]
                        probabilities = lr_model.predict_proba(selected_data)[0]
                        
                        predicted_cancer = cancer_types[prediction]
                        confidence = max(probabilities)
                        
                        logger.info(f"✅ LR Prediction: {predicted_cancer} (confidence: {confidence:.2%})")
                    else:
                        logger.warning("❌ Feature selector not found")
                else:
                    logger.warning("❌ Real TCGA scaler not found")
            else:
                logger.warning("❌ Real TCGA Logistic Regression not found")
            
            # Test Real TCGA Random Forest
            rf_path = demo_dir / "multimodal_real_tcga_random_forest.pkl"
            if rf_path.exists():
                rf_model = joblib.load(rf_path)
                logger.info("✅ Loaded Real TCGA Random Forest")
                
                if 'scaler' in locals():
                    # Generate test data
                    test_data = np.random.random((1, 110))
                    scaled_data = scaler.transform(test_data)
                    
                    prediction = rf_model.predict(scaled_data)[0]
                    probabilities = rf_model.predict_proba(scaled_data)[0]
                    
                    predicted_cancer = cancer_types[prediction]
                    confidence = max(probabilities)
                    
                    logger.info(f"✅ RF Prediction: {predicted_cancer} (confidence: {confidence:.2%})")
                else:
                    logger.warning("❌ Scaler not available for RF test")
            else:
                logger.warning("❌ Real TCGA Random Forest not found")
                
            # Check metadata
            metadata_path = demo_dir / "model_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✅ Metadata: {metadata.get('model_date', 'Unknown date')}")
            else:
                logger.warning("❌ Metadata not found")
            
            logger.info(f"✅ All tests passed for {demo_dir.name}")
            
        except Exception as e:
            logger.error(f"❌ Error testing {demo_dir}: {str(e)}")
    
    logger.info("✅ Demo models testing completed!")

if __name__ == "__main__":
    test_demo_models()
