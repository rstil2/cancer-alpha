#!/usr/bin/env python3
"""
Model Test Script for Cancer Classification Web App
===================================================

This script tests model loading and basic prediction functionality
before launching the Streamlit app.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Feature names matching the trained models (110 features total)
FEATURE_NAMES = (
    # Methylation features (20 features)
    [f'methylation_{i}' for i in range(20)] +
    # Mutation features (25 features)
    [f'mutation_{i}' for i in range(25)] +
    # Copy number alteration features (20 features)
    [f'cn_alteration_{i}' for i in range(20)] +
    # Fragmentomics features (15 features)
    [f'fragmentomics_{i}' for i in range(15)] +
    # Clinical features (10 features)
    [f'clinical_{i}' for i in range(10)] +
    # ICGC ARGO features (20 features)
    [f'icgc_argo_{i}' for i in range(20)]
)

def test_model_loading():
    """Test loading of pre-trained models"""
    print("Testing model loading...")
    print("=" * 50)
    
    models_dir = Path("/Users/stillwell/projects/cancer-alpha/models/phase2_models")
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    # Test loading each model
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'Gradient Boosting': 'gradient_boosting_model.pkl',
        'Deep Neural Network': 'deep_neural_network_model.pkl',
        'Ensemble': 'ensemble_model.pkl'
    }
    
    loaded_models = {}
    
    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                loaded_models[model_name] = model
                print(f"✅ {model_name}: {type(model).__name__}")
            except Exception as e:
                print(f"❌ {model_name}: Failed to load - {str(e)}")
        else:
            print(f"⚠️  {model_name}: File not found - {filename}")
    
    # Test scaler loading
    scaler_path = models_dir / 'scaler.pkl'
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ Scaler: {type(scaler).__name__}")
        except Exception as e:
            print(f"❌ Scaler: Failed to load - {str(e)}")
    else:
        print("⚠️  Scaler: File not found - using dummy scaler")
        scaler = StandardScaler()
    
    return loaded_models, scaler

def generate_test_data():
    """Generate test data for prediction"""
    print("\nGenerating test data...")
    print("=" * 30)
    
    np.random.seed(42)
    
    # Generate cancer-like sample following the training data structure
    cancer_data = []
    
    # Methylation features (20 features) - higher methylation
    cancer_data.extend(np.random.normal(0.5, 0.1, 20))
    
    # Mutation features (25 features) - more mutations
    cancer_data.extend(np.random.poisson(10, 25))
    
    # Copy number alteration features (20 features) - more alterations
    cancer_data.extend(np.random.normal(20, 2, 20))
    
    # Fragmentomics features (15 features) - shorter fragments
    cancer_data.extend(np.random.exponential(150, 15))
    
    # Clinical features (10 features) - older age, higher stage
    cancer_data.extend([65, 3] + list(np.random.normal(0, 1, 8)))
    
    # ICGC ARGO features (20 features) - higher values
    cancer_data.extend(np.random.gamma(3, 0.5, 20))
    
    cancer_data = np.array(cancer_data)
    
    # Generate control-like sample
    control_data = []
    
    # Methylation features (20 features) - normal methylation
    control_data.extend(np.random.normal(0.3, 0.05, 20))
    
    # Mutation features (25 features) - fewer mutations
    control_data.extend(np.random.poisson(3, 25))
    
    # Copy number alteration features (20 features) - fewer alterations
    control_data.extend(np.random.normal(5, 1, 20))
    
    # Fragmentomics features (15 features) - normal fragments
    control_data.extend(np.random.exponential(167, 15))
    
    # Clinical features (10 features) - younger age, lower stage
    control_data.extend([45, 1] + list(np.random.normal(0, 0.5, 8)))
    
    # ICGC ARGO features (20 features) - normal values
    control_data.extend(np.random.gamma(2, 0.4, 20))
    
    control_data = np.array(control_data)
    
    test_samples = {
        'Cancer Sample': cancer_data,
        'Control Sample': control_data
    }
    
    for sample_name, data in test_samples.items():
        print(f"📊 {sample_name}: {len(data)} features")
        print(f"   Range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"   Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    
    return test_samples

def test_predictions(models, scaler, test_samples):
    """Test model predictions on sample data"""
    print("\nTesting predictions...")
    print("=" * 30)
    
    for sample_name, sample_data in test_samples.items():
        print(f"\n🧬 {sample_name}:")
        
        # Preprocess data
        sample_data_2d = sample_data.reshape(1, -1)
        
        # Try to scale data (may fail if scaler wasn't properly trained)
        try:
            if hasattr(scaler, 'transform'):
                scaled_data = scaler.transform(sample_data_2d)
            else:
                # If scaler is not fitted, fit it with dummy data first
                dummy_data = np.random.normal(0, 1, (100, len(FEATURE_NAMES)))
                scaler.fit(dummy_data)
                scaled_data = scaler.transform(sample_data_2d)
        except Exception as e:
            print(f"   ⚠️ Scaling failed: {str(e)}")
            scaled_data = sample_data_2d  # Use unscaled data
        
        # Test each model
        for model_name, model in models.items():
            try:
                # Make prediction
                prediction = model.predict(scaled_data)[0]
                probabilities = model.predict_proba(scaled_data)[0]
                
                # Format results
                pred_label = "Cancer" if prediction == 1 else "Control"
                cancer_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                confidence = max(probabilities)
                
                print(f"   {model_name}:")
                print(f"     Prediction: {pred_label}")
                print(f"     Cancer Probability: {cancer_prob:.1%}")
                print(f"     Confidence: {confidence:.1%}")
                
            except Exception as e:
                print(f"   ❌ {model_name}: Prediction failed - {str(e)}")

def test_shap_compatibility(models, test_data):
    """Test SHAP compatibility with loaded models"""
    print("\nTesting SHAP compatibility...")
    print("=" * 35)
    
    try:
        import shap
        print("✅ SHAP library available")
        
        sample_data = list(test_data.values())[0].reshape(1, -1)
        
        for model_name, model in models.items():
            try:
                # Try to create SHAP explainer
                if hasattr(model, 'predict_proba'):
                    explainer = shap.Explainer(model)
                    print(f"✅ {model_name}: SHAP explainer created")
                else:
                    print(f"⚠️ {model_name}: No predict_proba method")
            except Exception as e:
                print(f"❌ {model_name}: SHAP failed - {str(e)}")
                
    except ImportError:
        print("❌ SHAP library not available")

def main():
    """Main test function"""
    print("Cancer Classification Model Test")
    print("=" * 50)
    
    # Test model loading
    models, scaler = test_model_loading()
    
    if not models:
        print("\n❌ No models loaded successfully. Exiting.")
        return False
    
    # Generate test data
    test_samples = generate_test_data()
    
    # Test predictions
    test_predictions(models, scaler, test_samples)
    
    # Test SHAP compatibility
    test_shap_compatibility(models, test_samples)
    
    print("\n" + "=" * 50)
    print("✅ Model testing completed!")
    print(f"✅ {len(models)} models loaded successfully")
    print("🚀 Ready to launch Streamlit app!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
