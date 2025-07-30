#!/usr/bin/env python3
"""
Generate Demo Data and Models for Cancer Alpha Lightweight Package

This script generates the necessary demo data and models for the Cancer Alpha
demo when they are not present. It creates synthetic genomic data and trains
basic models for demonstration purposes.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_SAMPLES = 2000
N_FEATURES = 110
N_CANCER_TYPES = 8
RANDOM_STATE = 42

CANCER_TYPES = [
    'BRCA', 'LUAD', 'COAD', 'STAD', 'BLCA', 'LIHC', 'CESC', 'KIRP'
]

def create_directories():
    """Create necessary directories for models and data."""
    directories = ['models', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

def generate_synthetic_genomic_data():
    """Generate synthetic genomic data with realistic cancer patterns."""
    print("🧬 Generating synthetic genomic data...")
    
    np.random.seed(RANDOM_STATE)
    
    # Generate base features with different distributions for different modalities
    # Methylation features (0-30): Beta values between 0 and 1
    methylation_features = np.random.beta(2, 2, (N_SAMPLES, 30))
    
    # Copy number alterations (30-60): Log2 ratios, mostly around 0
    cna_features = np.random.normal(0, 0.5, (N_SAMPLES, 30))
    
    # Mutation features (60-80): Binary/count data
    mutation_features = np.random.poisson(0.3, (N_SAMPLES, 20))
    
    # Fragmentomics features (80-100): Continuous values
    fragmentomics_features = np.random.gamma(2, 0.5, (N_SAMPLES, 20))
    
    # Clinical features (100-110): Mixed continuous and categorical
    clinical_features = np.random.randn(N_SAMPLES, 10)
    
    # Combine all features
    X = np.concatenate([
        methylation_features,
        cna_features, 
        mutation_features,
        fragmentomics_features,
        clinical_features
    ], axis=1)
    
    # Generate labels with realistic cancer type distribution
    y = np.random.choice(len(CANCER_TYPES), N_SAMPLES, 
                        p=[0.25, 0.15, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08])
    
    # Add some cancer-type specific patterns to make it more realistic
    for i, cancer_type in enumerate(CANCER_TYPES):
        mask = (y == i)
        if np.sum(mask) > 0:
            # Add cancer-specific patterns
            X[mask, i*5:(i+1)*5] += np.random.normal(0.5, 0.2, (np.sum(mask), 5))
    
    # Create feature names
    feature_names = []
    feature_names.extend([f'methylation_feature_{i}' for i in range(30)])
    feature_names.extend([f'cna_feature_{i}' for i in range(30)])
    feature_names.extend([f'mutation_feature_{i}' for i in range(20)])
    feature_names.extend([f'fragmentomics_feature_{i}' for i in range(20)])
    feature_names.extend([f'clinical_feature_{i}' for i in range(10)])
    
    return X, y, feature_names

def train_models(X, y):
    """Train demo models on the synthetic data."""
    print("🤖 Training demo models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    # Random Forest
    print("  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_STATE,
        max_depth=10,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    models['random_forest'] = rf_model
    print(f"    Random Forest Accuracy: {rf_accuracy:.3f}")
    
    # Gradient Boosting
    print("  Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        max_depth=6
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    models['gradient_boosting'] = gb_model
    print(f"    Gradient Boosting Accuracy: {gb_accuracy:.3f}")
    
    # Neural Network
    print("  Training Deep Neural Network...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        random_state=RANDOM_STATE,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_model.fit(X_train_scaled, y_train)
    nn_pred = nn_model.predict(X_test_scaled)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    models['deep_neural_network'] = nn_model
    print(f"    Deep Neural Network Accuracy: {nn_accuracy:.3f}")
    
    return models, scaler, X_test, y_test

def save_models_and_data(models, scaler, X, y, feature_names):
    """Save models and data to disk."""
    print("💾 Saving models and data...")
    
    # Save individual models
    for model_name, model in models.items():
        model_path = f'models/{model_name}_model.pkl'
        joblib.dump(model, model_path)
        print(f"  ✓ Saved {model_name} model to {model_path}")
    
    # Save scaler
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved scaler to {scaler_path}")
    
    # Save synthetic data
    data_path = 'data/tcga_processed_data.npz'
    np.savez_compressed(
        data_path,
        features=X,
        labels=y,
        feature_names=feature_names,
        cancer_types=CANCER_TYPES
    )
    print(f"  ✓ Saved synthetic data to {data_path}")
    
    # Create model info file
    model_info = {
        'models': list(models.keys()),
        'n_samples': N_SAMPLES,
        'n_features': N_FEATURES,
        'n_cancer_types': N_CANCER_TYPES,
        'cancer_types': CANCER_TYPES,
        'feature_names': feature_names,
        'data_type': 'synthetic_demo',
        'note': 'This is synthetic data generated for demonstration purposes only.'
    }
    
    import json
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("  ✓ Saved model info to models/model_info.json")

def main():
    """Main function to generate demo data and models."""
    print("🔬 Cancer Alpha Demo Data Generator")
    print("=" * 50)
    
    try:
        # Create directories
        create_directories()
        
        # Generate synthetic data
        X, y, feature_names = generate_synthetic_genomic_data()
        
        # Train models
        models, scaler, X_test, y_test = train_models(X, y)
        
        # Save everything
        save_models_and_data(models, scaler, X, y, feature_names)
        
        print("\n✅ Demo data and models generated successfully!")
        print("\nGenerated files:")
        print("  - models/random_forest_model.pkl")
        print("  - models/gradient_boosting_model.pkl") 
        print("  - models/deep_neural_network_model.pkl")
        print("  - models/scaler.pkl")
        print("  - models/model_info.json")
        print("  - data/tcga_processed_data.npz")
        print("\n🚀 You can now run the Streamlit demo!")
        
    except Exception as e:
        print(f"\n❌ Error generating demo data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
