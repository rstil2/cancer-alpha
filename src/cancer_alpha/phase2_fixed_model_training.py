#!/usr/bin/env python3
"""
Phase 2 - Fixed Model Training
============================

This script creates compatible models for the Phase 4 API by using standard pickle
instead of joblib and ensuring proper model serialization.

Author: Cancer Alpha Research Team
Date: July 18, 2025
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_data(n_samples=1000):
    """Create synthetic cancer genomics data"""
    print("Generating synthetic cancer genomics data...")
    
    np.random.seed(42)
    
    # Generate 8 cancer types
    cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
    n_types = len(cancer_types)
    samples_per_type = n_samples // n_types
    
    features = []
    labels = []
    
    for i, cancer_type in enumerate(cancer_types):
        # Each cancer type has distinct patterns
        base_pattern = i * 0.1 + 0.2
        
        for j in range(samples_per_type):
            # Generate 110 features with cancer-specific patterns
            sample_features = []
            
            # Methylation features (20 features)
            methylation = np.random.normal(base_pattern, 0.1, 20)
            sample_features.extend(methylation)
            
            # Mutation features (25 features)
            mutations = np.random.poisson(5 + i * 2, 25)
            sample_features.extend(mutations)
            
            # Copy number features (20 features)
            copy_numbers = np.random.normal(10 + i * 3, 2, 20)
            sample_features.extend(copy_numbers)
            
            # Fragmentomics features (15 features)
            fragmentomics = np.random.exponential(150 + i * 10, 15)
            sample_features.extend(fragmentomics)
            
            # Clinical features (10 features)
            clinical = np.random.normal(0.5 + i * 0.05, 0.1, 10)
            sample_features.extend(clinical)
            
            # ICGC ARGO features (20 features)
            icgc = np.random.gamma(2 + i * 0.3, 0.5, 20)
            sample_features.extend(icgc)
            
            features.append(sample_features)
            labels.append(i)
    
    # Convert to arrays
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Generated {n_samples} samples with {X.shape[1]} features")
    print(f"Cancer types: {cancer_types}")
    
    return X, y, cancer_types

def train_models():
    """Train and save compatible models"""
    print("="*60)
    print("PHASE 2 - FIXED MODEL TRAINING")
    print("="*60)
    
    # Create output directory
    output_dir = Path("results/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    X, y, cancer_types = create_synthetic_data(1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Models to train
    models = {}
    results = {}
    
    # 1. Deep Neural Network
    print("\n1. Training Deep Neural Network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    
    models['deep_neural_network'] = mlp
    results['deep_neural_network'] = {'test_accuracy': mlp_accuracy}
    print(f"Deep Neural Network accuracy: {mlp_accuracy:.4f}")
    
    # 2. Gradient Boosting
    print("\n2. Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    models['gradient_boosting'] = gb
    results['gradient_boosting'] = {'test_accuracy': gb_accuracy}
    print(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
    
    # 3. Random Forest
    print("\n3. Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    models['random_forest'] = rf
    results['random_forest'] = {'test_accuracy': rf_accuracy}
    print(f"Random Forest accuracy: {rf_accuracy:.4f}")
    
    # 4. Ensemble Model
    print("\n4. Creating Ensemble Model...")
    ensemble_predictions = []
    for model in [mlp, gb, rf]:
        pred_proba = model.predict_proba(X_test_scaled)
        ensemble_predictions.append(pred_proba)
    
    # Average predictions
    ensemble_proba = np.mean(ensemble_predictions, axis=0)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    ensemble_model = {
        'models': models,
        'weights': {'deep_neural_network': 1/3, 'gradient_boosting': 1/3, 'random_forest': 1/3},
        'test_accuracy': ensemble_accuracy
    }
    
    results['ensemble'] = {'test_accuracy': ensemble_accuracy}
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Save models using standard pickle
    print("\n5. Saving Models...")
    
    # Save individual models
    for name, model in models.items():
        model_path = output_dir / f"{name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved {name}_model.pkl")
    
    # Save ensemble model
    ensemble_path = output_dir / "ensemble_model.pkl"
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved ensemble_model.pkl")
    
    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved scaler.pkl")
    
    # Save feature importance (from tree-based models)
    feature_importance = {}
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = model.feature_importances_.tolist()
    
    if feature_importance:
        with open(output_dir / "feature_importance.json", 'w') as f:
            json.dump(feature_importance, f, indent=2)
        print(f"✓ Saved feature_importance.json")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 2 - Fixed Model Training',
        'dataset_info': {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X.shape[1],
            'classes': len(cancer_types)
        },
        'cancer_types': cancer_types,
        'results': results
    }
    
    with open(output_dir / "phase2_report.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved phase2_report.json")
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"Best performing model: {best_model[0]} ({best_model[1]['test_accuracy']:.4f})")
    
    return models, results

def test_model_loading():
    """Test that the saved models can be loaded correctly"""
    print("\n6. Testing Model Loading...")
    
    models_dir = Path("results/phase2")
    model_files = [
        'deep_neural_network_model.pkl',
        'gradient_boosting_model.pkl',
        'random_forest_model.pkl',
        'ensemble_model.pkl',
        'scaler.pkl'
    ]
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✓ Successfully loaded {model_file}")
            except Exception as e:
                print(f"✗ Error loading {model_file}: {e}")
        else:
            print(f"✗ {model_file} not found")

if __name__ == "__main__":
    models, results = train_models()
    test_model_loading()
