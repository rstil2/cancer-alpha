#!/usr/bin/env python3
"""
Generate demo data and models for Cancer Alpha Demo
This script creates the necessary data files and models when missing.
"""

import os
import numpy as np
import pickle
import json
from pathlib import Path

def generate_enhanced_data():
    """Generate enhanced synthetic genomic data"""
    print("🔬 Generating enhanced synthetic genomic data...")
    
    # Create data directories
    os.makedirs("data/enhanced", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Generate synthetic data with 270 features and 1000 samples
    n_samples = 1000
    n_features = 270
    n_classes = 8
    
    # Generate features with realistic genomic patterns
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to make classes separable
    for i in range(n_classes):
        class_mask = np.arange(i * n_samples // n_classes, (i + 1) * n_samples // n_classes)
        feature_offset = i * 30  # Each class has different feature patterns
        X[class_mask, feature_offset:feature_offset + 30] += 2.0
    
    # Generate labels
    y = np.repeat(range(n_classes), n_samples // n_classes)
    
    # Save enhanced data
    np.save("data/enhanced/enhanced_X.npy", X)
    np.save("data/enhanced/enhanced_y.npy", y)
    
    # Create feature names
    feature_names = [f"genomic_feature_{i}" for i in range(n_features)]
    with open("data/enhanced/enhanced_feature_names.json", "w") as f:
        json.dump(feature_names, f)
    
    # Create metadata
    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "cancer_types": ["BRCA", "LUAD", "COAD", "PRAD", "STAD", "KIRC", "HNSC", "LIHC"]
    }
    with open("data/enhanced/enhanced_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Generated enhanced data: {n_samples} samples, {n_features} features")
    return X, y

def generate_demo_models(X, y):
    """Generate demo models"""
    print("🤖 Training demo models...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Save Random Forest
    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    
    # Train Logistic Regression with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Save Logistic Regression and scaler
    with open("models/logistic_regression_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    
    with open("models/standard_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Test accuracy
    rf_score = rf_model.score(X_test, y_test)
    lr_score = lr_model.score(scaler.transform(X_test), y_test)
    
    print(f"✅ Random Forest accuracy: {rf_score:.3f}")
    print(f"✅ Logistic Regression accuracy: {lr_score:.3f}")

def main():
    """Main function to generate all demo data and models"""
    print("🚀 Cancer Alpha Demo - Generating Demo Data and Models")
    print("=" * 60)
    
    # Check if data already exists
    if Path("data/enhanced/enhanced_X.npy").exists():
        print("✅ Enhanced data already exists, loading...")
        X = np.load("data/enhanced/enhanced_X.npy")
        y = np.load("data/enhanced/enhanced_y.npy")
    else:
        X, y = generate_enhanced_data()
    
    # Check if models already exist
    if not Path("models/random_forest_model.pkl").exists():
        generate_demo_models(X, y)
    else:
        print("✅ Demo models already exist")
    
    print("=" * 60)
    print("🎉 Demo data and models ready!")
    print("   You can now run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
