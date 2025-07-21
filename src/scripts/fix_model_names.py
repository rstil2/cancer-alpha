#!/usr/bin/env python3
"""
Fix model file names for API compatibility
"""

import shutil
import os
from pathlib import Path

def fix_model_names():
    """Copy model files with correct names for API compatibility"""
    
    source_dir = Path("results/phase2_optimized")
    
    # Mapping of existing names to required names
    model_mapping = {
        "deep_neural_network_model.pkl": "deep_neural_network.pkl",
        "gradient_boosting_model.pkl": "gradient_boosting.pkl", 
        "random_forest_model.pkl": "random_forest.pkl",
        "ensemble_model.pkl": "ensemble_model.pkl",  # This one is already correct
        "scaler.pkl": "scaler.pkl"  # This one is already correct
    }
    
    print("Fixing model file names for API compatibility...")
    
    for source_name, target_name in model_mapping.items():
        source_path = source_dir / source_name
        target_path = source_dir / target_name
        
        if source_path.exists():
            if target_path.exists():
                print(f"✓ {target_name} already exists")
            else:
                shutil.copy2(source_path, target_path)
                print(f"✓ Created {target_name}")
        else:
            print(f"✗ Source file {source_name} not found")
    
    print("\nModel file names fixed!")
    print("API should now be able to load the models correctly.")

if __name__ == "__main__":
    fix_model_names()
