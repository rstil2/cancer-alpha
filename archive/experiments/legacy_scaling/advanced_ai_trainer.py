#!/usr/bin/env python3
"""
Oncura Advanced Multi-Omics AI Trainer (50K+ Samples)
======================================================

Trains advanced AI models on the comprehensive, integrated multi-omics dataset.

- Loads the integrated multi-omics dataset.
- Performs feature engineering to extract relevant features.
- Trains multiple machine learning models.
- Evaluates and saves the best-performing model.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import joblib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedAITrainer:
    """Trains advanced AI models on the integrated multi-omics dataset."""

    def __init__(self, integrated_data_path, output_dir):
        self.integrated_data_path = Path(integrated_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Loads the integrated multi-omics dataset."""
        logger.info(f"Loading integrated data from {self.integrated_data_path}...")
        self.df = pd.read_csv(self.integrated_data_path)
        logger.info(f"Successfully loaded {len(self.df)} samples.")

    def feature_engineering(self):
        """Performs feature engineering on the multi-omics data."""
        logger.info("Performing feature engineering...")
        
        # For simplicity, we'll use the presence/absence of data as features
        for col in self.df.columns:
            if col not in ['sample_id', 'cancer_type']:
                self.df[f'{col}_present'] = self.df[col].notna().astype(int)

        # Select features and target
        self.features = [col for col in self.df.columns if '_present' in col]
        self.target = 'cancer_type'

        X = self.df[self.features]
        y = self.df[self.target]

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_random_forest(self):
        """Trains a Random Forest model."""
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        return model

    def train_lightgbm(self):
        """Trains a LightGBM model."""
        logger.info("Training LightGBM model...")
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        return model

    def train_xgboost(self):
        """Trains an XGBoost model."""
        logger.info("Training XGBoost model...")
        # Need to encode labels for XGBoost
        y_train_encoded = self.y_train.astype('category').cat.codes
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(self.X_train, y_train_encoded)
        return model

    def evaluate_and_save_models(self):
        """Evaluates the models and saves the best one."""
        models = {
            'RandomForest': self.train_random_forest(),
            'LightGBM': self.train_lightgbm(),
            'XGBoost': self.train_xgboost(),
        }

        best_accuracy = 0
        best_model_name = None

        for name, model in models.items():
            if name == 'XGBoost':
                y_test_encoded = self.y_test.astype('category').cat.codes
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(y_test_encoded, y_pred)
            else:
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name

        logger.info(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")

        # Save the best model
        best_model = models[best_model_name]
        model_path = self.output_dir / f"oncura_best_model_50k.joblib"
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")

    def run(self):
        """Runs the full training pipeline."""
        self.load_data()
        self.feature_engineering()
        self.evaluate_and_save_models()

if __name__ == '__main__':
    integrated_data_path = '/Users/stillwell/projects/cancer-alpha/data/processed_50k/oncura_comprehensive_multi_omics_50k.csv'
    output_directory = '/Users/stillwell/projects/cancer-alpha/models/oncura_v2'

    trainer = AdvancedAITrainer(integrated_data_path, output_directory)
    trainer.run()
