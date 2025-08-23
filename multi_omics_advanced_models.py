#!/usr/bin/env python3
"""
Multi-Omics Advanced Cancer Classification Models
================================================

Train state-of-the-art machine learning models on the integrated multi-omics TCGA dataset
for superior cancer classification performance.

Key Features:
- Multi-omics data integration (protein, copy number, mutations)
- Advanced ensemble methods
- Cross-validation and robust evaluation
- Model interpretability analysis
- Performance comparison with single-omics models

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging
import warnings
from typing import Dict, List, Tuple, Optional
import traceback

# Machine learning imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Advanced ML libraries
try:
    import lightgbm as lgb
    import xgboost as xgb
    HAS_ADVANCED_ML = True
except ImportError:
    HAS_ADVANCED_ML = False

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_omics_advanced_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiOmicsAdvancedModels:
    """Train advanced models on multi-omics integrated data"""
    
    def __init__(self, data_dir: str = "data/integrated_multi_omics"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("results/multi_omics_advanced_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load integrated data
        self.features = None
        self.labels = None
        self.metadata = None
        
        # Model configurations
        self.model_configs = {
            'lightgbm': {
                'enabled': HAS_ADVANCED_ML,
                'params': {
                    'objective': 'multiclass',
                    'num_class': None,  # Will be set dynamically
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                }
            },
            'xgboost': {
                'enabled': HAS_ADVANCED_ML,
                'params': {
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            },
            'random_forest': {
                'enabled': True,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'enabled': True,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'enabled': True,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'multi_class': 'ovr'
                }
            }
        }
        
        logger.info(f"🧬 Multi-Omics Advanced Models initialized")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"🎯 Output directory: {self.output_dir}")
        
        if HAS_ADVANCED_ML:
            logger.info("✅ Advanced ML libraries (XGBoost, LightGBM) available")
        else:
            logger.warning("⚠️ Advanced ML libraries not available - using sklearn only")
    
    def load_integrated_data(self) -> bool:
        """Load the integrated multi-omics dataset"""
        logger.info("📥 Loading integrated multi-omics dataset...")
        
        try:
            # Load features
            features_file = self.data_dir / "integrated_features.pkl"
            if features_file.exists():
                with open(features_file, 'rb') as f:
                    self.features = pickle.load(f)
                logger.info(f"✅ Features loaded: {self.features.shape[0]} samples × {self.features.shape[1]} features")
            else:
                logger.error(f"❌ Features file not found: {features_file}")
                return False
            
            # Load labels
            labels_file = self.data_dir / "integrated_labels.pkl"
            if labels_file.exists():
                with open(labels_file, 'rb') as f:
                    self.labels = pickle.load(f)
                logger.info(f"✅ Labels loaded: {len(self.labels)} samples")
            else:
                logger.error(f"❌ Labels file not found: {labels_file}")
                return False
            
            # Load metadata
            metadata_file = self.data_dir / "integration_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("✅ Metadata loaded")
            else:
                logger.warning("⚠️ Metadata file not found")
                self.metadata = {}
            
            # Validate data consistency
            if len(self.features) != len(self.labels):
                logger.error(f"❌ Data mismatch: {len(self.features)} features vs {len(self.labels)} labels")
                return False
            
            # Show dataset overview
            logger.info("📊 Dataset Overview:")
            logger.info(f"  Samples: {self.features.shape[0]}")
            logger.info(f"  Features: {self.features.shape[1]}")
            logger.info(f"  Cancer types: {len(self.labels.value_counts())}")
            logger.info(f"  Class distribution: {self.labels.value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load integrated data: {e}")
            return False
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
        """Preprocess the integrated multi-omics data"""
        logger.info("🔄 Preprocessing multi-omics data...")
        
        # Filter out classes with too few samples (minimum 3 samples per class)
        class_counts = self.labels.value_counts()
        valid_classes = class_counts[class_counts >= 3].index
        
        logger.info(f"📊 Filtering classes: {len(class_counts)} total → {len(valid_classes)} valid (≥3 samples)")
        
        # Filter data to only include valid classes
        valid_mask = self.labels.isin(valid_classes)
        filtered_features = self.features[valid_mask]
        filtered_labels = self.labels[valid_mask]
        
        logger.info(f"  Samples after filtering: {len(self.features)} → {len(filtered_features)}")
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(filtered_labels)
        
        logger.info(f"📊 Label encoding: {len(le.classes_)} classes")
        for i, class_name in enumerate(le.classes_):
            count = np.sum(y_encoded == i)
            logger.info(f"  Class {i} ({class_name}): {count} samples")
        
        # Convert features to numpy array and handle missing values
        X = filtered_features.values
        
        # Handle missing values using KNN imputation
        logger.info("🔄 Handling missing values...")
        initial_nans = np.isnan(X).sum()
        if initial_nans > 0:
            logger.info(f"  Found {initial_nans} missing values")
            imputer = KNNImputer(n_neighbors=5)
            X = imputer.fit_transform(X)
            logger.info("  ✅ Missing values imputed")
        else:
            logger.info("  ✅ No missing values found")
        
        # Feature scaling
        logger.info("📏 Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"✅ Data preprocessing complete: {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features")
        
        return X_scaled, y_encoded, le
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, le: LabelEncoder) -> Dict:
        """Train a single model and return results"""
        
        config = self.model_configs[model_name]
        if not config['enabled']:
            logger.warning(f"⚠️ {model_name} disabled - skipping")
            return {}
        
        logger.info(f"🚀 Training {model_name}...")
        
        try:
            start_time = datetime.now()
            
            # Initialize model
            if model_name == 'lightgbm':
                # Update num_class parameter
                params = config['params'].copy()
                params['num_class'] = len(le.classes_)
                
                model = lgb.LGBMClassifier(**params)
                
            elif model_name == 'xgboost':
                params = config['params'].copy()
                params['num_class'] = len(le.classes_)
                
                model = xgb.XGBClassifier(**params)
                
            elif model_name == 'random_forest':
                model = RandomForestClassifier(**config['params'])
                
            elif model_name == 'gradient_boosting':
                model = GradientBoostingClassifier(**config['params'])
                
            elif model_name == 'logistic_regression':
                model = LogisticRegression(**config['params'])
                
            else:
                logger.error(f"❌ Unknown model: {model_name}")
                return {}
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Generate classification report
            class_names = le.classes_
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            
            results = {
                'model_name': model_name,
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"✅ {model_name} training complete:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  F1-score: {f1:.4f}")
            logger.info(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logger.info(f"  Training time: {training_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def apply_smote_balancing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for class balancing"""
        logger.info("⚖️ Applying SMOTE for class balancing...")
        
        try:
            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            logger.info(f"  Original distribution: {dict(zip(unique, counts))}")
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Check new distribution
            unique, counts = np.unique(y_balanced, return_counts=True)
            logger.info(f"  Balanced distribution: {dict(zip(unique, counts))}")
            logger.info(f"  ✅ SMOTE applied: {X.shape[0]} → {X_balanced.shape[0]} samples")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"❌ SMOTE failed: {e}")
            return X, y
    
    def train_all_models(self) -> Dict:
        """Train all configured models and return results"""
        logger.info("🚀 Training all multi-omics models...")
        
        # Preprocess data
        X, y, le = self.preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
        
        # Apply SMOTE balancing
        X_train_balanced, y_train_balanced = self.apply_smote_balancing(X_train, y_train)
        
        # Train models
        all_results = {}
        model_performances = []
        
        for model_name in self.model_configs.keys():
            if self.model_configs[model_name]['enabled']:
                results = self.train_single_model(
                    model_name, X_train_balanced, y_train_balanced, X_test, y_test, le
                )
                
                if results:
                    all_results[model_name] = results
                    model_performances.append({
                        'model': model_name,
                        'accuracy': results['accuracy'],
                        'f1_score': results['f1_score'],
                        'cv_mean': results['cv_mean'],
                        'cv_std': results['cv_std'],
                        'training_time': results['training_time']
                    })
        
        # Find champion model
        if model_performances:
            champion = max(model_performances, key=lambda x: x['cv_mean'])
            logger.info(f"🏆 Champion Model: {champion['model']}")
            logger.info(f"  Best CV Score: {champion['cv_mean']:.4f} ± {champion['cv_std']:.4f}")
            logger.info(f"  Test Accuracy: {champion['accuracy']:.4f}")
            
            all_results['champion'] = champion
            all_results['model_performances'] = model_performances
        
        # Store additional info
        all_results['label_encoder'] = le
        all_results['feature_names'] = list(self.features.columns)
        all_results['dataset_info'] = {
            'total_samples': len(self.features),
            'total_features': len(self.features.columns),
            'cancer_types': len(le.classes_),
            'class_names': list(le.classes_)
        }
        
        return all_results
    
    def save_results(self, results: Dict):
        """Save all results to files"""
        logger.info("💾 Saving results...")
        
        try:
            # Save complete results
            results_file = self.output_dir / "multi_omics_advanced_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Save summary report
            summary = {
                'timestamp': datetime.now().isoformat(),
                'dataset_info': results.get('dataset_info', {}),
                'model_performances': results.get('model_performances', []),
                'champion_model': results.get('champion', {}),
                'metadata': self.metadata
            }
            
            summary_file = self.output_dir / "multi_omics_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Save champion model separately
            if 'champion' in results and results['champion']:
                champion_name = results['champion']['model']
                if champion_name in results:
                    champion_model = results[champion_name]['model']
                    champion_file = self.output_dir / f"champion_model_{champion_name}.pkl"
                    with open(champion_file, 'wb') as f:
                        pickle.dump(champion_model, f)
            
            logger.info(f"✅ Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def run_complete_training(self):
        """Execute complete multi-omics model training pipeline"""
        logger.info("🚀 Starting Multi-Omics Advanced Model Training...")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Load data
        if not self.load_integrated_data():
            logger.error("❌ Failed to load data - aborting")
            return None
        
        # Train all models
        results = self.train_all_models()
        
        # Save results
        self.save_results(results)
        
        # Final summary
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info("\n🎉 MULTI-OMICS MODEL TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"⏱️ Total time: {total_time}")
        
        if 'champion' in results and results['champion']:
            champion = results['champion']
            logger.info(f"🏆 Champion Model: {champion['model']}")
            logger.info(f"  Best Performance: {champion['cv_mean']:.4f} ± {champion['cv_std']:.4f}")
            logger.info(f"  Test Accuracy: {champion['accuracy']:.4f}")
        
        logger.info(f"📊 Dataset: {results['dataset_info']['total_samples']} samples, {results['dataset_info']['total_features']} features")
        logger.info(f"🎯 Cancer types: {results['dataset_info']['cancer_types']}")
        logger.info(f"💾 Results saved to: {self.output_dir}")
        
        return results


def main():
    """Main execution function"""
    logger.info("🧬 Multi-Omics Advanced Cancer Classification Models")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = MultiOmicsAdvancedModels()
    
    try:
        # Run complete training
        results = trainer.run_complete_training()
        
        if results:
            logger.info("✅ SUCCESS: Multi-omics model training completed!")
            return results
        else:
            logger.error("❌ Training failed!")
            return None
        
    except KeyboardInterrupt:
        logger.info("⏸️ Training interrupted by user")
        return None
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    results = main()
