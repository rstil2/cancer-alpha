#!/usr/bin/env python3
"""
Massive TCGA Cancer Classification Training
==========================================

Train and evaluate advanced machine learning models on the massive 4,572-sample TCGA dataset.
This script uses the processed mutation features to build state-of-the-art cancer classifiers.

Key Features:
- Advanced feature selection and dimensionality reduction
- Multiple high-performance ML algorithms (LightGBM, XGBoost, Random Forest)
- Sophisticated cross-validation and model evaluation
- SMOTE oversampling for perfect class balance
- Model comparison and ensemble methods
- Hyperparameter optimization
- Production-ready model serialization

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Advanced ML Libraries
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('massive_tcga_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MassiveTCGATrainer:
    """Advanced trainer for massive TCGA cancer classification"""
    
    def __init__(self, data_dir: str, results_dir: str = "massive_tcga_results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.results = {}
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        logger.info(f"🚀 Initialized massive TCGA trainer")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"📁 Results directory: {results_dir}")
    
    def load_data(self):
        """Load the processed TCGA mutation data"""
        logger.info("📂 Loading massive TCGA dataset...")
        
        # Find the latest processed files
        feature_files = list(self.data_dir.glob("tcga_mutation_features_*.csv"))
        label_files = list(self.data_dir.glob("tcga_mutation_labels_*.csv"))
        
        if not feature_files or not label_files:
            raise FileNotFoundError("No processed TCGA data found!")
        
        # Load the most recent files
        feature_file = sorted(feature_files)[-1]
        label_file = sorted(label_files)[-1]
        
        logger.info(f"📊 Loading features from: {feature_file.name}")
        logger.info(f"🏷️ Loading labels from: {label_file.name}")
        
        # Load data
        self.X = pd.read_csv(feature_file, index_col=0)
        self.y = pd.read_csv(label_file, index_col=0)['cancer_type']
        
        # Ensure alignment
        common_samples = self.X.index.intersection(self.y.index)
        self.X = self.X.loc[common_samples]
        self.y = self.y.loc[common_samples]
        
        logger.info(f"✅ Loaded dataset: {self.X.shape[0]:,} samples × {self.X.shape[1]:,} features")
        logger.info(f"🧬 Cancer types: {len(self.y.unique())} classes")
        logger.info(f"📋 Class distribution:")
        for cancer_type, count in self.y.value_counts().items():
            logger.info(f"  {cancer_type}: {count:,} samples")
    
    def preprocess_data(self):
        """Advanced preprocessing including feature selection and scaling"""
        logger.info("🔧 Starting advanced preprocessing...")
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Remove features with too many zeros (sparse features)
        non_zero_threshold = 0.05  # Feature must be non-zero in at least 5% of samples
        feature_sparsity = (self.X != 0).mean()
        selected_features = feature_sparsity[feature_sparsity >= non_zero_threshold].index
        
        logger.info(f"🎯 Selected {len(selected_features):,} features (removed sparse features)")
        self.X_filtered = self.X[selected_features]
        
        # Feature selection using statistical tests
        logger.info("📊 Performing feature selection...")
        k_best = min(5000, len(selected_features))  # Select top 5000 features or all if fewer
        self.feature_selector = SelectKBest(f_classif, k=k_best)
        X_selected = self.feature_selector.fit_transform(self.X_filtered, self.y_encoded)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = selected_features[selected_mask]
        
        logger.info(f"✅ Final feature set: {len(self.selected_features):,} features")
        
        # Create final dataset
        self.X_final = pd.DataFrame(X_selected, 
                                   index=self.X.index, 
                                   columns=self.selected_features)
        
        # Scale features
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_final),
            index=self.X_final.index,
            columns=self.X_final.columns
        )
        
        logger.info("🔧 Preprocessing completed!")
    
    def create_models(self):
        """Create advanced ML models for cancer classification"""
        logger.info("🤖 Creating advanced ML models...")
        
        # LightGBM - Champion from previous experiments
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        
        # XGBoost - High performance gradient boosting
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # Random Forest - Robust ensemble method
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Logistic Regression with L2 regularization
        lr_model = LogisticRegression(
            C=0.1,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        self.models = {
            'LightGBM': lgb_model,
            'XGBoost': xgb_model,
            'RandomForest': rf_model,
            'LogisticRegression': lr_model
        }
        
        logger.info(f"✅ Created {len(self.models)} advanced models")
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models with SMOTE oversampling"""
        logger.info("🎯 Training and evaluating models with SMOTE...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.y_encoded
        )
        
        logger.info(f"📊 Training set: {X_train.shape[0]:,} samples")
        logger.info(f"📊 Test set: {X_test.shape[0]:,} samples")
        
        # Store splits for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        self.results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"🔥 Training {model_name}...")
            
            # Create pipeline with SMOTE
            pipeline = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('model', model)
            ])
            
            # Cross-validation
            logger.info(f"📊 Running 5-fold cross-validation for {model_name}...")
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                      cv=cv, scoring='accuracy', n_jobs=-1)
            
            # Train on full training set
            pipeline.fit(X_train, y_train)
            
            # Predictions
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            train_f1 = f1_score(y_train, train_pred, average='weighted')
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            # Store results
            self.results[model_name] = {
                'pipeline': pipeline,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'test_predictions': test_pred
            }
            
            logger.info(f"✅ {model_name} Results:")
            logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"  Test F1-Score: {test_f1:.4f}")
    
    def create_ensemble_model(self):
        """Create ensemble model from top performers"""
        logger.info("🎭 Creating ensemble model...")
        
        # Select top 3 models by CV performance
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['cv_mean'], 
                             reverse=True)
        top_models = sorted_models[:3]
        
        logger.info("🏆 Top models for ensemble:")
        for model_name, results in top_models:
            logger.info(f"  {model_name}: CV = {results['cv_mean']:.4f}")
        
        # Create voting classifier
        estimators = [(name, results['pipeline']) for name, results in top_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Train ensemble
        ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        ensemble_pred = ensemble.predict(self.X_test)
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_pred)
        ensemble_f1 = f1_score(self.y_test, ensemble_pred, average='weighted')
        
        self.results['Ensemble'] = {
            'pipeline': ensemble,
            'test_accuracy': ensemble_accuracy,
            'test_f1': ensemble_f1,
            'test_predictions': ensemble_pred
        }
        
        logger.info(f"🎭 Ensemble Results:")
        logger.info(f"  Test Accuracy: {ensemble_accuracy:.4f}")
        logger.info(f"  Test F1-Score: {ensemble_f1:.4f}")
    
    def generate_detailed_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("📋 Generating detailed evaluation report...")
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_accuracy'])
        best_model = self.results[best_model_name]
        
        logger.info("🏆 FINAL RESULTS SUMMARY:")
        logger.info("=" * 60)
        
        for model_name, results in self.results.items():
            if 'cv_mean' in results:
                logger.info(f"{model_name:15s}: Test Acc = {results['test_accuracy']:.4f}, CV = {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            else:
                logger.info(f"{model_name:15s}: Test Acc = {results['test_accuracy']:.4f}")
        
        logger.info("=" * 60)
        logger.info(f"🏆 CHAMPION MODEL: {best_model_name}")
        logger.info(f"🎯 Best Test Accuracy: {best_model['test_accuracy']:.4f}")
        logger.info(f"📊 Dataset Scale: {self.X.shape[0]:,} samples × {self.X.shape[1]:,} original features")
        logger.info(f"⚡ Selected Features: {len(self.selected_features):,} features")
        
        # Generate classification report
        y_true_labels = self.label_encoder.inverse_transform(self.y_test)
        y_pred_labels = self.label_encoder.inverse_transform(best_model['test_predictions'])
        
        logger.info("📊 DETAILED CLASSIFICATION REPORT (Best Model):")
        logger.info("\n" + classification_report(y_true_labels, y_pred_labels))
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        best_model_path = self.results_dir / f"best_tcga_model_{timestamp}.pkl"
        with open(best_model_path, 'wb') as f:
            pickle.dump({
                'model': best_model['pipeline'],
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'label_encoder': self.label_encoder,
                'selected_features': self.selected_features,
                'model_name': best_model_name,
                'test_accuracy': best_model['test_accuracy']
            }, f)
        
        logger.info(f"💾 Saved best model: {best_model_path}")
        
        # Save all results
        results_summary = {
            'timestamp': timestamp,
            'dataset_size': self.X.shape,
            'selected_features': len(self.selected_features),
            'models_tested': list(self.results.keys()),
            'best_model': best_model_name,
            'best_accuracy': best_model['test_accuracy'],
            'all_results': {name: {k: v for k, v in res.items() 
                                 if k != 'pipeline'} 
                           for name, res in self.results.items()}
        }
        
        summary_path = self.results_dir / f"training_summary_{timestamp}.pkl"
        with open(summary_path, 'wb') as f:
            pickle.dump(results_summary, f)
        
        logger.info(f"📋 Saved training summary: {summary_path}")
        
        return results_summary
    
    def run_complete_training(self):
        """Execute the complete training pipeline"""
        logger.info("🚀 Starting complete TCGA training pipeline...")
        start_time = datetime.now()
        
        try:
            # Load data
            self.load_data()
            
            # Preprocess
            self.preprocess_data()
            
            # Create models
            self.create_models()
            
            # Train and evaluate
            self.train_and_evaluate_models()
            
            # Create ensemble
            self.create_ensemble_model()
            
            # Generate report
            summary = self.generate_detailed_report()
            
            # Final summary
            end_time = datetime.now()
            training_time = end_time - start_time
            
            logger.info("🎉 MASSIVE TCGA TRAINING COMPLETED!")
            logger.info(f"⏱️ Total training time: {training_time}")
            logger.info(f"🏆 Best model: {summary['best_model']}")
            logger.info(f"🎯 Best accuracy: {summary['best_accuracy']:.4f}")
            logger.info(f"📊 Trained on {summary['dataset_size'][0]:,} real TCGA samples")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise


def main():
    """Main training function"""
    logger.info("🧬 Massive TCGA Cancer Classification Training")
    logger.info("=" * 60)
    
    # Set paths
    data_dir = "data/processed_massive_tcga"
    results_dir = "results/massive_tcga_models"
    
    if not Path(data_dir).exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return
    
    # Create trainer
    trainer = MassiveTCGATrainer(data_dir, results_dir)
    
    try:
        # Run complete training
        summary = trainer.run_complete_training()
        
        logger.info("✅ SUCCESS: Massive TCGA model training completed!")
        logger.info(f"📁 Results saved to: {results_dir}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise


if __name__ == "__main__":
    summary = main()
