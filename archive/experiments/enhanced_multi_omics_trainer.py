#!/usr/bin/env python3
"""
Enhanced Multi-Omics TCGA Machine Learning Trainer with Progress Tracking
=======================================================================

Features:
- Real-time progress tracking with timestamps
- Intermediate result saving
- Faster training with reduced hyperparameter search
- Progress bars and completion estimates
- Memory usage monitoring
- Model comparison and ensemble learning
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import psutil
import time
from tqdm import tqdm

class ProgressTracker:
    """Track progress with timestamps and estimates"""
    
    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.stages = {}
        
    def start_stage(self, stage_name: str, total_steps: int = 1):
        """Start tracking a new stage"""
        self.stages[stage_name] = {
            'start_time': time.time(),
            'total_steps': total_steps,
            'completed_steps': 0
        }
        self.logger.info(f"🚀 Starting {stage_name}...")
        
    def update_stage(self, stage_name: str, completed_steps: int = None):
        """Update progress for a stage"""
        if stage_name not in self.stages:
            return
            
        if completed_steps is not None:
            self.stages[stage_name]['completed_steps'] = completed_steps
        else:
            self.stages[stage_name]['completed_steps'] += 1
            
        stage = self.stages[stage_name]
        elapsed = time.time() - stage['start_time']
        progress = stage['completed_steps'] / stage['total_steps']
        
        if progress > 0:
            eta = elapsed / progress - elapsed
            self.logger.info(f"📊 {stage_name}: {stage['completed_steps']}/{stage['total_steps']} "
                           f"({progress:.1%}) - ETA: {eta:.0f}s")
        
    def complete_stage(self, stage_name: str):
        """Complete a stage"""
        if stage_name not in self.stages:
            return
            
        elapsed = time.time() - self.stages[stage_name]['start_time']
        self.logger.info(f"✅ {stage_name} completed in {elapsed:.1f}s")
        
    def get_total_elapsed(self):
        """Get total elapsed time"""
        return time.time() - self.start_time

class EnhancedMultiOmicsTrainer:
    """Enhanced Multi-Omics Machine Learning Trainer"""
    
    def __init__(self, dataset_path: str, output_dir: str = "results/enhanced_multi_omics"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging with progress
        self.setup_logging()
        self.tracker = ProgressTracker(self.logger)
        
        # Initialize attributes
        self.data = None
        self.X = None
        self.y = None
        self.label_encoder = None
        self.scaler = None
        self.feature_selector = None
        self.models = {}
        self.results = {}
        
        self.logger.info("🎓 Enhanced Multi-Omics Trainer initialized")
        self.logger.info(f"📂 Dataset: {dataset_path}")
        self.logger.info(f"📊 Output: {output_dir}")
        
    def setup_logging(self):
        """Setup logging with timestamps"""
        log_file = self.output_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def monitor_memory(self):
        """Monitor memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.logger.info(f"💾 Memory usage: {memory_mb:.1f} MB")
        
    def load_and_analyze_data(self):
        """Load dataset with progress tracking"""
        self.tracker.start_stage("Data Loading", 3)
        
        # Load data
        self.logger.info("📥 Loading multi-omics dataset...")
        self.data = pd.read_csv(self.dataset_path)
        self.tracker.update_stage("Data Loading")
        
        # Basic info
        self.logger.info(f"✅ Loaded dataset: {len(self.data):,} samples, {len(self.data.columns):,} features")
        self.monitor_memory()
        self.tracker.update_stage("Data Loading")
        
        # Quality analysis
        self.analyze_data_quality()
        self.tracker.update_stage("Data Loading")
        self.tracker.complete_stage("Data Loading")
        
        return self
        
    def analyze_data_quality(self):
        """Analyze data quality with progress"""
        # Missing values
        missing_cols = self.data.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        
        if len(missing_cols) > 0:
            self.logger.info(f"⚠️ Missing values found in {len(missing_cols)} columns")
            for col, count in missing_cols.head(10).items():
                pct = (count / len(self.data)) * 100
                self.logger.info(f"   {col}: {count:,} missing ({pct:.1f}%)")
                
        # Cancer types
        if 'cancer_type' in self.data.columns:
            cancer_counts = self.data['cancer_type'].value_counts()
            self.logger.info(f"🧬 Cancer types: {len(cancer_counts)} types")
            self.logger.info(f"   Most common: {dict(cancer_counts.head(3))}")
            self.logger.info(f"   Least common: {dict(cancer_counts.tail(3))}")
            
            # Filter small classes
            min_samples = 10
            valid_types = cancer_counts[cancer_counts >= min_samples].index
            self.data = self.data[self.data['cancer_type'].isin(valid_types)]
            self.logger.info(f"📊 After filtering (≥{min_samples} samples): {len(self.data):,} samples, {len(valid_types)} cancer types")
        
        # Omics coverage
        omics_cols = {}
        for col in self.data.columns:
            if col.startswith('cn_'):
                omics_cols.setdefault('copy_number', []).append(col)
            elif col.startswith('protein_'):
                omics_cols.setdefault('protein', []).append(col)
            elif col.startswith('mut_'):
                omics_cols.setdefault('mutation', []).append(col)
                
        # Count multi-omics samples
        multi_omics_count = {}
        for idx, row in self.data.iterrows():
            omics_present = 0
            for omics_type, cols in omics_cols.items():
                if any(pd.notna(row[col]) for col in cols):
                    omics_present += 1
            multi_omics_count[omics_present] = multi_omics_count.get(omics_present, 0) + 1
            
        self.logger.info("🎯 Multi-omics coverage:")
        for omics_count, sample_count in sorted(multi_omics_count.items(), reverse=True):
            if omics_count > 0:
                self.logger.info(f"   {omics_count} omics: {sample_count:,} samples")
                
    def prepare_features(self):
        """Prepare features with progress tracking"""
        self.tracker.start_stage("Feature Preparation", 6)
        
        # Separate features and target
        feature_cols = [col for col in self.data.columns if col not in ['sample_id', 'cancer_type']]
        self.X = self.data[feature_cols].copy()
        
        if 'cancer_type' in self.data.columns:
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.data['cancer_type'])
        else:
            raise ValueError("No cancer_type column found")
            
        self.logger.info(f"🎯 Feature columns: {len(feature_cols)}")
        self.tracker.update_stage("Feature Preparation")
        
        # Handle missing values with progress
        self.logger.info("🔄 Handling missing values...")
        imputer = KNNImputer(n_neighbors=5)
        self.X = pd.DataFrame(
            imputer.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )
        self.logger.info("✅ Missing values imputed using KNN")
        self.tracker.update_stage("Feature Preparation")
        
        # Create interaction features
        self.logger.info("🔗 Creating multi-omics interaction features...")
        interaction_features = []
        cn_cols = [col for col in self.X.columns if col.startswith('cn_')]
        protein_cols = [col for col in self.X.columns if col.startswith('protein_')]
        
        # Create interactions between different omics types
        if len(cn_cols) > 0 and len(protein_cols) > 0:
            for cn_col in cn_cols[:5]:  # Limit to prevent explosion
                for protein_col in protein_cols[:5]:
                    interaction_name = f"interact_{cn_col}_{protein_col}"
                    self.X[interaction_name] = self.X[cn_col] * self.X[protein_col]
                    interaction_features.append(interaction_name)
                    
        self.logger.info(f"✅ Created {len(interaction_features)} interaction features")
        self.tracker.update_stage("Feature Preparation")
        
        # Feature selection with progress
        self.logger.info("🎯 Performing feature selection...")
        k_features = min(100, len(self.X.columns))  # Limit features for speed
        self.feature_selector = SelectKBest(score_func=f_classif, k=k_features)
        self.X = pd.DataFrame(
            self.feature_selector.fit_transform(self.X, self.y),
            columns=self.X.columns[self.feature_selector.get_support()],
            index=self.X.index
        )
        
        self.logger.info(f"✅ Selected {len(self.X.columns)} features")
        feature_types = {}
        for col in self.X.columns:
            if col.startswith('cn_'):
                feature_types['Copy Number'] = feature_types.get('Copy Number', 0) + 1
            elif col.startswith('protein_'):
                feature_types['Protein'] = feature_types.get('Protein', 0) + 1
            elif col.startswith('interact_'):
                feature_types['Interaction'] = feature_types.get('Interaction', 0) + 1
                
        for ftype, count in feature_types.items():
            self.logger.info(f"   {ftype} features: {count}")
        self.tracker.update_stage("Feature Preparation")
        
        # Scale features
        self.logger.info("📏 Scaling features...")
        self.scaler = RobustScaler()
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )
        self.tracker.update_stage("Feature Preparation")
        
        # Save preprocessing objects
        preprocessing_path = self.output_dir / "preprocessing_objects.joblib"
        joblib.dump({
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector
        }, preprocessing_path)
        self.tracker.update_stage("Feature Preparation")
        
        self.tracker.complete_stage("Feature Preparation")
        return self
        
    def split_data(self):
        """Split data with progress tracking"""
        self.tracker.start_stage("Data Splitting", 2)
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.logger.info("📊 Data split complete:")
        self.logger.info(f"   Training: {len(X_train):,} samples")
        self.logger.info(f"   Testing: {len(X_test):,} samples")
        self.logger.info(f"   Features: {len(self.X.columns)}")
        self.logger.info(f"   Classes: {len(np.unique(self.y))}")
        self.tracker.update_stage("Data Splitting")
        
        # Apply SMOTE for class balancing
        self.logger.info("⚖️ Handling class imbalance...")
        smote = SMOTE(random_state=42, k_neighbors=3)  # Reduced k for speed
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        self.logger.info(f"   Original training samples: {len(X_train):,}")
        self.logger.info(f"   Balanced training samples: {len(X_train_balanced):,}")
        
        self.train_data = (X_train_balanced, y_train_balanced)
        self.test_data = (X_test, y_test)
        
        self.tracker.update_stage("Data Splitting")
        self.tracker.complete_stage("Data Splitting")
        
        return self
        
    def train_models(self):
        """Train models with progress tracking"""
        # Simplified model configs for faster training
        model_configs = {
            'LightGBM': {
                'model': LGBMClassifier(
                    n_estimators=100,  # Reduced from default
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbosity=-1
                ),
                'params': {}  # No grid search for speed
            },
            'XGBoost': {
                'model': XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbosity=0
                ),
                'params': {}
            },
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {}
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    n_jobs=-1
                ),
                'params': {}
            }
        }
        
        self.tracker.start_stage("Model Training", len(model_configs))
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        for model_name, config in model_configs.items():
            self.logger.info(f"🔄 Training {model_name}...")
            start_time = time.time()
            
            # Train model
            model = config['model']
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            train_bal_acc = balanced_accuracy_score(y_train, train_pred)
            test_bal_acc = balanced_accuracy_score(y_test, test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='balanced_accuracy')  # Reduced CV folds
            
            training_time = time.time() - start_time
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_balanced_accuracy': train_bal_acc,
                'test_balanced_accuracy': test_bal_acc,
                'cv_balanced_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            
            self.logger.info(f"✅ {model_name} completed in {training_time:.1f}s:")
            self.logger.info(f"   Test Accuracy: {test_acc:.4f}")
            self.logger.info(f"   Test Balanced Accuracy: {test_bal_acc:.4f}")
            self.logger.info(f"   CV Balanced Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Save intermediate results
            self.save_intermediate_results()
            self.monitor_memory()
            
            self.tracker.update_stage("Model Training")
            
        self.tracker.complete_stage("Model Training")
        return self
        
    def create_ensemble(self):
        """Create ensemble with progress tracking"""
        self.tracker.start_stage("Ensemble Creation", 2)
        
        # Select best models for ensemble
        best_models = []
        for model_name, results in self.results.items():
            if results['test_balanced_accuracy'] > 0.3:  # Threshold for inclusion
                best_models.append((model_name, self.models[model_name]))
                
        if len(best_models) >= 2:
            self.logger.info(f"🤝 Creating ensemble from {len(best_models)} models...")
            
            ensemble = VotingClassifier(
                estimators=best_models,
                voting='soft'
            )
            
            X_train, y_train = self.train_data
            X_test, y_test = self.test_data
            
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            test_pred = ensemble.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            test_bal_acc = balanced_accuracy_score(y_test, test_pred)
            
            self.models['Ensemble'] = ensemble
            self.results['Ensemble'] = {
                'test_accuracy': test_acc,
                'test_balanced_accuracy': test_bal_acc,
                'training_time': 0,  # Already included in individual models
                'cv_balanced_accuracy': test_bal_acc,  # Approximation
                'cv_std': 0
            }
            
            self.logger.info(f"✅ Ensemble created:")
            self.logger.info(f"   Test Accuracy: {test_acc:.4f}")
            self.logger.info(f"   Test Balanced Accuracy: {test_bal_acc:.4f}")
            
        self.tracker.update_stage("Ensemble Creation")
        self.tracker.complete_stage("Ensemble Creation")
        
        return self
        
    def save_intermediate_results(self):
        """Save intermediate results"""
        results_path = self.output_dir / "intermediate_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
    def generate_final_report(self):
        """Generate comprehensive final report"""
        self.tracker.start_stage("Report Generation", 3)
        
        # Model comparison
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values('test_balanced_accuracy', ascending=False)
        
        self.logger.info("🏆 Final Model Rankings:")
        self.logger.info("=" * 80)
        for idx, (model_name, results) in enumerate(comparison_df.iterrows(), 1):
            self.logger.info(f"{idx}. {model_name}:")
            self.logger.info(f"   Test Balanced Accuracy: {results['test_balanced_accuracy']:.4f}")
            self.logger.info(f"   Test Accuracy: {results['test_accuracy']:.4f}")
            if 'cv_balanced_accuracy' in results:
                self.logger.info(f"   CV Balanced Accuracy: {results['cv_balanced_accuracy']:.4f}")
            self.logger.info(f"   Training Time: {results.get('training_time', 0):.1f}s")
            self.logger.info("")
            
        self.tracker.update_stage("Report Generation")
        
        # Save detailed results
        comparison_df.to_csv(self.output_dir / "model_comparison.csv")
        
        # Save best model
        best_model_name = comparison_df.index[0]
        best_model = self.models[best_model_name]
        joblib.dump(best_model, self.output_dir / f"best_model_{best_model_name.lower()}.joblib")
        
        self.logger.info(f"💾 Best model ({best_model_name}) saved")
        self.tracker.update_stage("Report Generation")
        
        # Final summary
        total_time = self.tracker.get_total_elapsed()
        self.logger.info("🎉 Training Pipeline Complete!")
        self.logger.info(f"⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        self.logger.info(f"📂 Results saved to: {self.output_dir}")
        self.logger.info(f"🏆 Champion model: {best_model_name}")
        self.logger.info(f"🎯 Best performance: {comparison_df.iloc[0]['test_balanced_accuracy']:.4f}")
        
        self.tracker.update_stage("Report Generation")
        self.tracker.complete_stage("Report Generation")
        
        return comparison_df
        
    def run_complete_pipeline(self):
        """Run the complete training pipeline with progress tracking"""
        self.logger.info("🚀 Starting Enhanced Multi-Omics Training Pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Execute pipeline stages
            self.load_and_analyze_data()
            self.prepare_features()
            self.split_data()
            self.train_models()
            self.create_ensemble()
            results = self.generate_final_report()
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            self.logger.error("💾 Saving partial results...")
            self.save_intermediate_results()
            raise


def main():
    """Main execution function"""
    # Configuration
    dataset_path = "data/focused_multi_omics/tcga_focused_multi_omics_20250820_133706.csv"
    output_dir = "results/enhanced_multi_omics"
    
    print("🎓 Enhanced Multi-Omics TCGA Machine Learning")
    print("=" * 50)
    print(f"📂 Dataset: {dataset_path}")
    print(f"📊 Output: {output_dir}")
    print()
    
    # Initialize and run trainer
    trainer = EnhancedMultiOmicsTrainer(dataset_path, output_dir)
    results = trainer.run_complete_pipeline()
    
    return trainer, results


if __name__ == "__main__":
    trainer = main()
