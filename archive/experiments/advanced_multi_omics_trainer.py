#!/usr/bin/env python3
"""
Advanced Multi-Omics Machine Learning Trainer
=============================================

Train advanced machine learning models on the scaled-up multi-omics TCGA dataset
with comprehensive feature engineering, model selection, and evaluation.

Features:
- Multi-omics feature engineering and selection
- Advanced ensemble models (LightGBM, XGBoost, Random Forest)
- Stratified sampling for balanced training
- Comprehensive model evaluation
- Feature importance analysis
- Multi-class cancer type prediction

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import numpy as np
import pandas as pd
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel, mutual_info_classif
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Advanced models
import lightgbm as lgb
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('default')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_multi_omics_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedMultiOmicsTrainer:
    """Advanced trainer for multi-omics cancer classification"""
    
    def __init__(self, dataset_path: str, output_dir: str = "results/advanced_multi_omics"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.random_state = 42
        self.test_size = 0.2
        self.cv_folds = 5
        self.min_class_samples = 10  # Minimum samples per cancer type
        
        # Data containers
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None
        
        # Model containers
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        
        logger.info(f"🎓 Advanced Multi-Omics Trainer initialized")
        logger.info(f"📂 Dataset: {dataset_path}")
        logger.info(f"📊 Output: {output_dir}")
    
    def load_and_prepare_data(self):
        """Load and prepare the multi-omics dataset"""
        logger.info("📥 Loading multi-omics dataset...")
        
        # Load dataset
        self.df = pd.read_csv(self.dataset_path)
        logger.info(f"✅ Loaded dataset: {len(self.df):,} samples, {len(self.df.columns):,} features")
        
        # Data quality check
        logger.info("🔍 Data quality analysis...")
        
        # Missing values
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"⚠️ Missing values found in {(missing_counts > 0).sum()} columns")
            for col, count in missing_counts[missing_counts > 0].items():
                logger.info(f"  {col}: {count} missing ({count/len(self.df)*100:.1f}%)")
        else:
            logger.info("✅ No missing values found")
        
        # Cancer type distribution
        if 'cancer_type' in self.df.columns:
            cancer_counts = self.df['cancer_type'].value_counts()
            logger.info(f"🧬 Cancer types: {len(cancer_counts)} types")
            logger.info(f"  Most common: {cancer_counts.head(3).to_dict()}")
            logger.info(f"  Least common: {cancer_counts.tail(3).to_dict()}")
            
            # Filter cancer types with sufficient samples
            valid_cancers = cancer_counts[cancer_counts >= self.min_class_samples].index
            self.df = self.df[self.df['cancer_type'].isin(valid_cancers)]
            logger.info(f"📊 After filtering (≥{self.min_class_samples} samples): "
                       f"{len(self.df):,} samples, {len(valid_cancers)} cancer types")
        
        # Omics coverage analysis
        if 'omics_count' in self.df.columns:
            omics_dist = self.df['omics_count'].value_counts().sort_index(ascending=False)
            logger.info("🎯 Multi-omics coverage:")
            for count, samples in omics_dist.items():
                logger.info(f"  {count} omics: {samples:,} samples")
    
    def feature_engineering(self):
        """Advanced feature engineering for multi-omics data"""
        logger.info("🔧 Advanced feature engineering...")
        
        # Prepare target variable
        if 'cancer_type' in self.df.columns:
            # Convert cancer type to numeric labels
            label_encoder = LabelEncoder()
            self.df['cancer_type_encoded'] = label_encoder.fit_transform(self.df['cancer_type'])
            self.class_names = label_encoder.classes_
            target_col = 'cancer_type_encoded'
        else:
            raise ValueError("No cancer_type column found for classification")
        
        # Select feature columns (exclude metadata)
        exclude_cols = ['sample_id', 'cancer_type', 'cancer_type_encoded']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        logger.info(f"🎯 Feature columns: {len(feature_cols)}")
        
        # Prepare features and target
        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()
        
        # Handle missing values
        logger.info("🔄 Handling missing values...")
        if X.isnull().sum().sum() > 0:
            # Use KNN imputer for better imputation
            imputer = KNNImputer(n_neighbors=5)
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
            logger.info("✅ Missing values imputed using KNN")
        else:
            X_imputed = X.copy()
        
        # Feature scaling (robust to outliers)
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        # Create multi-omics interaction features
        logger.info("🔗 Creating multi-omics interaction features...")
        interaction_features = []
        
        # Identify omics types
        cn_features = [col for col in X_scaled.columns if col.startswith('cn_')]
        protein_features = [col for col in X_scaled.columns if col.startswith('protein_')]
        
        if cn_features and protein_features:
            # Create interaction features between copy number and protein
            for i, cn_feat in enumerate(cn_features[:10]):  # Limit to top 10 to avoid explosion
                for j, prot_feat in enumerate(protein_features[:10]):
                    interaction_name = f"interact_{cn_feat}_{prot_feat}"
                    X_scaled[interaction_name] = X_scaled[cn_feat] * X_scaled[prot_feat]
                    interaction_features.append(interaction_name)
                    
                    if len(interaction_features) >= 50:  # Limit total interactions
                        break
                if len(interaction_features) >= 50:
                    break
            
            logger.info(f"✅ Created {len(interaction_features)} interaction features")
        
        # Feature selection
        logger.info("🎯 Performing feature selection...")
        
        # Statistical feature selection
        selector = SelectKBest(score_func=f_classif, k=min(100, len(X_scaled.columns)))
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = X_scaled.columns[selector.get_support()].tolist()
        
        X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_scaled.index)
        
        logger.info(f"✅ Selected {len(selected_features)} features")
        logger.info(f"  Copy Number features: {sum(1 for f in selected_features if f.startswith('cn_'))}")
        logger.info(f"  Protein features: {sum(1 for f in selected_features if f.startswith('protein_'))}")
        logger.info(f"  Interaction features: {sum(1 for f in selected_features if f.startswith('interact_'))}")
        
        # Store processed data
        self.feature_names = selected_features
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_final, y, 
            test_size=self.test_size, 
            stratify=y,
            random_state=self.random_state
        )
        
        logger.info(f"📊 Data split complete:")
        logger.info(f"  Training: {len(self.X_train):,} samples")
        logger.info(f"  Testing: {len(self.X_test):,} samples")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Classes: {len(self.class_names)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple advanced models"""
        logger.info("🤖 Training advanced machine learning models...")
        
        # Handle class imbalance with SMOTE
        logger.info("⚖️ Handling class imbalance...")
        smote = SMOTE(random_state=self.random_state, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
        
        logger.info(f"  Original training samples: {len(self.X_train):,}")
        logger.info(f"  Balanced training samples: {len(X_train_balanced):,}")
        
        # Model configurations
        models_config = {
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                    objective='multiclass',
                    boosting_type='gbdt',
                    num_leaves=31,
                    learning_rate=0.05,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    multi_class='ovr'
                ),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        # Train models with grid search
        best_scores = {}
        
        for model_name, config in models_config.items():
            logger.info(f"🔄 Training {model_name}...")
            
            try:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_balanced, y_train_balanced)
                
                # Store best model
                self.models[model_name] = grid_search.best_estimator_
                best_scores[model_name] = grid_search.best_score_
                
                logger.info(f"  ✅ {model_name}: CV Score = {grid_search.best_score_:.4f}")
                logger.info(f"     Best params: {grid_search.best_params_}")
                
            except Exception as e:
                logger.error(f"  ❌ {model_name} training failed: {e}")
                continue
        
        # Select best model
        if best_scores:
            best_model_name = max(best_scores, key=best_scores.get)
            self.best_model = self.models[best_model_name]
            
            logger.info(f"🏆 Best model: {best_model_name} (CV Score: {best_scores[best_model_name]:.4f})")
        else:
            logger.error("❌ No models trained successfully")
            return None
        
        return self.models
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        logger.info("📈 Evaluating models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"🔍 Evaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted', zero_division=0
            )
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"  📊 {model_name} Results:")
            logger.info(f"     Accuracy:  {accuracy:.4f}")
            logger.info(f"     Precision: {precision:.4f}")
            logger.info(f"     Recall:    {recall:.4f}")
            logger.info(f"     F1-Score:  {f1:.4f}")
        
        # Feature importance analysis
        self.analyze_feature_importance()
        
        # Generate detailed classification report for best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        
        logger.info(f"🎯 Detailed results for best model ({best_model_name}):")
        report = classification_report(
            self.y_test, 
            best_predictions,
            target_names=[f"TCGA-{cls}" for cls in self.class_names],
            zero_division=0
        )
        logger.info(f"\n{report}")
        
        return results
    
    def analyze_feature_importance(self):
        """Analyze feature importance across models"""
        logger.info("🔍 Analyzing feature importance...")
        
        importance_data = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                continue
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_data[model_name] = importance_df
            
            logger.info(f"🎯 Top 10 features for {model_name}:")
            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.feature_importance = importance_data
        return importance_data
    
    def save_results(self, results: Dict):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model performance
        performance_summary = {}
        for model_name, metrics in results.items():
            performance_summary[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score'])
            }
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.df),
                'training_samples': len(self.X_train),
                'testing_samples': len(self.X_test),
                'features': len(self.feature_names),
                'cancer_types': len(self.class_names)
            },
            'model_performance': performance_summary,
            'best_model': max(performance_summary, key=lambda x: performance_summary[x]['accuracy']),
            'feature_count': {
                'copy_number': sum(1 for f in self.feature_names if f.startswith('cn_')),
                'protein': sum(1 for f in self.feature_names if f.startswith('protein_')),
                'interactions': sum(1 for f in self.feature_names if f.startswith('interact_'))
            }
        }
        
        # Save files
        summary_file = self.output_dir / f"training_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save feature importance
        if self.feature_importance:
            for model_name, importance_df in self.feature_importance.items():
                importance_file = self.output_dir / f"feature_importance_{model_name}_{timestamp}.csv"
                importance_df.to_csv(importance_file, index=False)
        
        logger.info(f"💾 Results saved:")
        logger.info(f"  📋 Summary: {summary_file}")
        logger.info(f"  🎯 Feature importance: {len(self.feature_importance)} files")
        
        return summary_file
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("🚀 Starting Advanced Multi-Omics Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Feature engineering
            self.feature_engineering()
            
            # Step 3: Train models
            models = self.train_models()
            if not models:
                return None
            
            # Step 4: Evaluate models
            results = self.evaluate_models()
            
            # Step 5: Save results
            summary_file = self.save_results(results)
            
            logger.info("✅ SUCCESS: Advanced multi-omics training completed!")
            logger.info(f"📂 Results: {summary_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR: {e}")
            raise


def main():
    """Main execution"""
    # Find the most recent dataset
    dataset_dir = Path("data/focused_multi_omics")
    if not dataset_dir.exists():
        logger.error("❌ Dataset directory not found")
        return None
    
    # Find latest dataset file
    dataset_files = list(dataset_dir.glob("tcga_focused_multi_omics_*.csv"))
    if not dataset_files:
        logger.error("❌ No dataset files found")
        return None
    
    latest_dataset = max(dataset_files, key=lambda x: x.stat().st_mtime)
    
    logger.info("🎓 Advanced Multi-Omics TCGA Machine Learning")
    logger.info("=" * 50)
    logger.info(f"📂 Dataset: {latest_dataset}")
    
    # Initialize trainer
    trainer = AdvancedMultiOmicsTrainer(latest_dataset)
    
    # Run pipeline
    results = trainer.run_complete_pipeline()
    
    if results:
        logger.info("🎉 Training pipeline completed successfully!")
        return trainer
    else:
        logger.error("❌ Training pipeline failed")
        return None


if __name__ == "__main__":
    trainer = main()
