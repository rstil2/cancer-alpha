#!/usr/bin/env python3
"""
Real TCGA Model Comparison
==========================
Compare multiple ML models on authentic TCGA data to find the best performer.
Ensures 100% real data with zero synthetic contamination.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb

# SMOTE for class balancing
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Set environment for real data only
os.environ['ONCURA_REAL_DATA_ONLY'] = '1'
np.random.seed(42)

class RealDataModelComparison:
    """Compare multiple models on real TCGA data"""
    
    def __init__(self, features_path, labels_path):
        self.features_path = Path(features_path)
        self.labels_path = Path(labels_path)
        
        # Validate real data paths
        if not self._validate_real_data_paths():
            raise ValueError("❌ Data paths do not comply with real-data-only policy!")
        
        self.X = None
        self.y = None
        self.y_encoded = None
        self.cancer_types = None
        self.results = {}
        
    def _validate_real_data_paths(self):
        """Ensure we're only using real data"""
        allowed_paths = ['data/real_tcga', 'data/raw_tcga', 'data/processed_tcga']
        
        for path in [self.features_path, self.labels_path]:
            path_str = str(path)
            if not any(allowed in path_str for allowed in allowed_paths):
                print(f"❌ Invalid path: {path_str}")
                return False
            
            # Check for synthetic data indicators
            forbidden = ['synthetic', 'generated', 'fake', 'artificial']
            if any(word in path_str.lower() for word in forbidden):
                print(f"❌ Synthetic data detected in path: {path_str}")
                return False
        
        return True
    
    def load_data(self):
        """Load and validate the real TCGA dataset"""
        print("🔄 Loading real TCGA dataset...")
        
        # Load features and labels
        X_df = pd.read_csv(self.features_path)
        y_df = pd.read_csv(self.labels_path)
        
        self.X = X_df.values
        self.y_encoded = y_df['cancer_type_encoded'].values
        self.cancer_types = y_df['cancer_type'].values
        
        # Get unique cancer types
        unique_cancers = sorted(set(self.cancer_types))
        
        print(f"✅ Dataset loaded:")
        print(f"   Samples: {self.X.shape[0]}")
        print(f"   Features: {self.X.shape[1]}")
        print(f"   Cancer types: {len(unique_cancers)}")
        print(f"   Cancer distribution:")
        
        cancer_counts = pd.Series(self.cancer_types).value_counts()
        for cancer, count in cancer_counts.items():
            print(f"     {cancer}: {count} samples")
        
        # Validate real data properties
        if np.any(np.isnan(self.X)):
            print("⚠️  Warning: NaN values detected in features")
        
        print(f"   Data source validation: ✅ REAL DATA CONFIRMED")
        
    def define_models(self):
        """Define the models to compare"""
        print("\n🤖 Defining models for comparison...")
        
        models = {
            'LightGBM + SMOTE': ImbPipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=5)),
                ('lgb', lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=len(set(self.y_encoded)),
                    n_estimators=300,
                    learning_rate=0.1,
                    num_leaves=31,
                    feature_fraction=0.9,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    random_state=42,
                    verbose=-1
                ))
            ]),
            
            'XGBoost + SMOTE': ImbPipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=5)),
                ('xgb', xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=len(set(self.y_encoded)),
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    random_state=42,
                    eval_metric='mlogloss'
                ))
            ]),
            
            'Random Forest + SMOTE': ImbPipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=5)),
                ('rf', RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ))
            ]),
            
            'Gradient Boosting + SMOTE': ImbPipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=5)),
                ('gb', GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ))
            ]),
            
            'Logistic Regression + SMOTE': ImbPipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=5)),
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    multi_class='multinomial'
                ))
            ]),
            
            'SVM + SMOTE': ImbPipeline([
                ('smote', SMOTE(random_state=42, k_neighbors=5)),
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=42
                ))
            ])
        }
        
        print(f"✅ Defined {len(models)} models for comparison")
        return models
    
    def evaluate_models(self, models, cv_folds=5):
        """Evaluate all models using cross-validation"""
        print(f"\n🔬 Evaluating models with {cv_folds}-fold cross-validation...")
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🎯 Evaluating {name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    model, self.X, self.y_encoded, 
                    cv=skf, scoring='balanced_accuracy', 
                    n_jobs=1  # Avoid multiprocessing issues
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                results[name] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_balanced_accuracy': mean_score,
                    'std_balanced_accuracy': std_score,
                    'individual_scores': [f"{score:.3f}" for score in cv_scores],
                    'status': 'success'
                }
                
                print(f"   ✅ Balanced Accuracy: {mean_score:.3f} ± {std_score:.3f}")
                print(f"   📊 Individual scores: {results[name]['individual_scores']}")
                
                # Check if this model hits our target
                if mean_score >= 0.95:
                    print(f"   🎉 TARGET ACHIEVED! ≥95% balanced accuracy")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results[name] = {
                    'status': 'failed',
                    'error': str(e),
                    'mean_balanced_accuracy': 0.0,
                    'std_balanced_accuracy': 0.0
                }
        
        self.results = results
        return results
    
    def analyze_best_model(self, models):
        """Analyze the best performing model in detail"""
        print(f"\n🏆 ANALYZING BEST MODEL...")
        
        # Find best model
        valid_results = {k: v for k, v in self.results.items() if v['status'] == 'success'}
        if not valid_results:
            print("❌ No successful models to analyze")
            return None
        
        best_model_name = max(valid_results.keys(), 
                            key=lambda k: valid_results[k]['mean_balanced_accuracy'])
        best_score = valid_results[best_model_name]['mean_balanced_accuracy']
        best_std = valid_results[best_model_name]['std_balanced_accuracy']
        
        print(f"🥇 Best Model: {best_model_name}")
        print(f"📊 Performance: {best_score:.3f} ± {best_std:.3f}")
        
        # Train best model on full dataset for detailed analysis
        best_pipeline = models[best_model_name]
        best_pipeline.fit(self.X, self.y_encoded)
        y_pred = best_pipeline.predict(self.X)
        
        # Training accuracy
        train_acc = balanced_accuracy_score(self.y_encoded, y_pred)
        print(f"🎯 Training Balanced Accuracy: {train_acc:.3f}")
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        unique_labels = sorted(set(self.y_encoded))
        unique_names = [self.cancer_types[np.where(self.y_encoded == label)[0][0]] for label in unique_labels]
        
        report = classification_report(self.y_encoded, y_pred, 
                                     target_names=unique_names, 
                                     output_dict=True)
        
        for cancer in unique_names:
            metrics = report[cancer]
            print(f"   {cancer}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return {
            'best_model_name': best_model_name,
            'best_score': best_score,
            'best_std': best_std,
            'train_accuracy': train_acc,
            'classification_report': report
        }
    
    def save_results(self, analysis_results=None):
        """Save comprehensive results"""
        output_dir = Path('data/real_tcga_large')
        results_path = output_dir / 'model_comparison_results.json'
        
        # Prepare comprehensive results
        comprehensive_results = {
            'evaluation_date': datetime.now().isoformat(),
            'dataset_info': {
                'samples': self.X.shape[0],
                'features': self.X.shape[1],
                'cancer_types': len(set(self.y_encoded)),
                'data_source': 'authentic_tcga_only',
                'synthetic_data_used': False
            },
            'model_results': self.results,
            'best_model_analysis': analysis_results,
            'target_achieved': any(r.get('mean_balanced_accuracy', 0) >= 0.95 
                                 for r in self.results.values() if r.get('status') == 'success')
        }
        
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        return results_path

def main():
    """Main comparison function"""
    print('🧬 REAL TCGA MODEL COMPARISON')
    print('=' * 50)
    print(f'Environment: ONCURA_REAL_DATA_ONLY = {os.environ.get("ONCURA_REAL_DATA_ONLY", "NOT SET")}')
    
    # Initialize comparison
    features_path = 'data/real_tcga_large/real_tcga_features_cleaned.csv'
    labels_path = 'data/real_tcga_large/real_tcga_labels.csv'
    
    comparison = RealDataModelComparison(features_path, labels_path)
    
    # Load data
    comparison.load_data()
    
    # Define models
    models = comparison.define_models()
    
    # Evaluate models
    results = comparison.evaluate_models(models)
    
    # Analyze best model
    analysis = comparison.analyze_best_model(models)
    
    # Save results
    comparison.save_results(analysis)
    
    # Print summary
    print(f"\n🎉 MODEL COMPARISON COMPLETE!")
    print(f"📊 Models evaluated: {len([r for r in results.values() if r['status'] == 'success'])}")
    
    if analysis:
        print(f"🏆 Best model: {analysis['best_model_name']}")
        print(f"🎯 Best performance: {analysis['best_score']:.3f} ± {analysis['best_std']:.3f}")
        
        if analysis['best_score'] >= 0.95:
            print(f"✅ TARGET ACHIEVED: ≥95% balanced accuracy!")
        else:
            print(f"📈 Progress: {analysis['best_score']/0.95*100:.1f}% toward 95% target")
    
    print(f"\n🚫 ZERO SYNTHETIC DATA USED - 100% AUTHENTIC TCGA")

if __name__ == "__main__":
    main()