#!/usr/bin/env python3
"""
STRICTLY Real TCGA Model Comparison - ZERO SYNTHETIC DATA
=========================================================
Compare ML models on authentic TCGA data with ZERO synthetic data generation.
No SMOTE, no interpolation, no artificial samples - only real patient data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports - NO imblearn/SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

# Set environment for real data only
os.environ['ONCURA_REAL_DATA_ONLY'] = '1'
np.random.seed(42)

def main():
    print('🧬 STRICTLY REAL TCGA MODEL COMPARISON')
    print('🚫 ZERO SYNTHETIC DATA - NO SMOTE - NO INTERPOLATION')
    print('=' * 60)
    
    # Load the real data
    features_path = 'data/real_tcga_large/real_tcga_features_cleaned.csv'
    labels_path = 'data/real_tcga_large/real_tcga_labels.csv'
    
    print(f'📁 Loading data from:')
    print(f'   Features: {features_path}')
    print(f'   Labels: {labels_path}')
    
    X_df = pd.read_csv(features_path)
    y_df = pd.read_csv(labels_path)
    
    X = X_df.values
    y_encoded = y_df['cancer_type_encoded'].values
    cancer_types = y_df['cancer_type'].values
    
    print(f'✅ Dataset loaded:')
    print(f'   Samples: {X.shape[0]}')
    print(f'   Features: {X.shape[1]}')
    print(f'   Cancer types: {len(set(cancer_types))}')
    
    # Show class distribution
    print(f'\\n📊 Class distribution (perfectly balanced real data):')
    cancer_counts = pd.Series(cancer_types).value_counts()
    for cancer, count in cancer_counts.items():
        print(f'   {cancer}: {count} samples')
    
    # Define models WITHOUT SMOTE - using class weights for imbalance
    models = {
        'LightGBM (Real Only)': lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(set(y_encoded)),
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            class_weight='balanced',  # Handle imbalance without synthetic data
            random_state=42,
            verbose=-1
        ),
        
        'XGBoost (Real Only)': xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(set(y_encoded)),
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='mlogloss'
        ),
        
        'Random Forest (Real Only)': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        
        'Gradient Boosting (Real Only)': GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        
        'Logistic Regression (Real Only)': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial',
                class_weight='balanced'
            ))
        ]),
        
        'SVM (Real Only)': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                random_state=42
            ))
        ])
    }
    
    print(f'\\n🤖 Evaluating {len(models)} models with STRICTLY REAL DATA...')
    print('⚠️  Note: No SMOTE, no interpolation, no synthetic samples')
    
    # Cross-validation results
    cv_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f'\\n🎯 Evaluating {name}...')
        
        try:
            cv_scores = cross_val_score(
                model, X, y_encoded, 
                cv=skf, scoring='balanced_accuracy',
                n_jobs=1
            )
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            cv_results[name] = {
                'mean_balanced_accuracy': mean_score,
                'std_balanced_accuracy': std_score,
                'individual_scores': [f'{score:.3f}' for score in cv_scores],
                'status': 'success'
            }
            
            print(f'   ✅ Balanced Accuracy: {mean_score:.3f} ± {std_score:.3f}')
            print(f'   📊 Individual scores: {cv_results[name]["individual_scores"]}')
            
            if mean_score >= 0.95:
                print(f'   🎉 TARGET ACHIEVED! ≥95% on REAL DATA ONLY')
            
        except Exception as e:
            print(f'   ❌ Error: {str(e)}')
            cv_results[name] = {
                'status': 'failed',
                'error': str(e),
                'mean_balanced_accuracy': 0.0
            }
    
    # Find best model
    successful_models = {k: v for k, v in cv_results.items() if v['status'] == 'success'}
    
    if successful_models:
        best_model_name = max(successful_models.keys(),
                            key=lambda k: successful_models[k]['mean_balanced_accuracy'])
        best_score = successful_models[best_model_name]['mean_balanced_accuracy']
        best_std = successful_models[best_model_name]['std_balanced_accuracy']
        
        print(f'\\n🏆 BEST MODEL ON STRICTLY REAL DATA:')
        print(f'   🥇 Model: {best_model_name}')
        print(f'   📊 Performance: {best_score:.3f} ± {best_std:.3f}')
        
        if best_score >= 0.95:
            print(f'   ✅ TARGET ACHIEVED: ≥95% with ZERO synthetic data!')
        else:
            print(f'   📈 Progress: {best_score/0.95*100:.1f}% toward 95% target')
    
    # Save results
    output_dir = Path('data/real_tcga_large')
    results_path = output_dir / 'model_comparison_real_only.json'
    
    comprehensive_results = {
        'evaluation_date': datetime.now().isoformat(),
        'dataset_info': {
            'samples': X.shape[0],
            'features': X.shape[1],
            'cancer_types': len(set(y_encoded)),
            'data_source': 'authentic_tcga_only',
            'synthetic_data_used': False,
            'smote_used': False,
            'interpolation_used': False,
            'strictly_real_only': True
        },
        'model_results': cv_results,
        'best_model': {
            'name': best_model_name if successful_models else None,
            'score': best_score if successful_models else 0,
            'std': best_std if successful_models else 0
        },
        'target_achieved': best_score >= 0.95 if successful_models else False
    }
    
    with open(results_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f'\\n💾 Results saved to: {results_path}')
    print(f'\\n🎯 SUMMARY:')
    print(f'   📊 Models evaluated: {len(successful_models)}')
    if successful_models:
        print(f'   🏆 Best performance: {best_score:.3f} ± {best_std:.3f}')
        print(f'   🚫 ZERO synthetic data used')
        print(f'   ✅ 100% authentic TCGA patient data')
    
if __name__ == "__main__":
    main()