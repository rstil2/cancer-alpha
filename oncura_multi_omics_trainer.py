#!/usr/bin/env python3
"""
🧬 ONCURA - Next-Generation Cancer Genomics Platform
🔬 Advanced Multi-Omics TCGA Trainer

High-performance multi-omics feature extraction and model training for 
9,000+ TCGA samples across multiple cancer types with integrated ML pipeline.
"""

import os
import pandas as pd
import numpy as np
import logging
import argparse
import joblib
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
import xgboost as xgb

# Progress and resource monitoring
from tqdm import tqdm
import psutil
import gzip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oncura_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OncuraTrainer')

class OncuraMultiOmicsTrainer:
    """Advanced Multi-Omics Feature Extraction and Model Training"""
    
    def __init__(self, integration_file: str, output_dir: str = "data/oncura_models"):
        self.integration_file = integration_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Feature extraction settings
        self.max_mutation_features = 1000  # Top mutation features
        self.max_protein_features = 500    # Top protein features
        self.max_copy_features = 300       # Top copy number features
        
        # Model registry
        self.models = {
            'LightGBM': lgb.LGBMClassifier(
                objective='multiclass',
                num_class=None,  # Will be set dynamically
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        }
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def extract_mutation_features(self, maf_file: str, max_features: int = 50) -> dict:
        """Extract mutation features from MAF file"""
        features = defaultdict(int)
        
        try:
            # Open MAF file (handle .gz)
            open_func = gzip.open if maf_file.endswith('.gz') else open
            mode = 'rt' if maf_file.endswith('.gz') else 'r'
            
            with open_func(maf_file, mode) as f:
                header = next(f).strip().split('\t')
                
                # Find relevant columns
                gene_col = next((i for i, col in enumerate(header) if 'Hugo_Symbol' in col), 0)
                variant_col = next((i for i, col in enumerate(header) if 'Variant_Classification' in col), 8)
                
                mutation_count = 0
                for line in f:
                    if mutation_count >= 5000:  # Limit processing per file
                        break
                        
                    parts = line.strip().split('\t')
                    if len(parts) > max(gene_col, variant_col):
                        gene = parts[gene_col]
                        variant_type = parts[variant_col]
                        
                        # Create feature names
                        if gene and variant_type:
                            features[f'mut_{gene}'] += 1
                            features[f'var_{variant_type}'] += 1
                        
                        mutation_count += 1
                        
        except Exception as e:
            logger.warning(f"Failed to extract mutations from {maf_file}: {e}")
            
        # Return top features
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:max_features])
    
    def extract_protein_features(self, protein_file: str) -> dict:
        """Extract protein expression features"""
        features = {}
        
        try:
            df = pd.read_csv(protein_file, sep='\t', index_col=0, nrows=500)
            
            # Take numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:50]:  # Top 50 proteins
                if len(df[col].dropna()) > 0:
                    features[f'prot_{col}'] = df[col].mean()
                    
        except Exception as e:
            logger.warning(f"Failed to extract proteins from {protein_file}: {e}")
            
        return features
    
    def extract_copy_number_features(self, cnv_file: str) -> dict:
        """Extract copy number variation features"""
        features = {}
        
        try:
            df = pd.read_csv(cnv_file, sep='\t', nrows=1000)
            
            # Process copy number data
            if 'Gene Symbol' in df.columns or 'gene' in df.columns.str.lower():
                gene_col = next((col for col in df.columns if 'gene' in col.lower()), df.columns[0])
                
                for _, row in df.head(30).iterrows():
                    gene = str(row[gene_col])
                    # Look for copy number values
                    for col in df.columns[1:]:
                        if col != gene_col and pd.api.types.is_numeric_dtype(df[col]):
                            value = row[col]
                            if not pd.isna(value):
                                features[f'cnv_{gene}_{col}'] = float(value)
                                break
                                
        except Exception as e:
            logger.warning(f"Failed to extract copy numbers from {cnv_file}: {e}")
            
        return features
    
    def build_feature_matrix(self, df: pd.DataFrame, max_samples: int = 8000) -> tuple:
        """Build comprehensive feature matrix from multi-omics data"""
        logger.info(f"🔬 Building feature matrix for {len(df)} samples")
        
        feature_data = []
        labels = []
        sample_ids = []
        
        # Process samples
        processed = 0
        for _, row in tqdm(df.iterrows(), total=min(len(df), max_samples), desc="Extracting features"):
            if processed >= max_samples:
                break
                
            sample_features = {
                'sample_id': row['sample_id'],
                'cancer_type': row['cancer_type'],
                'num_omics_types': row['num_omics_types'],
                'total_files': row['total_files']
            }
            
            # Extract mutation features
            if row['has_mutations'] and row['mutations_file']:
                mut_features = self.extract_mutation_features(row['mutations_file'])
                sample_features.update(mut_features)
            
            # Extract protein features
            if row['has_protein'] and row['protein_file']:
                prot_features = self.extract_protein_features(row['protein_file'])
                sample_features.update(prot_features)
            
            # Extract copy number features
            if row['has_copy_number'] and row['copy_number_file']:
                cnv_features = self.extract_copy_number_features(row['copy_number_file'])
                sample_features.update(cnv_features)
            
            feature_data.append(sample_features)
            labels.append(row['cancer_type'])
            sample_ids.append(row['sample_id'])
            processed += 1
            
            # Memory management
            if processed % 1000 == 0:
                logger.info(f"⚡ Processed {processed} samples. Memory: {psutil.virtual_memory().percent:.1f}%")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_data)
        features_df = features_df.fillna(0)
        
        logger.info(f"✅ Feature matrix: {features_df.shape}")
        logger.info(f"📊 Feature types: {features_df.columns[:10].tolist()}...")
        
        return features_df, labels, sample_ids
    
    def train_models(self, X: pd.DataFrame, y: list) -> dict:
        """Train multiple ML models"""
        logger.info(f"🚀 Training models on {X.shape} dataset")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Remove non-numeric columns
        feature_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[feature_cols]
        
        # Feature selection
        if len(feature_cols) > self.max_mutation_features:
            selector = SelectKBest(f_classif, k=self.max_mutation_features)
            X_selected = selector.fit_transform(X_numeric, y_encoded)
            selected_features = feature_cols[selector.get_support()]
            logger.info(f"🎯 Selected {len(selected_features)} top features")
        else:
            X_selected = X_numeric
            selected_features = feature_cols
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"📊 Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train models
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"🔥 Training {name}...")
            
            try:
                # Set num_class for LightGBM
                if name == 'LightGBM':
                    model.num_class = len(np.unique(y_encoded))
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Evaluate
                accuracy = (y_pred == y_test).mean()
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                
                # AUC for multiclass
                if y_proba is not None and len(np.unique(y_encoded)) > 2:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                else:
                    auc = 0.0
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'auc': auc,
                    'predictions': y_pred,
                    'true_labels': y_test
                }
                
                logger.info(f"✅ {name}: Accuracy={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {name}: {e}")
                continue
        
        return results, selected_features
    
    def save_models_and_results(self, results: dict, features: list) -> None:
        """Save trained models and results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_model = results[best_model_name]['model']
        
        model_path = self.output_dir / f'oncura_best_model_{timestamp}.pkl'
        joblib.dump({
            'model': best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'features': features,
            'model_name': best_model_name
        }, model_path)
        
        logger.info(f"💾 Best model ({best_model_name}) saved: {model_path}")
        
        # Save results summary
        summary = []
        for name, result in results.items():
            summary.append({
                'model': name,
                'accuracy': result['accuracy'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'auc': result['auc']
            })
        
        summary_df = pd.DataFrame(summary).sort_values('accuracy', ascending=False)
        summary_path = self.output_dir / f'oncura_results_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"📊 Results summary saved: {summary_path}")
        print("\n🏆 Model Performance Summary:")
        print(summary_df.to_string(index=False, float_format='%.3f'))

def main():
    parser = argparse.ArgumentParser(description="Oncura Multi-Omics Trainer")
    parser.add_argument('--integration-file', type=str, required=True,
                       help='Path to integration CSV file')
    parser.add_argument('--max-samples', type=int, default=8000,
                       help='Maximum samples to process')
    parser.add_argument('--output-dir', type=str, default='data/oncura_models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    print("🧬 ONCURA - Next-Generation Cancer Genomics Platform")
    print("🔬 Advanced Multi-Omics TCGA Trainer")
    print("=" * 60)
    
    # Load integration data
    logger.info(f"📂 Loading integration data: {args.integration_file}")
    df = pd.read_csv(args.integration_file)
    logger.info(f"📊 Loaded {len(df)} samples across {df['cancer_type'].nunique()} cancer types")
    
    # Initialize trainer
    trainer = OncuraMultiOmicsTrainer(args.integration_file, args.output_dir)
    
    # Build feature matrix
    features_df, labels, sample_ids = trainer.build_feature_matrix(df, args.max_samples)
    
    # Train models
    results, selected_features = trainer.train_models(features_df, labels)
    
    # Save results
    trainer.save_models_and_results(results, selected_features)
    
    print(f"\n✅ Training complete!")
    print(f"🎯 Processed {len(features_df)} samples")
    print(f"🧬 Features: {len(selected_features)}")
    print(f"🎪 Cancer types: {len(set(labels))}")
    print(f"💾 Models saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
