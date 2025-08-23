#!/usr/bin/env python3
"""
ADVANCED 50K DATASET FEATURE ENGINEERING & PREPROCESSING
========================================================
Comprehensive preprocessing pipeline for the 50,000 sample TCGA dataset
- Feature engineering and encoding
- Multi-omics integration features
- Cancer type encoding strategies
- Quality-based sample weighting
- Stratified dataset splits
- Feature scaling and normalization

Prepares ML-ready datasets for various modeling approaches
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import logging
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

class Advanced50kPreprocessor:
    def __init__(self, dataset_path):
        self.logger = self.setup_logging()
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path("data/50k_preprocessing_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.class_weights = {}
        
        # Processed datasets
        self.processed_datasets = {}
        self.preprocessing_stats = {
            'original_shape': None,
            'processed_shapes': {},
            'feature_engineering': {},
            'encoding_mappings': {},
            'scaling_parameters': {},
            'split_statistics': {}
        }
        
        # Load the dataset
        self.df = None
        self.load_dataset()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_dataset(self):
        """Load the 50k dataset"""
        self.logger.info(f"📊 Loading dataset from {self.dataset_path}")
        
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.preprocessing_stats['original_shape'] = self.df.shape
            self.logger.info(f"✅ Dataset loaded: {self.df.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset: {e}")
            raise

    def engineer_features(self):
        """Engineer new features from existing data"""
        self.logger.info("🔧 Engineering new features...")
        
        df_engineered = self.df.copy()
        
        # 1. Multi-omics Integration Features
        omics_cols = ['has_expression', 'has_methylation', 'has_copy_number', 
                     'has_mutations', 'has_protein', 'has_clinical']
        
        # Multi-omics completeness score
        df_engineered['omics_completeness_score'] = df_engineered[omics_cols].sum(axis=1) / len(omics_cols)
        
        # Multi-omics diversity index (Shannon entropy-like measure)
        def calculate_diversity_index(row):
            present_omics = row[omics_cols].sum()
            if present_omics == 0:
                return 0
            # Simple diversity measure
            return present_omics * (1 - abs(present_omics - 3) / 3)  # Peak at 3 omics types
        
        df_engineered['omics_diversity_index'] = df_engineered.apply(calculate_diversity_index, axis=1)
        
        # 2. Cancer Type Hierarchical Features
        # Extract cancer family/category from TCGA codes
        def extract_cancer_family(cancer_type):
            cancer_families = {
                'BRCA': 'Breast', 'COAD': 'Colorectal', 'READ': 'Colorectal',
                'LUAD': 'Lung', 'LUSC': 'Lung', 'KIRC': 'Kidney', 'KIRP': 'Kidney', 'KICH': 'Kidney',
                'PRAD': 'Prostate', 'THCA': 'Thyroid', 'LIHC': 'Liver', 'BLCA': 'Bladder',
                'CESC': 'Cervical', 'HNSC': 'Head/Neck', 'STAD': 'Stomach', 'UCEC': 'Endometrial',
                'OV': 'Ovarian', 'GBM': 'Brain', 'LGG': 'Brain', 'LAML': 'Leukemia',
                'SKCM': 'Skin', 'PAAD': 'Pancreatic', 'ESCA': 'Esophageal', 'PCPG': 'Adrenal',
                'SARC': 'Sarcoma', 'ACC': 'Adrenal', 'MESO': 'Mesothelioma', 'UCS': 'Uterine',
                'CHOL': 'Bile_duct', 'THYM': 'Thymus', 'TGCT': 'Testicular', 'UVM': 'Eye', 'DLBC': 'Lymphoma'
            }
            
            if isinstance(cancer_type, str) and cancer_type.startswith('TCGA-'):
                cancer_code = cancer_type.split('-')[1]
                return cancer_families.get(cancer_code, 'Other')
            return 'Other'
        
        df_engineered['cancer_family'] = df_engineered['cancer_type'].apply(extract_cancer_family)
        
        # 3. Sample Quality Features
        # Normalized quality score
        df_engineered['quality_score_normalized'] = (
            df_engineered['quality_score'] - df_engineered['quality_score'].min()
        ) / (df_engineered['quality_score'].max() - df_engineered['quality_score'].min())
        
        # Quality tier classification
        quality_percentiles = df_engineered['quality_score'].quantile([0.33, 0.67])
        def quality_tier(score):
            if score <= quality_percentiles.iloc[0]:
                return 'Low'
            elif score <= quality_percentiles.iloc[1]:
                return 'Medium'
            else:
                return 'High'
        
        df_engineered['quality_tier'] = df_engineered['quality_score'].apply(quality_tier)
        
        # 4. Data Richness Features
        # File density per MB
        df_engineered['file_density'] = df_engineered['num_files'] / (df_engineered['total_size_mb'] + 1e-6)
        
        # Average file size
        df_engineered['avg_file_size_mb'] = df_engineered['total_size_mb'] / (df_engineered['num_files'] + 1e-6)
        
        # 5. Multi-omics Combination Features (One-hot encoding of common combinations)
        top_combinations = df_engineered['data_types'].value_counts().head(10).index
        for i, combo in enumerate(top_combinations):
            df_engineered[f'omics_combo_{i+1}'] = (df_engineered['data_types'] == combo).astype(int)
        
        # 6. Cancer Type Frequency Features
        cancer_frequencies = df_engineered['cancer_type'].value_counts()
        df_engineered['cancer_type_frequency'] = df_engineered['cancer_type'].map(cancer_frequencies)
        df_engineered['cancer_type_rarity'] = 1 / (df_engineered['cancer_type_frequency'] + 1)
        
        # Log feature engineering stats
        new_features = set(df_engineered.columns) - set(self.df.columns)
        self.preprocessing_stats['feature_engineering'] = {
            'new_features_count': len(new_features),
            'new_features': list(new_features),
            'original_features': len(self.df.columns),
            'total_features': len(df_engineered.columns)
        }
        
        self.logger.info(f"✅ Engineered {len(new_features)} new features")
        self.logger.info(f"   Total features: {len(self.df.columns)} → {len(df_engineered.columns)}")
        
        return df_engineered

    def encode_categorical_features(self, df):
        """Encode categorical features using multiple strategies"""
        self.logger.info("🏷️ Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # 1. Cancer Type Encoding (Multiple strategies for different use cases)
        
        # Label Encoding for tree-based models
        le_cancer = LabelEncoder()
        df_encoded['cancer_type_label'] = le_cancer.fit_transform(df_encoded['cancer_type'])
        self.encoders['cancer_type_label'] = le_cancer
        
        # Ordinal Encoding by frequency (for neural networks)
        cancer_freq_order = df['cancer_type'].value_counts().index.tolist()
        oe_cancer = OrdinalEncoder(categories=[cancer_freq_order])
        df_encoded['cancer_type_ordinal'] = oe_cancer.fit_transform(df_encoded[['cancer_type']]).flatten()
        self.encoders['cancer_type_ordinal'] = oe_cancer
        
        # One-hot encoding for top cancer types (to avoid too many features)
        top_10_cancers = df['cancer_type'].value_counts().head(10).index
        df_encoded['cancer_type_top10'] = df_encoded['cancer_type'].apply(
            lambda x: x if x in top_10_cancers else 'Other'
        )
        
        # One-hot encode the top 10 + Other
        ohe_cancer = OneHotEncoder(sparse_output=False, drop='first')
        cancer_onehot = ohe_cancer.fit_transform(df_encoded[['cancer_type_top10']])
        cancer_onehot_cols = [f'cancer_type_{cat}' for cat in ohe_cancer.categories_[0][1:]]  # Skip first due to drop='first'
        
        for i, col in enumerate(cancer_onehot_cols):
            df_encoded[col] = cancer_onehot[:, i]
        
        self.encoders['cancer_type_onehot'] = ohe_cancer
        
        # 2. Cancer Family Encoding
        le_family = LabelEncoder()
        df_encoded['cancer_family_label'] = le_family.fit_transform(df_encoded['cancer_family'])
        self.encoders['cancer_family_label'] = le_family
        
        # 3. Quality Tier Encoding
        quality_tier_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df_encoded['quality_tier_ordinal'] = df_encoded['quality_tier'].map(quality_tier_map)
        
        # 4. Boolean feature encoding (ensure they're properly encoded as 0/1)
        boolean_cols = ['has_expression', 'has_methylation', 'has_copy_number', 
                       'has_mutations', 'has_protein', 'has_clinical', 'has_mirna', 'has_data']
        
        for col in boolean_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(int)
        
        # Store encoding mappings
        self.preprocessing_stats['encoding_mappings'] = {
            'cancer_type_classes': le_cancer.classes_.tolist(),
            'cancer_family_classes': le_family.classes_.tolist(),
            'quality_tier_mapping': quality_tier_map,
            'top_10_cancers': top_10_cancers.tolist(),
            'onehot_categories': [cat.tolist() for cat in ohe_cancer.categories_]
        }
        
        self.logger.info(f"✅ Categorical encoding complete")
        self.logger.info(f"   Cancer types: {len(le_cancer.classes_)} unique types")
        self.logger.info(f"   Cancer families: {len(le_family.classes_)} families")
        
        return df_encoded

    def create_feature_sets(self, df):
        """Create different feature sets for different modeling approaches"""
        self.logger.info("📋 Creating specialized feature sets...")
        
        feature_sets = {}
        
        # Base features (always included)
        base_features = [
            'num_data_types', 'num_files', 'total_size_mb', 'quality_score',
            'omics_completeness_score', 'omics_diversity_index', 
            'quality_score_normalized', 'file_density', 'avg_file_size_mb',
            'cancer_type_frequency', 'cancer_type_rarity'
        ]
        
        # Boolean omics features
        omics_features = [
            'has_expression', 'has_methylation', 'has_copy_number', 
            'has_mutations', 'has_protein', 'has_clinical'
        ]
        
        # Engineered omics combination features
        omics_combo_features = [col for col in df.columns if col.startswith('omics_combo_')]
        
        # Cancer type encoding features
        cancer_label_features = ['cancer_type_label', 'cancer_family_label']
        cancer_onehot_features = [col for col in df.columns if col.startswith('cancer_type_')]
        
        # 1. Minimal Feature Set (for quick prototyping)
        feature_sets['minimal'] = base_features[:4] + omics_features + ['cancer_type_label']
        
        # 2. Standard Feature Set (balanced approach)
        feature_sets['standard'] = (base_features + omics_features + 
                                  cancer_label_features + ['quality_tier_ordinal'])
        
        # 3. Rich Feature Set (all engineered features)
        feature_sets['rich'] = (base_features + omics_features + omics_combo_features +
                               cancer_label_features + ['quality_tier_ordinal'])
        
        # 4. One-hot Feature Set (for neural networks)
        feature_sets['onehot'] = (base_features + omics_features + 
                                cancer_onehot_features + ['quality_tier_ordinal'])
        
        # 5. Multi-omics Focused Set
        feature_sets['multi_omics'] = ([col for col in base_features if 'omics' in col or 'quality' in col] +
                                     omics_features + omics_combo_features + ['cancer_type_label'])
        
        # 6. Tree-friendly Set (ordinal and label encoded)
        feature_sets['tree_friendly'] = (base_features + omics_features + 
                                        ['cancer_type_label', 'cancer_type_ordinal', 
                                         'cancer_family_label', 'quality_tier_ordinal'])
        
        # Validate feature sets
        for name, features in feature_sets.items():
            available_features = [f for f in features if f in df.columns]
            missing_features = [f for f in features if f not in df.columns]
            
            if missing_features:
                self.logger.warning(f"⚠️ {name} set missing features: {missing_features}")
            
            feature_sets[name] = available_features
            self.logger.info(f"📋 {name.title()} feature set: {len(available_features)} features")
        
        return feature_sets

    def apply_scaling(self, df, feature_sets):
        """Apply different scaling strategies to feature sets"""
        self.logger.info("⚖️ Applying feature scaling...")
        
        scaled_datasets = {}
        
        # Features that should not be scaled (categorical/binary)
        no_scale_features = [
            'cancer_type_label', 'cancer_type_ordinal', 'cancer_family_label',
            'quality_tier_ordinal', 'has_expression', 'has_methylation', 
            'has_copy_number', 'has_mutations', 'has_protein', 'has_clinical'
        ] + [col for col in df.columns if col.startswith('cancer_type_') or col.startswith('omics_combo_')]
        
        scaling_methods = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        for scale_name, scaler in scaling_methods.items():
            scaled_datasets[scale_name] = {}
            self.scalers[scale_name] = {}
            
            for set_name, features in feature_sets.items():
                df_scaled = df[features + ['cancer_type']].copy()  # Include target
                
                # Separate features that need scaling
                scale_features = [f for f in features if f not in no_scale_features and f in df.columns]
                
                if scale_features:
                    # Fit and transform scaling features
                    scaler_fitted = scaling_methods[scale_name].__class__()  # Create new instance
                    df_scaled[scale_features] = scaler_fitted.fit_transform(df_scaled[scale_features])
                    self.scalers[scale_name][set_name] = scaler_fitted
                
                scaled_datasets[scale_name][set_name] = df_scaled
        
        self.logger.info(f"✅ Applied {len(scaling_methods)} scaling methods to {len(feature_sets)} feature sets")
        
        return scaled_datasets

    def compute_class_weights(self, df):
        """Compute class weights to handle imbalanced cancer types"""
        self.logger.info("⚖️ Computing class weights for imbalanced classes...")
        
        # Get class distribution
        cancer_counts = df['cancer_type'].value_counts()
        
        # Compute class weights using different strategies
        y_labels = df['cancer_type_label'].values
        unique_labels = np.unique(y_labels)
        
        # 1. Balanced class weights (sklearn default)
        balanced_weights = compute_class_weight('balanced', classes=unique_labels, y=y_labels)
        balanced_weight_dict = dict(zip(unique_labels, balanced_weights))
        
        # 2. Custom weights (square root of inverse frequency)
        total_samples = len(df)
        custom_weights = {}
        for label in unique_labels:
            cancer_type = df[df['cancer_type_label'] == label]['cancer_type'].iloc[0]
            freq = cancer_counts[cancer_type]
            custom_weights[label] = np.sqrt(total_samples / freq)
        
        # 3. Log-based weights (less aggressive than balanced)
        log_weights = {}
        max_count = cancer_counts.max()
        for label in unique_labels:
            cancer_type = df[df['cancer_type_label'] == label]['cancer_type'].iloc[0]
            freq = cancer_counts[cancer_type]
            log_weights[label] = np.log(max_count / freq + 1)
        
        self.class_weights = {
            'balanced': balanced_weight_dict,
            'custom_sqrt': custom_weights,
            'log_based': log_weights
        }
        
        # Log weight statistics
        for weight_name, weights in self.class_weights.items():
            min_weight = min(weights.values())
            max_weight = max(weights.values())
            self.logger.info(f"   {weight_name}: weight range {min_weight:.3f} - {max_weight:.3f}")
        
        return self.class_weights

    def create_stratified_splits(self, df, feature_sets, test_size=0.2, val_size=0.2):
        """Create stratified train/val/test splits"""
        self.logger.info("🔄 Creating stratified dataset splits...")
        
        splits = {}
        
        # Ensure we have enough samples per class for splitting
        cancer_counts = df['cancer_type'].value_counts()
        min_samples = cancer_counts.min()
        
        if min_samples < 3:
            self.logger.warning(f"⚠️ Some cancer types have only {min_samples} samples - may affect stratification")
        
        # Primary stratification by cancer type
        X = df.drop(['cancer_type'], axis=1)
        y = df['cancer_type']
        
        # First split: train+val vs test
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_val_idx, test_idx = next(sss1.split(X, y))
        
        # Second split: train vs val from train+val
        X_train_val = X.iloc[train_val_idx]
        y_train_val = y.iloc[train_val_idx]
        
        adjusted_val_size = val_size / (1 - test_size)  # Adjust for already removed test set
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=42)
        train_idx_relative, val_idx_relative = next(sss2.split(X_train_val, y_train_val))
        
        # Convert back to absolute indices
        train_idx = train_val_idx[train_idx_relative]
        val_idx = train_val_idx[val_idx_relative]
        
        # Create splits for each feature set
        for set_name, features in feature_sets.items():
            splits[set_name] = {
                'train': {
                    'X': df.iloc[train_idx][features],
                    'y': df.iloc[train_idx]['cancer_type'],
                    'y_label': df.iloc[train_idx]['cancer_type_label'],
                    'indices': train_idx
                },
                'val': {
                    'X': df.iloc[val_idx][features],
                    'y': df.iloc[val_idx]['cancer_type'],
                    'y_label': df.iloc[val_idx]['cancer_type_label'],
                    'indices': val_idx
                },
                'test': {
                    'X': df.iloc[test_idx][features],
                    'y': df.iloc[test_idx]['cancer_type'],
                    'y_label': df.iloc[test_idx]['cancer_type_label'],
                    'indices': test_idx
                }
            }
        
        # Log split statistics
        split_stats = {
            'total_samples': len(df),
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'test_samples': len(test_idx),
            'train_ratio': len(train_idx) / len(df),
            'val_ratio': len(val_idx) / len(df),
            'test_ratio': len(test_idx) / len(df)
        }
        
        # Check class distribution in splits
        train_cancer_dist = df.iloc[train_idx]['cancer_type'].value_counts()
        val_cancer_dist = df.iloc[val_idx]['cancer_type'].value_counts()
        test_cancer_dist = df.iloc[test_idx]['cancer_type'].value_counts()
        
        split_stats['cancer_distribution'] = {
            'train': train_cancer_dist.to_dict(),
            'val': val_cancer_dist.to_dict(),
            'test': test_cancer_dist.to_dict()
        }
        
        self.preprocessing_stats['split_statistics'] = split_stats
        
        self.logger.info(f"✅ Created stratified splits:")
        self.logger.info(f"   Train: {len(train_idx):,} samples ({len(train_idx)/len(df)*100:.1f}%)")
        self.logger.info(f"   Val: {len(val_idx):,} samples ({len(val_idx)/len(df)*100:.1f}%)")
        self.logger.info(f"   Test: {len(test_idx):,} samples ({len(test_idx)/len(df)*100:.1f}%)")
        
        return splits

    def save_processed_datasets(self, scaled_datasets, splits, feature_sets):
        """Save all processed datasets and preprocessing objects"""
        self.logger.info("💾 Saving processed datasets and preprocessing objects...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directory structure
        datasets_dir = self.output_dir / f"processed_datasets_{timestamp}"
        datasets_dir.mkdir(exist_ok=True)
        
        # Save feature sets
        with open(datasets_dir / "feature_sets.json", 'w') as f:
            json.dump(feature_sets, f, indent=2)
        
        # Save preprocessing objects
        with open(datasets_dir / "preprocessing_objects.pkl", 'wb') as f:
            pickle.dump({
                'scalers': self.scalers,
                'encoders': self.encoders,
                'class_weights': self.class_weights,
                'preprocessing_stats': self.preprocessing_stats
            }, f)
        
        # Save datasets in different formats
        saved_files = []
        
        # 1. Save scaled datasets
        for scale_method, scale_datasets in scaled_datasets.items():
            scale_dir = datasets_dir / scale_method
            scale_dir.mkdir(exist_ok=True)
            
            for set_name, dataset in scale_datasets.items():
                file_path = scale_dir / f"{set_name}_features.csv"
                dataset.to_csv(file_path, index=False)
                saved_files.append(str(file_path))
        
        # 2. Save train/val/test splits
        splits_dir = datasets_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Save splits with standard scaling (most common)
        if 'standard' in scaled_datasets:
            for set_name, features in feature_sets.items():
                set_dir = splits_dir / set_name
                set_dir.mkdir(exist_ok=True)
                
                if set_name in splits:
                    for split_type in ['train', 'val', 'test']:
                        split_data = splits[set_name][split_type]
                        
                        # Save features and labels
                        split_data['X'].to_csv(set_dir / f"{split_type}_X.csv", index=False)
                        pd.Series(split_data['y']).to_csv(set_dir / f"{split_type}_y.csv", index=False)
                        pd.Series(split_data['y_label']).to_csv(set_dir / f"{split_type}_y_label.csv", index=False)
                        
                        saved_files.extend([
                            str(set_dir / f"{split_type}_X.csv"),
                            str(set_dir / f"{split_type}_y.csv"),
                            str(set_dir / f"{split_type}_y_label.csv")
                        ])
        
        # Save comprehensive preprocessing report
        report = {
            'timestamp': timestamp,
            'original_dataset': str(self.dataset_path),
            'preprocessing_stats': self.preprocessing_stats,
            'feature_sets': {name: len(features) for name, features in feature_sets.items()},
            'scaling_methods': list(self.scalers.keys()),
            'class_weights_available': list(self.class_weights.keys()),
            'saved_files_count': len(saved_files)
        }
        
        with open(datasets_dir / "preprocessing_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"✅ Saved {len(saved_files)} processed dataset files")
        self.logger.info(f"📁 Output directory: {datasets_dir}")
        
        return datasets_dir, saved_files

    def run_comprehensive_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        self.logger.info("🚀 Starting comprehensive preprocessing pipeline...")
        
        try:
            # Step 1: Feature Engineering
            df_engineered = self.engineer_features()
            
            # Step 2: Categorical Encoding
            df_encoded = self.encode_categorical_features(df_engineered)
            
            # Step 3: Create Feature Sets
            feature_sets = self.create_feature_sets(df_encoded)
            
            # Step 4: Apply Scaling
            scaled_datasets = self.apply_scaling(df_encoded, feature_sets)
            
            # Step 5: Compute Class Weights
            class_weights = self.compute_class_weights(df_encoded)
            
            # Step 6: Create Stratified Splits
            splits = self.create_stratified_splits(df_encoded, feature_sets)
            
            # Step 7: Save Everything
            output_dir, saved_files = self.save_processed_datasets(scaled_datasets, splits, feature_sets)
            
            # Print Summary
            self.print_preprocessing_summary(feature_sets, output_dir)
            
            return {
                'output_directory': output_dir,
                'feature_sets': feature_sets,
                'scaled_datasets': scaled_datasets,
                'splits': splits,
                'class_weights': class_weights,
                'preprocessing_stats': self.preprocessing_stats,
                'saved_files': saved_files
            }
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {e}")
            raise

    def print_preprocessing_summary(self, feature_sets, output_dir):
        """Print comprehensive preprocessing summary"""
        stats = self.preprocessing_stats
        
        print(f"""
============================================================
🔧 50K DATASET PREPROCESSING COMPLETE
============================================================

📊 TRANSFORMATION SUMMARY:
   Original shape: {stats['original_shape']}
   Features engineered: {stats['feature_engineering']['new_features_count']}
   Total features after engineering: {stats['feature_engineering']['total_features']}

📋 FEATURE SETS CREATED: {len(feature_sets)}""")
        
        for name, features in feature_sets.items():
            print(f"   {name.title()}: {len(features)} features")
        
        print(f"""
⚖️ SCALING & ENCODING:
   Scaling methods: {len(self.scalers)} (Standard, MinMax, Robust)
   Cancer types encoded: {len(stats['encoding_mappings']['cancer_type_classes'])}
   Cancer families: {len(stats['encoding_mappings']['cancer_family_classes'])}

🔄 DATASET SPLITS:
   Total samples: {stats['split_statistics']['total_samples']:,}
   Train: {stats['split_statistics']['train_samples']:,} ({stats['split_statistics']['train_ratio']*100:.1f}%)
   Validation: {stats['split_statistics']['val_samples']:,} ({stats['split_statistics']['val_ratio']*100:.1f}%)
   Test: {stats['split_statistics']['test_samples']:,} ({stats['split_statistics']['test_ratio']*100:.1f}%)

⚖️ CLASS WEIGHTS COMPUTED:
   Balanced: {min(self.class_weights['balanced'].values()):.3f} - {max(self.class_weights['balanced'].values()):.3f}
   Custom: {min(self.class_weights['custom_sqrt'].values()):.3f} - {max(self.class_weights['custom_sqrt'].values()):.3f}
   Log-based: {min(self.class_weights['log_based'].values()):.3f} - {max(self.class_weights['log_based'].values()):.3f}

📁 OUTPUT DIRECTORY: {output_dir}

============================================================
✅ READY FOR ADVANCED MACHINE LEARNING!
============================================================
""")

def main():
    print("=" * 70)
    print("🔧 ADVANCED 50K DATASET PREPROCESSING PIPELINE")
    print("=" * 70)
    print("Comprehensive feature engineering and ML preparation")
    print("=" * 70)
    
    # Use the latest 50k dataset
    dataset_path = "data/ultra_permissive_50k_output/ultra_permissive_50k_plus_50000_20250822_184637.csv"
    
    try:
        preprocessor = Advanced50kPreprocessor(dataset_path)
        results = preprocessor.run_comprehensive_preprocessing()
        
        print(f"\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📁 Output directory: {results['output_directory']}")
        print(f"📋 Feature sets created: {len(results['feature_sets'])}")
        print(f"💾 Files saved: {len(results['saved_files'])}")
        
        print(f"\n🚀 Ready to proceed with advanced machine learning!")
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()
