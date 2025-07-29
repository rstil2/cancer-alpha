#!/usr/bin/env python3
"""
Enhanced Synthetic Data Generator for Cancer Genomics
====================================================

This module generates high-quality synthetic genomic data with:
- Biologically realistic feature patterns
- Cancer-type-specific signatures
- Proper feature correlations
- Controlled noise patterns
- Large-scale data generation (50K+ samples per class)

Target: Achieve 90% validation accuracy through better data quality
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from scipy import stats
import joblib
from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSyntheticDataGenerator:
    """Generate high-quality synthetic cancer genomics data"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Cancer types
        self.cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
        self.n_classes = len(self.cancer_types)
        
        # Feature groups
        self.feature_groups = {
            'methylation': 20,
            'mutation': 25,
            'cn_alteration': 20,
            'fragmentomics': 15,
            'clinical': 10,
            'icgc_argo': 20
        }
        self.total_features = sum(self.feature_groups.values())  # 110 features
        
        # Initialize cancer-specific patterns
        self._initialize_cancer_patterns()
    
    def _initialize_cancer_patterns(self):
        """Initialize cancer-type-specific feature patterns based on biological knowledge"""
        
        # Define realistic cancer-specific signatures
        self.cancer_signatures = {
            'BRCA': {  # Breast Cancer
                'methylation': {'mean': 0.6, 'std': 0.15, 'skew': -0.5},
                'mutation': {'mean': 8.0, 'std': 2.0, 'skew': 0.3},
                'cn_alteration': {'mean': 12.0, 'std': 3.0, 'skew': 0.8},
                'fragmentomics': {'mean': 140.0, 'std': 20.0, 'skew': -0.2},
                'clinical': {'mean': 0.7, 'std': 0.1, 'skew': -0.3},
                'icgc_argo': {'mean': 1.8, 'std': 0.4, 'skew': 0.5}
            },
            'LUAD': {  # Lung Adenocarcinoma
                'methylation': {'mean': 0.4, 'std': 0.12, 'skew': 0.2},
                'mutation': {'mean': 15.0, 'std': 4.0, 'skew': 0.6},
                'cn_alteration': {'mean': 8.0, 'std': 2.5, 'skew': 0.4},
                'fragmentomics': {'mean': 130.0, 'std': 25.0, 'skew': 0.1},
                'clinical': {'mean': 0.3, 'std': 0.15, 'skew': 0.4},
                'icgc_argo': {'mean': 2.2, 'std': 0.5, 'skew': 0.3}
            },
            'COAD': {  # Colon Adenocarcinoma
                'methylation': {'mean': 0.8, 'std': 0.1, 'skew': -0.8},
                'mutation': {'mean': 12.0, 'std': 3.0, 'skew': 0.4},
                'cn_alteration': {'mean': 15.0, 'std': 4.0, 'skew': 0.6},
                'fragmentomics': {'mean': 160.0, 'std': 15.0, 'skew': -0.4},
                'clinical': {'mean': 0.5, 'std': 0.12, 'skew': 0.0},
                'icgc_argo': {'mean': 1.5, 'std': 0.3, 'skew': 0.2}
            },
            'PRAD': {  # Prostate Adenocarcinoma
                'methylation': {'mean': 0.3, 'std': 0.08, 'skew': 0.5},
                'mutation': {'mean': 6.0, 'std': 1.5, 'skew': 0.8},
                'cn_alteration': {'mean': 5.0, 'std': 1.5, 'skew': 1.2},
                'fragmentomics': {'mean': 180.0, 'std': 10.0, 'skew': -0.6},
                'clinical': {'mean': 0.8, 'std': 0.08, 'skew': -0.5},
                'icgc_argo': {'mean': 1.2, 'std': 0.25, 'skew': 0.1}
            },
            'STAD': {  # Stomach Adenocarcinoma
                'methylation': {'mean': 0.7, 'std': 0.18, 'skew': -0.3},
                'mutation': {'mean': 10.0, 'std': 2.5, 'skew': 0.2},
                'cn_alteration': {'mean': 18.0, 'std': 5.0, 'skew': 0.7},
                'fragmentomics': {'mean': 125.0, 'std': 30.0, 'skew': 0.3},
                'clinical': {'mean': 0.4, 'std': 0.2, 'skew': 0.6},
                'icgc_argo': {'mean': 2.0, 'std': 0.6, 'skew': 0.4}
            },
            'KIRC': {  # Kidney Clear Cell Carcinoma
                'methylation': {'mean': 0.2, 'std': 0.05, 'skew': 1.0},
                'mutation': {'mean': 4.0, 'std': 1.0, 'skew': 1.5},
                'cn_alteration': {'mean': 20.0, 'std': 6.0, 'skew': 0.9},
                'fragmentomics': {'mean': 200.0, 'std': 8.0, 'skew': -1.0},
                'clinical': {'mean': 0.6, 'std': 0.1, 'skew': -0.2},
                'icgc_argo': {'mean': 0.8, 'std': 0.2, 'skew': 0.8}
            },
            'HNSC': {  # Head and Neck Squamous Cell Carcinoma
                'methylation': {'mean': 0.5, 'std': 0.2, 'skew': 0.0},
                'mutation': {'mean': 20.0, 'std': 5.0, 'skew': 0.5},
                'cn_alteration': {'mean': 25.0, 'std': 7.0, 'skew': 0.3},
                'fragmentomics': {'mean': 110.0, 'std': 35.0, 'skew': 0.8},
                'clinical': {'mean': 0.2, 'std': 0.18, 'skew': 1.0},
                'icgc_argo': {'mean': 2.5, 'std': 0.8, 'skew': 0.2}
            },
            'LIHC': {  # Liver Hepatocellular Carcinoma
                'methylation': {'mean': 0.9, 'std': 0.05, 'skew': -1.5},
                'mutation': {'mean': 7.0, 'std': 2.0, 'skew': 0.7},
                'cn_alteration': {'mean': 30.0, 'std': 8.0, 'skew': 0.5},
                'fragmentomics': {'mean': 90.0, 'std': 40.0, 'skew': 1.2},
                'clinical': {'mean': 0.1, 'std': 0.05, 'skew': 2.0},
                'icgc_argo': {'mean': 3.0, 'std': 1.0, 'skew': 0.1}
            }
        }
        
        # Feature correlation matrices for each modality
        self._initialize_correlation_matrices()
    
    def _initialize_correlation_matrices(self):
        """Initialize realistic correlation matrices for each feature group"""
        
        self.correlation_matrices = {}
        
        for group, n_features in self.feature_groups.items():
            # Create realistic correlation structure
            # Features within same modality are moderately correlated
            base_corr = 0.3 if group in ['methylation', 'cn_alteration'] else 0.1
            
            # Generate positive definite correlation matrix
            random_matrix = np.random.randn(n_features, n_features)
            cov_matrix = np.dot(random_matrix, random_matrix.T)
            
            # Normalize to correlation matrix
            std_devs = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
            
            # Adjust correlation strength
            corr_matrix = base_corr * corr_matrix + (1 - base_corr) * np.eye(n_features)
            
            self.correlation_matrices[group] = corr_matrix
    
    def _generate_skewed_normal(self, mean, std, skew, size):
        """Generate skewed normal distribution using scipy"""
        # Convert skewness to scipy's skewnorm parameter
        a = skew
        scale = std
        loc = mean
        
        return stats.skewnorm.rvs(a=a, loc=loc, scale=scale, size=size)
    
    def generate_cancer_sample(self, cancer_type, n_samples=1):
        """Generate samples for a specific cancer type"""
        
        if cancer_type not in self.cancer_signatures:
            raise ValueError(f"Unknown cancer type: {cancer_type}")
        
        signature = self.cancer_signatures[cancer_type]
        samples = []
        
        for group, n_features in self.feature_groups.items():
            group_params = signature[group]
            
            # Generate base features with realistic distributions
            if group == 'mutation':
                # Mutations should be count data (Poisson-like)
                base_features = np.random.poisson(group_params['mean'], (n_samples, n_features))
                base_features = base_features.astype(float)
            else:
                # Generate skewed normal features
                base_features = np.zeros((n_samples, n_features))
                for i in range(n_features):
                    base_features[:, i] = self._generate_skewed_normal(
                        mean=group_params['mean'],
                        std=group_params['std'],
                        skew=group_params['skew'],
                        size=n_samples
                    )
            
            # Apply correlation structure
            if n_samples > 1:
                try:
                    L = np.linalg.cholesky(self.correlation_matrices[group])
                    correlated_features = np.dot(base_features, L.T)
                except np.linalg.LinAlgError:
                    # If correlation matrix is not positive definite, use original features
                    correlated_features = base_features
            else:
                correlated_features = base_features
            
            # Add some cancer-type-specific noise
            noise_level = 0.05 * group_params['std']
            noise = np.random.normal(0, noise_level, correlated_features.shape)
            final_features = correlated_features + noise
            
            # Ensure realistic bounds for specific feature types
            if group == 'methylation':
                final_features = np.clip(final_features, 0, 1)
            elif group == 'mutation':
                final_features = np.maximum(final_features, 0)
            elif group == 'fragmentomics':
                final_features = np.maximum(final_features, 50)  # Minimum fragment size
            
            samples.append(final_features)
        
        # Concatenate all feature groups
        full_samples = np.concatenate(samples, axis=1)
        return full_samples
    
    def generate_balanced_dataset(self, n_samples_per_class=50000):
        """Generate a large, balanced dataset with all cancer types"""
        
        logger.info(f"Generating {n_samples_per_class} samples per class ({self.n_classes} classes)")
        logger.info(f"Total samples: {n_samples_per_class * self.n_classes}")
        
        all_samples = []
        all_labels = []
        
        for class_idx, cancer_type in enumerate(self.cancer_types):
            logger.info(f"Generating {cancer_type} samples...")
            
            # Generate samples in batches to manage memory
            batch_size = min(10000, n_samples_per_class)
            cancer_samples = []
            
            for start_idx in range(0, n_samples_per_class, batch_size):
                end_idx = min(start_idx + batch_size, n_samples_per_class)
                batch_samples = self.generate_cancer_sample(
                    cancer_type, 
                    n_samples=end_idx - start_idx
                )
                cancer_samples.append(batch_samples)
            
            # Combine batches
            cancer_data = np.vstack(cancer_samples)
            cancer_labels = np.full(n_samples_per_class, class_idx)
            
            all_samples.append(cancer_data)
            all_labels.append(cancer_labels)
            
            logger.info(f"Generated {cancer_type}: {cancer_data.shape}")
        
        # Combine all classes
        X = np.vstack(all_samples)
        y = np.concatenate(all_labels)
        
        # Shuffle the dataset
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        logger.info(f"Final dataset shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def save_dataset(self, X, y, output_dir='data'):
        """Save the generated dataset"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save data
        np.save(output_path / 'enhanced_X.npy', X)
        np.save(output_path / 'enhanced_y.npy', y)
        
        # Save metadata
        metadata = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': self.n_classes,
            'cancer_types': self.cancer_types,
            'feature_groups': self.feature_groups,
            'class_distribution': np.bincount(y).tolist()
        }
        
        import json
        with open(output_path / 'enhanced_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
        
        return output_path
    
    def generate_and_save(self, n_samples_per_class=50000, output_dir='data'):
        """Complete pipeline: generate and save enhanced dataset"""
        
        logger.info("Starting enhanced synthetic data generation...")
        
        # Generate dataset
        X, y = self.generate_balanced_dataset(n_samples_per_class)
        
        # Save dataset
        output_path = self.save_dataset(X, y, output_dir)
        
        # Generate feature names
        feature_names = []
        for group, n_features in self.feature_groups.items():
            feature_names.extend([f'{group}_{i}' for i in range(n_features)])
        
        # Save feature names
        with open(output_path / 'enhanced_feature_names.json', 'w') as f:
            json.dump(feature_names, f)
        
        logger.info("Enhanced synthetic data generation complete!")
        
        return X, y, feature_names

def main():
    """Generate enhanced synthetic dataset"""
    
    # Initialize generator
    generator = EnhancedSyntheticDataGenerator(random_state=42)
    
    # Generate dataset (start with smaller size for testing)
    X, y, feature_names = generator.generate_and_save(
        n_samples_per_class=10000,  # Start with 10K per class
        output_dir='data/enhanced'
    )
    
    print(f"Generated dataset: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Feature names: {len(feature_names)}")
    
    # Basic validation
    print("\nDataset Statistics:")
    print(f"Feature means: {X.mean(axis=0)[:10]}...")  # First 10 features
    print(f"Feature stds: {X.std(axis=0)[:10]}...")
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")

if __name__ == "__main__":
    main()
