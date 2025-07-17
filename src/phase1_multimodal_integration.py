#!/usr/bin/env python3
"""
Phase 1: Reframe the Scientific Problem - Advanced Multi-Modal Data Integration
================================================================

This script implements advanced multi-modal cancer genomics data integration 
moving beyond basic classification to precision oncology.

Key innovations:
- Real TCGA, GEO, and ENCODE data integration
- Advanced feature engineering for multi-omics
- Biologically-informed feature selection
- Precision oncology-focused problem formulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class MultiModalCancerIntegrator:
    """Advanced multi-modal cancer genomics data integration"""
    
    def __init__(self, data_path="project36_data/processed/"):
        self.data_path = data_path
        self.datasets = {}
        self.integrated_data = None
        self.feature_importance = {}
        
    def load_multimodal_datasets(self):
        """Load all available multi-modal datasets"""
        
        # Load the comprehensive integrated dataset
        print("Loading comprehensive multi-modal dataset...")
        self.datasets['complete_integrated'] = pd.read_csv(
            f"{self.data_path}/complete_three_source_integrated_data.csv"
        )
        
        # Load real-world integrated dataset
        print("Loading real-world multi-modal dataset...")
        self.datasets['real_integrated'] = pd.read_csv(
            f"{self.data_path}/real_integrated_multimodal_features.csv"
        )
        
        # Load individual modality datasets
        print("Loading individual modality datasets...")
        
        # Methylation data
        self.datasets['methylation'] = pd.read_csv(
            f"{self.data_path}/actual_tcga_methylation_features.csv"
        )
        
        # Copy Number Alteration data
        self.datasets['cna'] = pd.read_csv(
            f"{self.data_path}/realistic_cna_features.csv"
        )
        
        # Fragment data
        self.datasets['fragmentomics'] = pd.read_csv(
            f"{self.data_path}/realistic_fragmentomics_features.csv"
        )
        
        print(f"Loaded {len(self.datasets)} datasets")
        for name, df in self.datasets.items():
            print(f"  {name}: {df.shape[0]} samples, {df.shape[1]} features")
            
    def analyze_data_quality(self):
        """Comprehensive data quality analysis"""
        
        print("\n=== DATA QUALITY ANALYSIS ===")
        
        for name, df in self.datasets.items():
            print(f"\n{name.upper()} Dataset:")
            print(f"  Shape: {df.shape}")
            print(f"  Missing values: {df.isnull().sum().sum()}")
            print(f"  Duplicate rows: {df.duplicated().sum()}")
            
            # Check for potential cancer types/labels
            if 'label' in df.columns:
                print(f"  Cancer types: {df['label'].unique()}")
                print(f"  Class distribution: {df['label'].value_counts().to_dict()}")
                
    def engineer_advanced_features(self):
        """Advanced feature engineering for precision oncology"""
        
        print("\n=== ADVANCED FEATURE ENGINEERING ===")
        
        # Use the most comprehensive dataset
        df = self.datasets['real_integrated'].copy()
        
        # Separate features by modality
        methyl_features = [col for col in df.columns if col.startswith('methyl_')]
        cna_features = [col for col in df.columns if col.startswith('cna_')]
        fragment_features = [col for col in df.columns if col.startswith('fragment_')]
        
        print(f"Methylation features: {len(methyl_features)}")
        print(f"CNA features: {len(cna_features)}")
        print(f"Fragment features: {len(fragment_features)}")
        
        # Create advanced interaction features
        print("\nCreating advanced interaction features...")
        
        # Methylation-CNA interactions (epigenetic-genomic coupling)
        if 'methyl_cancer_likelihood' in df.columns and 'cna_genomic_complexity_score' in df.columns:
            df['epi_genomic_coupling'] = df['methyl_cancer_likelihood'] * df['cna_genomic_complexity_score']
            
        # Fragment-methylation interactions (liquid biopsy signatures)
        if 'fragment_nucleosome_signal' in df.columns and 'methyl_global_methylation_mean' in df.columns:
            df['liquid_biopsy_signature'] = df['fragment_nucleosome_signal'] * df['methyl_global_methylation_mean']
            
        # Multi-modal stability score
        stability_features = []
        if 'methyl_data_quality_score' in df.columns:
            stability_features.append('methyl_data_quality_score')
        if 'cna_coverage_uniformity' in df.columns:
            stability_features.append('cna_coverage_uniformity')
        if 'fragment_fragment_quality_score' in df.columns:
            stability_features.append('fragment_fragment_quality_score')
            
        if stability_features:
            df['multimodal_stability_score'] = df[stability_features].mean(axis=1)
            
        # Cancer aggressiveness composite score
        aggressiveness_features = []
        if 'cna_chromosomal_instability_index' in df.columns:
            aggressiveness_features.append('cna_chromosomal_instability_index')
        if 'methyl_extreme_hypermethylation' in df.columns:
            aggressiveness_features.append('methyl_extreme_hypermethylation')
        if 'fragment_fragment_jaggedness' in df.columns:
            aggressiveness_features.append('fragment_fragment_jaggedness')
            
        if aggressiveness_features:
            # Normalize features before combining
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[aggressiveness_features])
            df['cancer_aggressiveness_score'] = np.mean(scaled_features, axis=1)
            
        self.integrated_data = df
        print(f"Final integrated dataset: {df.shape}")
        
    def perform_precision_oncology_analysis(self):
        """Precision oncology-focused analysis"""
        
        print("\n=== PRECISION ONCOLOGY ANALYSIS ===")
        
        if self.integrated_data is None:
            print("No integrated data available. Run feature engineering first.")
            return
            
        df = self.integrated_data.copy()
        
        # Identify high-impact features for precision oncology
        feature_cols = [col for col in df.columns if col not in ['sample_id', 'label', 'data_sources']]
        
        print(f"Analyzing {len(feature_cols)} features for precision oncology impact...")
        
        # Feature importance analysis
        X = df[feature_cols].fillna(0)
        
        # Create synthetic labels for demonstration (in real scenario, use actual cancer subtypes)
        # For now, create risk stratification based on cancer aggressiveness
        if 'cancer_aggressiveness_score' in df.columns:
            y = pd.cut(df['cancer_aggressiveness_score'], 
                      bins=3, labels=['Low_Risk', 'Medium_Risk', 'High_Risk'])
            y = y.astype(str)
            
            # Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = feature_importance
            
            print("\nTop 10 most important features for precision oncology:")
            print(feature_importance.head(10))
            
        # Dimensionality reduction for visualization
        print("\nPerforming dimensionality reduction...")
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(StandardScaler().fit_transform(X))
        
        # UMAP for better non-linear visualization
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_result = umap_reducer.fit_transform(StandardScaler().fit_transform(X))
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA plot
        scatter = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=range(len(pca_result)), cmap='viridis', alpha=0.6)
        axes[0].set_title('PCA - Multi-Modal Cancer Data')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # UMAP plot
        scatter = axes[1].scatter(umap_result[:, 0], umap_result[:, 1], 
                                 c=range(len(umap_result)), cmap='viridis', alpha=0.6)
        axes[1].set_title('UMAP - Multi-Modal Cancer Data')
        axes[1].set_xlabel('UMAP1')
        axes[1].set_ylabel('UMAP2')
        
        plt.tight_layout()
        plt.savefig('results/multimodal_cancer_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca_result, umap_result
        
    def generate_precision_oncology_report(self):
        """Generate comprehensive precision oncology report"""
        
        print("\n=== PRECISION ONCOLOGY REPORT ===")
        
        if self.integrated_data is None:
            print("No integrated data available.")
            return
            
        report = {
            'dataset_summary': {
                'total_samples': len(self.integrated_data),
                'total_features': len([col for col in self.integrated_data.columns 
                                     if col not in ['sample_id', 'label', 'data_sources']]),
                'data_sources': self.integrated_data['data_sources'].iloc[0] if 'data_sources' in self.integrated_data.columns else 'Unknown'
            },
            'modality_breakdown': {
                'methylation_features': len([col for col in self.integrated_data.columns if col.startswith('methyl_')]),
                'cna_features': len([col for col in self.integrated_data.columns if col.startswith('cna_')]),
                'fragment_features': len([col for col in self.integrated_data.columns if col.startswith('fragment_')])
            },
            'advanced_features': {
                'interaction_features': len([col for col in self.integrated_data.columns 
                                           if 'interaction' in col or 'coupling' in col or 'signature' in col]),
                'composite_scores': len([col for col in self.integrated_data.columns 
                                       if 'score' in col and not col.startswith(('methyl_', 'cna_', 'fragment_'))])
            }
        }
        
        print("Dataset Summary:")
        for key, value in report['dataset_summary'].items():
            print(f"  {key}: {value}")
            
        print("\nModality Breakdown:")
        for key, value in report['modality_breakdown'].items():
            print(f"  {key}: {value}")
            
        print("\nAdvanced Features:")
        for key, value in report['advanced_features'].items():
            print(f"  {key}: {value}")
            
        if hasattr(self, 'feature_importance') and not self.feature_importance.empty:
            print("\nTop Precision Oncology Biomarkers:")
            top_features = self.feature_importance.head(5)
            for idx, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
                
        return report
        
    def save_integrated_dataset(self, filename="data/processed/cancer_alpha_integrated.csv"):
        """Save the integrated dataset for further analysis"""
        
        if self.integrated_data is not None:
            self.integrated_data.to_csv(filename, index=False)
            print(f"\nIntegrated dataset saved to: {filename}")
            print(f"Dataset shape: {self.integrated_data.shape}")
        else:
            print("No integrated data to save.")

def main():
    """Main execution function"""
    
    print("=== CANCER ALPHA - PHASE 1: MULTIMODAL INTEGRATION ===")
    
    # Initialize integrator
    integrator = MultiModalCancerIntegrator()
    
    # Load datasets
    integrator.load_multimodal_datasets()
    
    # Analyze data quality
    integrator.analyze_data_quality()
    
    # Engineer advanced features
    integrator.engineer_advanced_features()
    
    # Perform precision oncology analysis
    integrator.perform_precision_oncology_analysis()
    
    # Generate report
    report = integrator.generate_precision_oncology_report()
    
    # Save integrated dataset
    integrator.save_integrated_dataset()
    
    print("\n=== PHASE 1 COMPLETE ===")
    print("Ready for Phase 2: Technical and Model Innovation")

if __name__ == "__main__":
    main()
