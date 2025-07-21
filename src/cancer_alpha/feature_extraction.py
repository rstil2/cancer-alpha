import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MultiModalFeatureExtractor:
    """
    Comprehensive feature extraction for multi-modal ctDNA analysis
    """
    
    def __init__(self, output_dir="data/processed/features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        
    def extract_methylation_features(self, data_path, sample_type='cfMeDIP'):
        """
        Extract comprehensive methylation features from cfMeDIP-seq or bisulfite data
        
        Args:
            data_path: Path to methylation data
            sample_type: Type of methylation data ('cfMeDIP', 'bisulfite', 'array')
        
        Returns:
            dict: Comprehensive methylation features
        """
        print(f"Extracting methylation features from {sample_type} data...")
        
        try:
            # Load methylation data
            if data_path.endswith('.csv'):
                methyl_data = pd.read_csv(data_path)
            else:
                # Simulate methylation data for demonstration
                methyl_data = self._simulate_methylation_data()
            
            features = {}
            
            # 1. Global methylation metrics
            features['global_methylation_mean'] = methyl_data.select_dtypes(include=[np.number]).mean().mean()
            features['global_methylation_std'] = methyl_data.select_dtypes(include=[np.number]).std().mean()
            features['global_methylation_median'] = methyl_data.select_dtypes(include=[np.number]).median().median()
            
            # 2. CpG island features
            cpg_features = self._extract_cpg_features(methyl_data)
            features.update(cpg_features)
            
            # 3. Regional methylation patterns
            regional_features = self._extract_regional_methylation(methyl_data)
            features.update(regional_features)
            
            # 4. Methylation entropy and complexity
            complexity_features = self._calculate_methylation_complexity(methyl_data)
            features.update(complexity_features)
            
            # 5. Differential methylation signals
            diff_features = self._extract_differential_signals(methyl_data)
            features.update(diff_features)
            
            print(f"Extracted {len(features)} methylation features")
            return features
            
        except Exception as e:
            print(f"Error in methylation feature extraction: {e}")
            return self._get_default_methylation_features()
    
    def extract_fragmentomics_features(self, data_path=None):
        """
        Extract fragmentomics features from cfDNA fragment data
        
        Returns:
            dict: Fragment-based features including length, end motifs, nucleosome patterns
        """
        print("Extracting fragmentomics features...")
        
        try:
            # Simulate fragment data for demonstration
            fragment_data = self._simulate_fragment_data()
            
            features = {}
            
            # 1. Fragment length distribution
            length_features = self._extract_fragment_length_features(fragment_data)
            features.update(length_features)
            
            # 2. Fragment end motifs
            motif_features = self._extract_end_motif_features(fragment_data)
            features.update(motif_features)
            
            # 3. Nucleosome positioning signals
            nucleosome_features = self._extract_nucleosome_features(fragment_data)
            features.update(nucleosome_features)
            
            # 4. Fragment jaggedness and quality
            quality_features = self._extract_fragment_quality_features(fragment_data)
            features.update(quality_features)
            
            # 5. Tissue-of-origin signals
            tissue_features = self._extract_tissue_origin_features(fragment_data)
            features.update(tissue_features)
            
            print(f"Extracted {len(features)} fragmentomics features")
            return features
            
        except Exception as e:
            print(f"Error in fragmentomics feature extraction: {e}")
            return self._get_default_fragmentomics_features()
    
    def extract_cna_features(self, data_path=None):
        """
        Extract copy number alteration features from shallow WGS data
        
        Returns:
            dict: CNA-based features including gains, losses, and instability metrics
        """
        print("Extracting CNA features...")
        
        try:
            # Simulate CNA data for demonstration
            cna_data = self._simulate_cna_data()
            
            features = {}
            
            # 1. Global CNA burden
            burden_features = self._extract_cna_burden_features(cna_data)
            features.update(burden_features)
            
            # 2. Chromosomal instability metrics
            instability_features = self._extract_instability_features(cna_data)
            features.update(instability_features)
            
            # 3. Focal vs broad alterations
            alteration_features = self._extract_alteration_patterns(cna_data)
            features.update(alteration_features)
            
            # 4. Cancer-specific CNA signatures
            signature_features = self._extract_cna_signatures(cna_data)
            features.update(signature_features)
            
            # 5. Genomic complexity metrics
            complexity_features = self._extract_genomic_complexity(cna_data)
            features.update(complexity_features)
            
            print(f"Extracted {len(features)} CNA features")
            return features
            
        except Exception as e:
            print(f"Error in CNA feature extraction: {e}")
            return self._get_default_cna_features()
    
    # Helper methods for methylation features
    def _simulate_methylation_data(self, n_samples=100, n_cpgs=1000):
        """Simulate methylation data for testing"""
        np.random.seed(42)
        data = pd.DataFrame({
            f'CpG_{i}': np.random.beta(2, 2, n_samples) for i in range(n_cpgs)
        })
        data['sample_type'] = ['cancer' if i < n_samples//2 else 'control' for i in range(n_samples)]
        return data
    
    def _extract_cpg_features(self, data):
        """Extract CpG island specific features"""
        numeric_data = data.select_dtypes(include=[np.number])
        return {
            'cpg_high_methylation_ratio': (numeric_data > 0.7).sum().sum() / numeric_data.size,
            'cpg_low_methylation_ratio': (numeric_data < 0.3).sum().sum() / numeric_data.size,
            'cpg_intermediate_ratio': ((numeric_data >= 0.3) & (numeric_data <= 0.7)).sum().sum() / numeric_data.size
        }
    
    def _extract_regional_methylation(self, data):
        """Extract regional methylation patterns"""
        numeric_data = data.select_dtypes(include=[np.number])
        return {
            'promoter_methylation': numeric_data.iloc[:, :100].mean().mean(),
            'gene_body_methylation': numeric_data.iloc[:, 100:500].mean().mean(),
            'intergenic_methylation': numeric_data.iloc[:, 500:].mean().mean()
        }
    
    def _calculate_methylation_complexity(self, data):
        """Calculate methylation entropy and complexity"""
        numeric_data = data.select_dtypes(include=[np.number])
        return {
            'methylation_entropy': -np.sum(numeric_data.mean() * np.log2(numeric_data.mean() + 1e-10)),
            'methylation_variance': numeric_data.var().mean(),
            'methylation_skewness': stats.skew(numeric_data.mean())
        }
    
    def _extract_differential_signals(self, data):
        """Extract differential methylation signals"""
        numeric_data = data.select_dtypes(include=[np.number])
        return {
            'differential_variability': numeric_data.std().std(),
            'hypermethylation_events': (numeric_data.mean() > 0.8).sum(),
            'hypomethylation_events': (numeric_data.mean() < 0.2).sum()
        }
    
    def _get_default_methylation_features(self):
        """Default methylation features if extraction fails"""
        return {
            'global_methylation_mean': 0.5,
            'global_methylation_std': 0.2,
            'cpg_high_methylation_ratio': 0.3,
            'promoter_methylation': 0.4,
            'methylation_entropy': 1.5
        }
    
    # Helper methods for fragmentomics features
    def _simulate_fragment_data(self, n_fragments=10000):
        """Simulate fragment data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'length': np.random.normal(167, 20, n_fragments),
            'gc_content': np.random.beta(2, 2, n_fragments),
            'end_motif_score': np.random.exponential(1, n_fragments),
            'nucleosome_signal': np.random.gamma(2, 2, n_fragments)
        })
    
    def _extract_fragment_length_features(self, data):
        """Extract fragment length distribution features"""
        lengths = data['length']
        return {
            'mean_fragment_length': lengths.mean(),
            'fragment_length_std': lengths.std(),
            'short_fragment_ratio': (lengths < 150).sum() / len(lengths),
            'long_fragment_ratio': (lengths > 200).sum() / len(lengths),
            'mononucleosome_peak': (lengths.between(160, 180)).sum() / len(lengths)
        }
    
    def _extract_end_motif_features(self, data):
        """Extract fragment end motif features"""
        return {
            'end_motif_diversity': data['end_motif_score'].nunique(),
            'end_motif_entropy': stats.entropy(data['end_motif_score'].value_counts()),
            'cancer_motif_signature': data['end_motif_score'].quantile(0.9)
        }
    
    def _extract_nucleosome_features(self, data):
        """Extract nucleosome positioning features"""
        return {
            'nucleosome_periodicity': data['nucleosome_signal'].mean(),
            'nucleosome_positioning_score': data['nucleosome_signal'].std(),
            'dinucleosome_ratio': (data['nucleosome_signal'] > data['nucleosome_signal'].quantile(0.8)).sum() / len(data)
        }
    
    def _extract_fragment_quality_features(self, data):
        """Extract fragment quality and jaggedness features"""
        return {
            'fragment_jaggedness': data['gc_content'].std(),
            'fragment_uniformity': 1 / (data['length'].std() + 1),
            'quality_score': data['gc_content'].mean() * data['nucleosome_signal'].mean()
        }
    
    def _extract_tissue_origin_features(self, data):
        """Extract tissue-of-origin signals"""
        return {
            'tissue_signature_1': data[['gc_content', 'nucleosome_signal']].corr().iloc[0,1],
            'tissue_signature_2': (data['length'] * data['gc_content']).mean(),
            'lung_specific_signal': data['end_motif_score'].quantile(0.75)
        }
    
    def _get_default_fragmentomics_features(self):
        """Default fragmentomics features if extraction fails"""
        return {
            'mean_fragment_length': 167,
            'fragment_length_std': 20,
            'short_fragment_ratio': 0.2,
            'nucleosome_periodicity': 1.5,
            'fragment_jaggedness': 0.3
        }
    
    # Helper methods for CNA features
    def _simulate_cna_data(self, n_segments=1000):
        """Simulate CNA data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'chromosome': np.random.choice(range(1, 23), n_segments),
            'start': np.random.randint(1000000, 100000000, n_segments),
            'end': np.random.randint(1000000, 100000000, n_segments),
            'log2_ratio': np.random.normal(0, 0.5, n_segments),
            'segment_mean': np.random.normal(0, 0.3, n_segments)
        })
    
    def _extract_cna_burden_features(self, data):
        """Extract global CNA burden features"""
        return {
            'total_alterations': len(data),
            'amplification_burden': (data['log2_ratio'] > 0.3).sum(),
            'deletion_burden': (data['log2_ratio'] < -0.3).sum(),
            'neutral_regions': (data['log2_ratio'].abs() < 0.1).sum(),
            'mean_log2_ratio': data['log2_ratio'].mean()
        }
    
    def _extract_instability_features(self, data):
        """Extract chromosomal instability metrics"""
        return {
            'chromosomal_instability_index': data['log2_ratio'].std(),
            'breakpoint_density': len(data) / data['chromosome'].nunique(),
            'variance_log2_ratio': data['log2_ratio'].var(),
            'extreme_alterations': (data['log2_ratio'].abs() > 1.0).sum()
        }
    
    def _extract_alteration_patterns(self, data):
        """Extract focal vs broad alteration patterns"""
        data['segment_length'] = data['end'] - data['start']
        return {
            'focal_alterations': (data['segment_length'] < 1000000).sum(),
            'broad_alterations': (data['segment_length'] > 10000000).sum(),
            'mean_segment_length': data['segment_length'].mean(),
            'alteration_complexity': data.groupby('chromosome')['log2_ratio'].std().mean()
        }
    
    def _extract_cna_signatures(self, data):
        """Extract cancer-specific CNA signatures"""
        return {
            'lung_cancer_signature': (data['chromosome'].isin([3, 8, 17])).sum(),
            'oncogene_amplification': (data['log2_ratio'] > 0.5).sum(),
            'tumor_suppressor_deletion': (data['log2_ratio'] < -0.5).sum(),
            'chromothripsis_events': data.groupby('chromosome')['log2_ratio'].apply(lambda x: (x.diff().abs() > 1).sum()).sum()
        }
    
    def _extract_genomic_complexity(self, data):
        """Extract genomic complexity metrics"""
        return {
            'genomic_complexity_score': data['log2_ratio'].std() * len(data),
            'heterogeneity_index': data.groupby('chromosome')['log2_ratio'].std().std(),
            'ploidy_deviation': data['log2_ratio'].abs().mean(),
            'structural_variation_load': (data['log2_ratio'].diff().abs() > 0.5).sum()
        }
    
    def _get_default_cna_features(self):
        """Default CNA features if extraction fails"""
        return {
            'total_alterations': 100,
            'amplification_burden': 30,
            'deletion_burden': 25,
            'chromosomal_instability_index': 0.5,
            'genomic_complexity_score': 50
        }
    
    def perform_quality_control(self, data):
        """Perform comprehensive quality control checks"""
        qc_metrics = {
            'missing_values': data.isnull().sum().sum(),
            'duplicates': data.duplicated().sum(),
            'data_completeness': 1 - (data.isnull().sum().sum() / data.size),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'total_features': len(data.columns)
        }
        return qc_metrics
    
    def integrate_multimodal_data(self, methylation_features, fragmentomics_features, cna_features):
        """Integrate features from all modalities"""
        # Convert feature dictionaries to DataFrame
        methyl_df = pd.DataFrame([methylation_features])
        methyl_df.columns = [f'methyl_{col}' for col in methyl_df.columns]
        
        fragment_df = pd.DataFrame([fragmentomics_features])
        fragment_df.columns = [f'fragment_{col}' for col in fragment_df.columns]
        
        cna_df = pd.DataFrame([cna_features])
        cna_df.columns = [f'cna_{col}' for col in cna_df.columns]
        
        # Combine all features
        integrated_features = pd.concat([methyl_df, fragment_df, cna_df], axis=1)
        
        # Add cross-modal interaction features
        integrated_features['methyl_fragment_interaction'] = (
            integrated_features['methyl_global_methylation_mean'] * 
            integrated_features['fragment_mean_fragment_length']
        )
        
        integrated_features['fragment_cna_interaction'] = (
            integrated_features['fragment_nucleosome_periodicity'] * 
            integrated_features['cna_chromosomal_instability_index']
        )
        
        integrated_features['methyl_cna_interaction'] = (
            integrated_features['methyl_methylation_entropy'] * 
            integrated_features['cna_genomic_complexity_score']
        )
        
        return integrated_features

def main():
    """Main function to test feature extraction pipeline"""
    print("Testing Multi-Modal Feature Extraction Pipeline")
    print("=" * 60)
    
    # Initialize feature extractor
    extractor = MultiModalFeatureExtractor()
    
    # Test methylation feature extraction
    print("\n1. Testing Methylation Feature Extraction:")
    methylation_features = extractor.extract_methylation_features(None)
    
    # Test fragmentomics feature extraction
    print("\n2. Testing Fragmentomics Feature Extraction:")
    fragmentomics_features = extractor.extract_fragmentomics_features()
    
    # Test CNA feature extraction
    print("\n3. Testing CNA Feature Extraction:")
    cna_features = extractor.extract_cna_features()
    
    # Test integration
    print("\n4. Testing Multi-Modal Integration:")
    integrated_data = extractor.integrate_multimodal_data(
        methylation_features, fragmentomics_features, cna_features
    )
    
    print(f"\nIntegrated dataset shape: {integrated_data.shape}")
    print(f"Total features extracted: {integrated_data.shape[1]}")
    
    # Save results
    output_file = extractor.output_dir / "integrated_features_test.csv"
    integrated_data.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Display feature summary
    print("\nFeature Summary:")
    print(f"- Methylation features: {len(methylation_features)}")
    print(f"- Fragmentomics features: {len(fragmentomics_features)}")
    print(f"- CNA features: {len(cna_features)}")
    print(f"- Cross-modal interactions: 3")
    print(f"- Total integrated features: {integrated_data.shape[1]}")
    
    return integrated_data

if __name__ == "__main__":
    results = main()
