#!/usr/bin/env python3
"""
Focused Multi-Omics TCGA Processor
==================================

Focus on the omics types with clear TCGA sample ID mapping:
- Copy Number Variation (CNV) data
- Protein expression (RPPA) data

Create a robust, large-scale dataset from these two well-mapped omics types
and process mutations data separately where possible.

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import numpy as np
import pandas as pd
import pickle
import re
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import psutil
import time
import gc

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('focused_multi_omics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FocusedMultiOmicsProcessor:
    """Focused processor for CNV and protein data with mutations"""
    
    def __init__(self, base_dir: str = "data/production_tcga", 
                 output_dir: str = "data/focused_multi_omics"):
        
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # System resources
        self.cpu_count = mp.cpu_count()
        self.max_workers = min(self.cpu_count - 1, 8)
        
        logger.info(f"🎯 Focused Multi-Omics Processor initialized")
        logger.info(f"📁 Base directory: {base_dir}")
        logger.info(f"💻 System: {self.cpu_count} cores")
        logger.info(f"⚙️ Using {self.max_workers} workers")
    
    def extract_tcga_sample_id(self, filename: str) -> Optional[str]:
        """Extract TCGA sample ID from filename with multiple patterns"""
        patterns = [
            # Full TCGA sample ID: TCGA-XX-XXXX-XXX-XXX-XXXX-XX
            r'(TCGA-\w{2}-\w{4}-\w{2}[A-Z]-\w{2}[A-Z]-\w{4}-\w{2})',
            # TCGA case ID: TCGA-XX-XXXX
            r'(TCGA-\w{2}-\w{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                tcga_id = match.group(1).upper()
                # Standardize to case-level ID (first 12 characters)
                return tcga_id[:12] if len(tcga_id) > 12 else tcga_id
        
        return None
    
    def scan_copy_number_data(self) -> Dict[str, List[Path]]:
        """Scan copy number data directory"""
        logger.info("📂 Scanning copy number data...")
        
        cn_dir = self.base_dir / "copy_number"
        cn_files = {}
        
        if not cn_dir.exists():
            logger.warning("⚠️ Copy number directory not found")
            return cn_files
        
        for project_dir in cn_dir.iterdir():
            if not project_dir.is_dir() or not project_dir.name.startswith('TCGA-'):
                continue
            
            for file_path in project_dir.glob("*"):
                if not file_path.is_file() or file_path.stat().st_size == 0:
                    continue
                
                sample_id = self.extract_tcga_sample_id(file_path.name)
                if sample_id:
                    if sample_id not in cn_files:
                        cn_files[sample_id] = []
                    cn_files[sample_id].append(file_path)
        
        logger.info(f"  📊 Found {len(cn_files)} copy number samples, {sum(len(files) for files in cn_files.values())} files")
        return cn_files
    
    def scan_protein_data(self) -> Dict[str, List[Path]]:
        """Scan protein data directory"""
        logger.info("📂 Scanning protein data...")
        
        protein_dir = self.base_dir / "protein"
        protein_files = {}
        
        if not protein_dir.exists():
            logger.warning("⚠️ Protein directory not found")
            return protein_files
        
        for project_dir in protein_dir.iterdir():
            if not project_dir.is_dir() or not project_dir.name.startswith('TCGA-'):
                continue
            
            for file_path in project_dir.glob("*"):
                if not file_path.is_file() or file_path.stat().st_size == 0:
                    continue
                
                sample_id = self.extract_tcga_sample_id(file_path.name)
                if sample_id:
                    if sample_id not in protein_files:
                        protein_files[sample_id] = []
                    protein_files[sample_id].append(file_path)
        
        logger.info(f"  📊 Found {len(protein_files)} protein samples, {sum(len(files) for files in protein_files.values())} files")
        return protein_files
    
    def scan_mutation_data(self) -> Dict[str, List[Path]]:
        """Scan mutation data and try to extract TCGA IDs from content"""
        logger.info("📂 Scanning mutation data...")
        
        mut_dir = self.base_dir / "mutations"
        mut_files = {}
        
        if not mut_dir.exists():
            logger.warning("⚠️ Mutations directory not found")
            return mut_files
        
        # For mutations, we'll need to sample files and try to extract TCGA IDs
        sample_count = 0
        for project_dir in mut_dir.iterdir():
            if not project_dir.is_dir() or not project_dir.name.startswith('TCGA-'):
                continue
            
            project_name = project_dir.name
            
            for file_path in list(project_dir.glob("*"))[:100]:  # Sample first 100 files per project
                if not file_path.is_file() or file_path.stat().st_size == 0:
                    continue
                
                # Try to extract sample ID from filename first
                sample_id = self.extract_tcga_sample_id(file_path.name)
                if not sample_id:
                    # For mutations, use project name as fallback
                    sample_id = project_name
                
                if sample_id not in mut_files:
                    mut_files[sample_id] = []
                mut_files[sample_id].append(file_path)
                sample_count += 1
        
        logger.info(f"  📊 Found {len(mut_files)} mutation groups, {sample_count} files sampled")
        return mut_files
    
    def process_copy_number_file(self, file_path: Path) -> Optional[Dict[str, float]]:
        """Process copy number file with comprehensive feature extraction"""
        try:
            # Read the file
            if file_path.suffix.lower() in ['.tsv', '.txt']:
                df = pd.read_csv(file_path, sep='\t', low_memory=False, nrows=100000)
            else:
                return None
            
            if df.empty or df.shape[1] < 2:
                return None
            
            features = {}
            
            # Check for segmentation data format
            if 'Segment_Mean' in df.columns:
                seg_means = df['Segment_Mean'].dropna()
                
                features.update({
                    'cn_num_segments': len(df),
                    'cn_mean_value': seg_means.mean(),
                    'cn_median_value': seg_means.median(),
                    'cn_std_value': seg_means.std(),
                    'cn_min_value': seg_means.min(),
                    'cn_max_value': seg_means.max(),
                    'cn_amplifications': (seg_means > 0.3).sum(),
                    'cn_deletions': (seg_means < -0.3).sum(),
                    'cn_neutral': ((seg_means >= -0.3) & (seg_means <= 0.3)).sum(),
                    'cn_high_amp': (seg_means > 1.0).sum(),
                    'cn_deep_del': (seg_means < -1.0).sum(),
                })
                
                # Segment length features if available
                if 'Start' in df.columns and 'End' in df.columns:
                    lengths = df['End'] - df['Start']
                    lengths = lengths[lengths > 0]  # Valid lengths only
                    
                    if len(lengths) > 0:
                        features.update({
                            'cn_mean_seg_length': lengths.mean(),
                            'cn_median_seg_length': lengths.median(),
                            'cn_total_coverage': lengths.sum(),
                            'cn_long_segments': (lengths > 10000000).sum(),  # >10Mb
                            'cn_short_segments': (lengths < 1000000).sum(),  # <1Mb
                        })
            
            else:
                # Generic numeric processing
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    all_values = df[numeric_cols].values.flatten()
                    all_values = all_values[~np.isnan(all_values)]
                    
                    if len(all_values) > 0:
                        features.update({
                            'cn_mean': np.mean(all_values),
                            'cn_std': np.std(all_values),
                            'cn_median': np.median(all_values),
                            'cn_q75': np.percentile(all_values, 75),
                            'cn_q25': np.percentile(all_values, 25),
                            'cn_positive_ratio': (all_values > 0).mean(),
                            'cn_extreme_values': ((all_values > 2) | (all_values < -2)).sum(),
                            'cn_total_measurements': len(all_values)
                        })
            
            return features if features else None
            
        except Exception as e:
            logger.debug(f"Error processing CN file {file_path.name}: {e}")
            return None
    
    def process_protein_file(self, file_path: Path) -> Optional[Dict[str, float]]:
        """Process protein file with comprehensive feature extraction"""
        try:
            if not file_path.name.endswith(('.tsv', '.txt')):
                return None
            
            df = pd.read_csv(file_path, sep='\t', low_memory=False, nrows=10000)
            
            if df.empty:
                return None
            
            features = {}
            
            # Get numeric columns (protein expression values)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            protein_values = df[numeric_cols].values.flatten()
            protein_values = protein_values[~np.isnan(protein_values)]
            
            if len(protein_values) == 0:
                return None
            
            # Basic statistics
            features.update({
                'protein_mean': np.mean(protein_values),
                'protein_std': np.std(protein_values),
                'protein_median': np.median(protein_values),
                'protein_q75': np.percentile(protein_values, 75),
                'protein_q25': np.percentile(protein_values, 25),
                'protein_min': np.min(protein_values),
                'protein_max': np.max(protein_values),
                'protein_range': np.max(protein_values) - np.min(protein_values),
            })
            
            # Expression level features
            features.update({
                'protein_positive_ratio': (protein_values > 0).mean(),
                'protein_negative_ratio': (protein_values < 0).mean(),
                'protein_high_expr': (protein_values > np.percentile(protein_values, 90)).sum(),
                'protein_low_expr': (protein_values < np.percentile(protein_values, 10)).sum(),
                'protein_extreme_high': (protein_values > np.percentile(protein_values, 95)).sum(),
                'protein_extreme_low': (protein_values < np.percentile(protein_values, 5)).sum(),
                'protein_total_measured': len(protein_values),
            })
            
            # Variance and distribution features
            features.update({
                'protein_cv': np.std(protein_values) / np.mean(protein_values) if np.mean(protein_values) != 0 else 0,
                'protein_skewness': self.calculate_skewness(protein_values),
                'protein_kurtosis': self.calculate_kurtosis(protein_values),
                'protein_zero_ratio': (protein_values == 0).mean(),
            })
            
            return features
            
        except Exception as e:
            logger.debug(f"Error processing protein file {file_path.name}: {e}")
            return None
    
    def calculate_skewness(self, values):
        """Calculate skewness manually"""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0
            return np.mean(((values - mean_val) / std_val) ** 3)
        except:
            return 0
    
    def calculate_kurtosis(self, values):
        """Calculate kurtosis manually"""
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return 0
            return np.mean(((values - mean_val) / std_val) ** 4) - 3
        except:
            return 0
    
    def process_mutation_file(self, file_path: Path) -> Optional[Dict[str, float]]:
        """Process mutation file efficiently"""
        try:
            # Read MAF format
            df = pd.read_csv(file_path, sep='\t', low_memory=False, nrows=100000)
            
            if df.empty:
                return None
            
            features = {
                'mut_total_mutations': len(df),
            }
            
            # Gene-level features
            if 'Hugo_Symbol' in df.columns:
                features['mut_unique_genes'] = df['Hugo_Symbol'].nunique()
                features['mut_genes_multi_hit'] = (df['Hugo_Symbol'].value_counts() > 1).sum()
            
            # Mutation type features
            if 'Variant_Classification' in df.columns:
                mut_types = df['Variant_Classification'].value_counts()
                total_muts = len(df)
                
                for mut_type in ['Missense_Mutation', 'Silent', 'Nonsense_Mutation', 
                               'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site',
                               'In_Frame_Del', 'In_Frame_Ins']:
                    features[f'mut_{mut_type.lower()}_count'] = mut_types.get(mut_type, 0)
                    features[f'mut_{mut_type.lower()}_ratio'] = mut_types.get(mut_type, 0) / total_muts
            
            # Variant type features
            if 'Variant_Type' in df.columns:
                var_types = df['Variant_Type'].value_counts()
                for var_type in ['SNP', 'DNP', 'INS', 'DEL']:
                    features[f'mut_variant_{var_type.lower()}_count'] = var_types.get(var_type, 0)
            
            # Chromosome distribution
            if 'Chromosome' in df.columns:
                chr_counts = df['Chromosome'].value_counts()
                features['mut_chromosomes_affected'] = len(chr_counts)
                features['mut_max_chr_mutations'] = chr_counts.max() if len(chr_counts) > 0 else 0
            
            return features
            
        except Exception as e:
            logger.debug(f"Error processing mutation file {file_path.name}: {e}")
            return None
    
    def create_multi_omics_dataset(self):
        """Create the focused multi-omics dataset"""
        logger.info("🚀 Creating focused multi-omics dataset...")
        
        # Scan all data types
        cn_files = self.scan_copy_number_data()
        protein_files = self.scan_protein_data()
        mutation_files = self.scan_mutation_data()
        
        # Find overlapping samples
        cn_samples = set(cn_files.keys())
        protein_samples = set(protein_files.keys())
        mutation_samples = set(mutation_files.keys())
        
        # Multi-omics intersections
        cn_protein = cn_samples.intersection(protein_samples)
        cn_mutation = cn_samples.intersection(mutation_samples)
        protein_mutation = protein_samples.intersection(mutation_samples)
        all_three = cn_samples.intersection(protein_samples).intersection(mutation_samples)
        
        logger.info("🎯 Sample overlap analysis:")
        logger.info(f"  Copy Number samples: {len(cn_samples):,}")
        logger.info(f"  Protein samples: {len(protein_samples):,}")
        logger.info(f"  Mutation samples: {len(mutation_samples):,}")
        logger.info(f"  CN + Protein overlap: {len(cn_protein):,}")
        logger.info(f"  CN + Mutation overlap: {len(cn_mutation):,}")
        logger.info(f"  Protein + Mutation overlap: {len(protein_mutation):,}")
        logger.info(f"  All three overlap: {len(all_three):,}")
        
        # Process all samples with at least one data type
        all_samples = cn_samples.union(protein_samples).union(mutation_samples)
        logger.info(f"📊 Processing {len(all_samples):,} total samples...")
        
        # Process samples
        results = []
        
        for i, sample_id in enumerate(all_samples):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processing sample {i + 1:,}/{len(all_samples):,}")
            
            sample_data = {
                'sample_id': sample_id,
                'cancer_type': sample_id.split('-')[1] if len(sample_id.split('-')) > 1 else 'Unknown'
            }
            
            omics_count = 0
            
            # Process copy number
            if sample_id in cn_files:
                cn_features = None
                for file_path in cn_files[sample_id]:
                    cn_features = self.process_copy_number_file(file_path)
                    if cn_features:
                        break
                
                if cn_features:
                    sample_data.update(cn_features)
                    sample_data['copy_number_available'] = 1
                    omics_count += 1
                else:
                    sample_data['copy_number_available'] = 0
            else:
                sample_data['copy_number_available'] = 0
            
            # Process protein
            if sample_id in protein_files:
                protein_features = None
                for file_path in protein_files[sample_id]:
                    protein_features = self.process_protein_file(file_path)
                    if protein_features:
                        break
                
                if protein_features:
                    sample_data.update(protein_features)
                    sample_data['protein_available'] = 1
                    omics_count += 1
                else:
                    sample_data['protein_available'] = 0
            else:
                sample_data['protein_available'] = 0
            
            # Process mutations
            if sample_id in mutation_files:
                mut_features = None
                for file_path in mutation_files[sample_id]:
                    mut_features = self.process_mutation_file(file_path)
                    if mut_features:
                        break
                
                if mut_features:
                    sample_data.update(mut_features)
                    sample_data['mutations_available'] = 1
                    omics_count += 1
                else:
                    sample_data['mutations_available'] = 0
            else:
                sample_data['mutations_available'] = 0
            
            sample_data['omics_count'] = omics_count
            
            # Only include samples with at least one successfully processed omics type
            if omics_count > 0:
                results.append(sample_data)
        
        # Create DataFrame
        if results:
            df = pd.DataFrame(results)
            
            logger.info("✅ Dataset creation completed!")
            logger.info(f"  📊 Total samples: {len(df):,}")
            logger.info(f"  📈 Total features: {len(df.columns):,}")
            
            # Show omics coverage
            omics_coverage = df['omics_count'].value_counts().sort_index(ascending=False)
            logger.info("🎯 Multi-omics coverage:")
            for count, samples in omics_coverage.items():
                logger.info(f"  {count} omics: {samples:,} samples")
            
            # Show cancer type distribution
            if 'cancer_type' in df.columns:
                logger.info("🧬 Top cancer types:")
                for cancer_type, count in df['cancer_type'].value_counts().head(10).items():
                    logger.info(f"  TCGA-{cancer_type}: {count:,} samples")
            
            return df
        
        else:
            logger.error("❌ No samples processed successfully")
            return pd.DataFrame()
    
    def save_dataset(self, df: pd.DataFrame):
        """Save the dataset with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main dataset
        dataset_file = self.output_dir / f"tcga_focused_multi_omics_{timestamp}.csv"
        df.to_csv(dataset_file, index=False)
        logger.info(f"💾 Dataset saved: {dataset_file}")
        
        # Create summary
        summary = {
            'created': datetime.now().isoformat(),
            'total_samples': len(df),
            'total_features': len(df.columns),
            'omics_coverage': df['omics_count'].value_counts().to_dict() if 'omics_count' in df.columns else {},
            'cancer_types': df['cancer_type'].value_counts().to_dict() if 'cancer_type' in df.columns else {},
            'dataset_path': str(dataset_file),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        summary_file = self.output_dir / f"dataset_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Summary saved: {summary_file}")
        
        return dataset_file, summary_file


def main():
    """Main execution"""
    logger.info("🎯 Focused Multi-Omics TCGA Processor")
    logger.info("=" * 50)
    
    try:
        processor = FocusedMultiOmicsProcessor()
        
        # Create dataset
        dataset_df = processor.create_multi_omics_dataset()
        
        if not dataset_df.empty:
            # Save results
            dataset_file, summary_file = processor.save_dataset(dataset_df)
            
            logger.info("✅ SUCCESS: Focused multi-omics dataset created!")
            logger.info(f"📂 Dataset: {dataset_file}")
            logger.info(f"📋 Summary: {summary_file}")
            
            return dataset_df
        else:
            logger.error("❌ Failed to create dataset")
            return None
    
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        raise


if __name__ == "__main__":
    result = main()
