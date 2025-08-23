#!/usr/bin/env python3
"""
Large-Scale Multi-Omics TCGA Data Processor
==========================================

Scale up multi-omics processing by focusing on filename-based sample matching
and processing the largest available dataset with maximum efficiency.

Strategy:
1. Identify omics types with direct TCGA ID mapping (copy_number, protein, some mutations)
2. Process samples in large batches with parallel processing
3. Create comprehensive feature matrices for machine learning
4. Focus on multi-omics samples that have data across multiple modalities

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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        logging.FileHandler('scale_multi_omics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LargeScaleMultiOmicsProcessor:
    """Large-scale multi-omics processor with aggressive parallelization"""
    
    def __init__(self, base_dir: str = "data/production_tcga", 
                 output_dir: str = "data/massive_multi_omics", 
                 cache_dir: str = "cache"):
        
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # System resources
        self.total_memory = psutil.virtual_memory().total // (1024**3)  # GB
        self.cpu_count = mp.cpu_count()
        self.max_workers = min(self.cpu_count - 1, 8)  # Leave one core free
        
        # Processing configuration
        self.batch_size = 1000  # Files per batch
        self.chunk_size = 100   # Samples per processing chunk
        self.memory_limit_gb = max(4, self.total_memory // 2)  # Use half available RAM
        
        # Sample tracking
        self.sample_registry = {}
        self.omics_file_counts = defaultdict(int)
        self.processed_samples = set()
        
        logger.info(f"🚀 Large-Scale Multi-Omics Processor initialized")
        logger.info(f"💻 System: {self.cpu_count} cores, {self.total_memory}GB RAM")
        logger.info(f"⚙️ Config: {self.max_workers} workers, {self.batch_size} batch size")
        logger.info(f"🧠 Memory limit: {self.memory_limit_gb}GB")
    
    def extract_tcga_sample_id(self, filename: str) -> Optional[str]:
        """Extract TCGA sample ID from filename with multiple patterns"""
        patterns = [
            # Full TCGA sample ID: TCGA-XX-XXXX-XXX-XXX-XXXX-XX
            r'(TCGA-\w{2}-\w{4}-\w{2}[A-Z]-\w{2}[A-Z]-\w{4}-\w{2})',
            # TCGA case ID: TCGA-XX-XXXX
            r'(TCGA-\w{2}-\w{4})',
            # Alternative patterns found in files
            r'(TCGA\.\w{2}\.\w{4})',  # Dot notation
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    def scan_omics_directories(self) -> Dict[str, Dict[str, List[Path]]]:
        """Scan all omics directories and catalog files by sample ID"""
        logger.info("🔍 Scanning omics directories for comprehensive file catalog...")
        
        omics_types = ['mutations', 'expression', 'copy_number', 'methylation', 'protein']
        sample_files = defaultdict(lambda: defaultdict(list))
        file_counts = defaultdict(int)
        
        for omics_type in omics_types:
            omics_dir = self.base_dir / omics_type
            if not omics_dir.exists():
                logger.warning(f"⚠️ Directory not found: {omics_dir}")
                continue
            
            logger.info(f"📂 Scanning {omics_type}...")
            
            # Scan all project directories
            for project_dir in omics_dir.iterdir():
                if not project_dir.is_dir() or not project_dir.name.startswith('TCGA-'):
                    continue
                
                # Process all files in project
                for file_path in project_dir.glob("*"):
                    if not file_path.is_file() or file_path.stat().st_size == 0:
                        continue
                    
                    # Extract sample ID
                    sample_id = self.extract_tcga_sample_id(file_path.name)
                    if sample_id:
                        sample_files[sample_id][omics_type].append(file_path)
                        file_counts[omics_type] += 1
        
        # Log statistics
        logger.info("📊 File catalog statistics:")
        for omics_type, count in file_counts.items():
            logger.info(f"  {omics_type}: {count:,} files")
        
        logger.info(f"🧬 Total samples with files: {len(sample_files):,}")
        
        # Calculate multi-omics coverage
        coverage_stats = defaultdict(int)
        for sample_id, omics_data in sample_files.items():
            coverage = len(omics_data)
            coverage_stats[coverage] += 1
        
        logger.info("🎯 Multi-omics coverage:")
        for coverage, count in sorted(coverage_stats.items(), reverse=True):
            logger.info(f"  {coverage} omics: {count:,} samples")
        
        return dict(sample_files)
    
    def process_copy_number_file(self, file_path: Path) -> Optional[pd.Series]:
        """Process a single copy number file efficiently"""
        try:
            # Determine file format and read appropriately
            if file_path.suffix.lower() in ['.tsv', '.txt']:
                df = pd.read_csv(file_path, sep='\t', low_memory=False, nrows=10000)
            else:
                return None
            
            if df.empty or df.shape[1] < 2:
                return None
            
            # Extract meaningful features based on common TCGA copy number formats
            if 'Segment_Mean' in df.columns:
                # Segmentation data
                features = {
                    'cn_num_segments': len(df),
                    'cn_mean_segment_length': df['End'].sub(df['Start']).mean() if 'End' in df.columns and 'Start' in df.columns else 0,
                    'cn_mean_value': df['Segment_Mean'].mean(),
                    'cn_std_value': df['Segment_Mean'].std(),
                    'cn_amplifications': (df['Segment_Mean'] > 0.3).sum(),
                    'cn_deletions': (df['Segment_Mean'] < -0.3).sum(),
                    'cn_neutral': ((df['Segment_Mean'] >= -0.3) & (df['Segment_Mean'] <= 0.3)).sum()
                }
            else:
                # Generic numeric data processing
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data_values = df[numeric_cols].values.flatten()
                    data_values = data_values[~np.isnan(data_values)]
                    
                    if len(data_values) > 0:
                        features = {
                            'cn_mean': np.mean(data_values),
                            'cn_std': np.std(data_values),
                            'cn_median': np.median(data_values),
                            'cn_q75': np.percentile(data_values, 75),
                            'cn_q25': np.percentile(data_values, 25),
                            'cn_positive_ratio': (data_values > 0).mean(),
                            'cn_high_values': (data_values > 1.0).sum(),
                            'cn_low_values': (data_values < -1.0).sum(),
                            'cn_total_features': len(data_values)
                        }
                    else:
                        return None
                else:
                    return None
            
            return pd.Series(features)
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path.name}: {e}")
            return None
    
    def process_protein_file(self, file_path: Path) -> Optional[pd.Series]:
        """Process a single protein file efficiently"""
        try:
            if not file_path.name.endswith(('.tsv', '.txt')):
                return None
            
            df = pd.read_csv(file_path, sep='\t', low_memory=False, nrows=5000)
            
            if df.empty:
                return None
            
            # Extract protein expression features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            
            # Get protein expression values
            protein_values = df[numeric_cols].values.flatten()
            protein_values = protein_values[~np.isnan(protein_values)]
            
            if len(protein_values) == 0:
                return None
            
            features = {
                'protein_mean': np.mean(protein_values),
                'protein_std': np.std(protein_values),
                'protein_median': np.median(protein_values),
                'protein_q75': np.percentile(protein_values, 75),
                'protein_q25': np.percentile(protein_values, 25),
                'protein_positive_ratio': (protein_values > 0).mean(),
                'protein_high_expr': (protein_values > np.percentile(protein_values, 90)).sum(),
                'protein_low_expr': (protein_values < np.percentile(protein_values, 10)).sum(),
                'protein_total_measured': len(protein_values)
            }
            
            return pd.Series(features)
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path.name}: {e}")
            return None
    
    def process_mutation_file(self, file_path: Path) -> Optional[pd.Series]:
        """Process mutation file efficiently"""
        try:
            # Read MAF format file
            df = pd.read_csv(file_path, sep='\t', low_memory=False, nrows=50000)
            
            if df.empty or len(df) == 0:
                return None
            
            # Extract mutation features
            features = {
                'mut_total_mutations': len(df),
                'mut_unique_genes': df['Hugo_Symbol'].nunique() if 'Hugo_Symbol' in df.columns else 0,
            }
            
            # Mutation types
            if 'Variant_Classification' in df.columns:
                mut_types = df['Variant_Classification'].value_counts()
                for mut_type in ['Missense_Mutation', 'Silent', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']:
                    features[f'mut_{mut_type.lower()}'] = mut_types.get(mut_type, 0)
            
            # Variant types
            if 'Variant_Type' in df.columns:
                variant_types = df['Variant_Type'].value_counts()
                for var_type in ['SNP', 'DNP', 'INS', 'DEL']:
                    features[f'mut_variant_{var_type.lower()}'] = variant_types.get(var_type, 0)
            
            return pd.Series(features)
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing {file_path.name}: {e}")
            return None
    
    def process_sample_batch(self, sample_batch: List[Tuple[str, Dict[str, List[Path]]]]) -> pd.DataFrame:
        """Process a batch of samples to extract multi-omics features"""
        logger.info(f"🔄 Processing batch of {len(sample_batch)} samples...")
        
        results = []
        
        for sample_id, omics_files in sample_batch:
            try:
                sample_features = {'sample_id': sample_id}
                
                # Extract cancer type from sample ID
                if sample_id.startswith('TCGA-'):
                    try:
                        project_code = sample_id.split('-')[1]
                        sample_features['cancer_type'] = f'TCGA-{project_code}'
                    except:
                        sample_features['cancer_type'] = 'Unknown'
                
                # Process each omics type
                for omics_type, file_list in omics_files.items():
                    if not file_list:
                        continue
                    
                    # Process the first (representative) file for each omics type
                    file_path = file_list[0]
                    
                    if omics_type == 'copy_number':
                        features = self.process_copy_number_file(file_path)
                    elif omics_type == 'protein':
                        features = self.process_protein_file(file_path)
                    elif omics_type == 'mutations':
                        features = self.process_mutation_file(file_path)
                    else:
                        # Skip expression and methylation for now due to UUID complexity
                        continue
                    
                    if features is not None:
                        # Add omics type prefix to feature names
                        prefixed_features = {f'{omics_type}_{key}': value for key, value in features.items()}
                        sample_features.update(prefixed_features)
                        sample_features[f'{omics_type}_available'] = 1
                    else:
                        sample_features[f'{omics_type}_available'] = 0
                
                # Only include samples with at least 2 omics types
                omics_count = sum(1 for k, v in sample_features.items() 
                                if k.endswith('_available') and v == 1)
                
                if omics_count >= 2:
                    sample_features['omics_count'] = omics_count
                    results.append(sample_features)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing sample {sample_id}: {e}")
                continue
        
        if results:
            df = pd.DataFrame(results)
            logger.info(f"✅ Batch processed: {len(df)} multi-omics samples")
            return df
        else:
            logger.warning("⚠️ No multi-omics samples in this batch")
            return pd.DataFrame()
    
    def create_massive_dataset(self) -> pd.DataFrame:
        """Create the largest possible multi-omics dataset"""
        logger.info("🚀 Creating massive multi-omics dataset...")
        
        # Scan all files
        sample_files = self.scan_omics_directories()
        
        # Filter for multi-omics samples only
        multi_omics_samples = {
            sample_id: omics_data 
            for sample_id, omics_data in sample_files.items() 
            if len(omics_data) >= 2  # At least 2 omics types
        }
        
        logger.info(f"🎯 Processing {len(multi_omics_samples):,} multi-omics samples...")
        
        # Convert to list for batch processing
        sample_list = list(multi_omics_samples.items())
        
        # Process in parallel batches
        all_results = []
        batch_size = self.batch_size
        
        # Use ProcessPoolExecutor for CPU-intensive tasks
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create batches
            batches = [sample_list[i:i + batch_size] 
                      for i in range(0, len(sample_list), batch_size)]
            
            logger.info(f"📊 Processing {len(batches)} batches with {self.max_workers} workers...")
            
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_sample_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_df = future.result()
                    if not batch_df.empty:
                        all_results.append(batch_df)
                        logger.info(f"✅ Completed batch {batch_idx + 1}/{len(batches)}: "
                                  f"{len(batch_df)} samples processed")
                    
                    # Periodic memory cleanup
                    if len(all_results) % 5 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx + 1} failed: {e}")
        
        # Combine all results
        if all_results:
            logger.info("🔗 Combining all batch results...")
            final_df = pd.concat(all_results, ignore_index=True)
            
            # Clean up memory
            del all_results
            gc.collect()
            
            logger.info(f"✅ Dataset created: {len(final_df):,} samples, {len(final_df.columns):,} features")
            
            return final_df
        else:
            logger.error("❌ No samples processed successfully")
            return pd.DataFrame()
    
    def save_dataset(self, df: pd.DataFrame, suffix: str = "massive"):
        """Save the dataset with comprehensive metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main dataset
        dataset_file = self.output_dir / f"tcga_multi_omics_{suffix}_{timestamp}.csv"
        df.to_csv(dataset_file, index=False)
        logger.info(f"💾 Dataset saved: {dataset_file}")
        
        # Save metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'samples': len(df),
            'features': len(df.columns),
            'omics_types_available': [col.replace('_available', '') for col in df.columns if col.endswith('_available')],
            'cancer_types': df['cancer_type'].value_counts().to_dict() if 'cancer_type' in df.columns else {},
            'omics_coverage': df['omics_count'].value_counts().to_dict() if 'omics_count' in df.columns else {},
            'file_path': str(dataset_file),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        metadata_file = self.output_dir / f"metadata_{suffix}_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📋 Metadata saved: {metadata_file}")
        
        # Log final statistics
        logger.info("📊 Final Dataset Statistics:")
        logger.info(f"  📁 Samples: {metadata['samples']:,}")
        logger.info(f"  📈 Features: {metadata['features']:,}")
        logger.info(f"  💾 Size: {metadata['memory_usage_mb']:.1f} MB")
        
        if 'cancer_type' in df.columns:
            logger.info("🎯 Cancer Type Distribution:")
            for cancer_type, count in df['cancer_type'].value_counts().head(10).items():
                logger.info(f"  {cancer_type}: {count:,} samples")
        
        return dataset_file


def main():
    """Main execution function"""
    logger.info("🚀 Large-Scale Multi-Omics TCGA Processor")
    logger.info("=" * 60)
    
    try:
        # Initialize processor
        processor = LargeScaleMultiOmicsProcessor()
        
        # Create massive dataset
        dataset_df = processor.create_massive_dataset()
        
        if not dataset_df.empty:
            # Save the dataset
            dataset_file = processor.save_dataset(dataset_df)
            
            logger.info("✅ SUCCESS: Massive multi-omics dataset created!")
            logger.info(f"📂 Output file: {dataset_file}")
            
            return dataset_df
        else:
            logger.error("❌ Failed to create dataset")
            return None
            
    except KeyboardInterrupt:
        logger.info("⏸️ Processing interrupted by user")
        return None
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise


if __name__ == "__main__":
    result = main()
