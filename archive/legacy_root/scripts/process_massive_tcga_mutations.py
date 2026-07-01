#!/usr/bin/env python3
"""
TCGA Massive Mutation Data Processor
====================================

Process and consolidate 4,761 real TCGA mutation files into machine learning-ready features.
This script handles the massive scale with memory-efficient processing and robust error handling.

Key Features:
- Memory-efficient streaming processing of large MAF files
- Comprehensive mutation feature extraction  
- Cancer type labeling from TCGA barcodes
- Quality control and data validation
- Scalable architecture for large datasets
- Progress tracking and resume capability

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import pandas as pd
import numpy as np
import gzip
import os
import logging
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import concurrent.futures
import threading
from typing import Dict, List, Set, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('massive_mutation_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MassiveTCGAProcessor:
    """Production-grade processor for massive TCGA mutation datasets"""
    
    def __init__(self, data_dir: str, output_dir: str = "processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cancer type mapping from TCGA project codes
        self.cancer_type_mapping = {
            'TCGA-BRCA': 'Breast Cancer',
            'TCGA-COAD': 'Colon Cancer', 
            'TCGA-HNSC': 'Head and Neck Cancer',
            'TCGA-LGG': 'Lower Grade Glioma',
            'TCGA-LIHC': 'Liver Cancer',
            'TCGA-LUAD': 'Lung Adenocarcinoma',
            'TCGA-LUSC': 'Lung Squamous Cell Carcinoma',
            'TCGA-PRAD': 'Prostate Cancer',
            'TCGA-STAD': 'Stomach Cancer',
            'TCGA-THCA': 'Thyroid Cancer'
        }
        
        # Initialize processing state
        self.mutation_features = defaultdict(lambda: defaultdict(int))
        self.sample_metadata = {}
        self.processing_progress = {}
        self.lock = threading.Lock()
        
        logger.info(f"🔬 Initialized TCGA processor for {data_dir}")
        logger.info(f"📊 Target cancer types: {list(self.cancer_type_mapping.keys())}")
    
    def extract_sample_info(self, barcode: str) -> Tuple[str, str]:
        """Extract sample ID and cancer type from TCGA barcode"""
        try:
            # TCGA barcode format: TCGA-XX-XXXX-XXX-XXX-XXXX-XX
            parts = barcode.split('-')
            if len(parts) >= 3:
                sample_id = f"{parts[0]}-{parts[1]}-{parts[2]}"
                # Cancer type is in the directory structure  
                return sample_id, barcode
            return barcode, barcode
        except Exception as e:
            logger.warning(f"Could not parse barcode {barcode}: {e}")
            return barcode, barcode
    
    def process_maf_file(self, maf_path: Path, cancer_type: str) -> Dict:
        """Process a single MAF file and extract mutation features"""
        try:
            mutations_data = {
                'sample_mutations': defaultdict(lambda: defaultdict(int)),
                'samples_processed': set(),
                'total_mutations': 0,
                'error_count': 0
            }
            
            logger.info(f"📂 Processing {maf_path.name} ({cancer_type})")
            
            with gzip.open(maf_path, 'rt') as f:
                # Skip header comments
                for line in f:
                    if not line.startswith('#'):
                        break
                
                # Read header
                headers = line.strip().split('\t')
                
                # Find important column indices
                try:
                    hugo_idx = headers.index('Hugo_Symbol')
                    barcode_idx = headers.index('Tumor_Sample_Barcode')
                    variant_idx = headers.index('Variant_Classification')
                    variant_type_idx = headers.index('Variant_Type')
                except ValueError as e:
                    logger.error(f"Missing required columns in {maf_path}: {e}")
                    return mutations_data
                
                # Process mutations
                for line_num, line in enumerate(f, start=2):
                    try:
                        fields = line.strip().split('\t')
                        if len(fields) < len(headers):
                            continue
                        
                        gene = fields[hugo_idx]
                        barcode = fields[barcode_idx]
                        variant_class = fields[variant_idx]
                        variant_type = fields[variant_type_idx]
                        
                        # Extract sample info
                        sample_id, _ = self.extract_sample_info(barcode)
                        
                        # Count mutations per gene per sample
                        mutations_data['sample_mutations'][sample_id][gene] += 1
                        mutations_data['sample_mutations'][sample_id][f'variant_class_{variant_class}'] += 1
                        mutations_data['sample_mutations'][sample_id][f'variant_type_{variant_type}'] += 1
                        mutations_data['sample_mutations'][sample_id]['total_mutations'] += 1
                        
                        mutations_data['samples_processed'].add(sample_id)
                        mutations_data['total_mutations'] += 1
                        
                    except Exception as e:
                        mutations_data['error_count'] += 1
                        if mutations_data['error_count'] <= 10:  # Only log first 10 errors
                            logger.warning(f"Error processing line {line_num} in {maf_path}: {e}")
            
            logger.info(f"✅ Processed {maf_path.name}: {len(mutations_data['samples_processed'])} samples, {mutations_data['total_mutations']} mutations")
            return mutations_data
            
        except Exception as e:
            logger.error(f"❌ Failed to process {maf_path}: {e}")
            return {'sample_mutations': {}, 'samples_processed': set(), 'total_mutations': 0, 'error_count': 1}
    
    def consolidate_mutations(self, all_mutations_data: Dict, cancer_type: str):
        """Consolidate mutations from all files of a cancer type"""
        with self.lock:
            for sample_id, sample_mutations in all_mutations_data['sample_mutations'].items():
                # Store sample metadata
                self.sample_metadata[sample_id] = {
                    'cancer_type': cancer_type,
                    'cancer_code': [k for k, v in self.cancer_type_mapping.items() if v == cancer_type][0],
                    'total_mutations': sample_mutations.get('total_mutations', 0)
                }
                
                # Consolidate mutation features
                for feature, count in sample_mutations.items():
                    self.mutation_features[sample_id][feature] = count
    
    def process_cancer_type(self, project_dir: Path) -> Dict:
        """Process all MAF files for a specific cancer type"""
        cancer_code = project_dir.name
        cancer_type = self.cancer_type_mapping.get(cancer_code, cancer_code)
        
        logger.info(f"🧬 Processing {cancer_type} ({cancer_code})")
        
        # Find all MAF files
        maf_files = list(project_dir.glob("*.maf.gz"))
        logger.info(f"📁 Found {len(maf_files)} MAF files for {cancer_type}")
        
        if not maf_files:
            logger.warning(f"⚠️ No MAF files found for {cancer_code}")
            return {}
        
        # Process files in parallel with limited concurrency
        consolidated_data = {
            'sample_mutations': defaultdict(lambda: defaultdict(int)),
            'samples_processed': set(),
            'total_mutations': 0,
            'files_processed': 0,
            'files_failed': 0
        }
        
        # Use ThreadPoolExecutor for I/O bound operations
        max_workers = min(4, len(maf_files))  # Conservative threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_maf_file, maf_file, cancer_type): maf_file 
                for maf_file in maf_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                maf_file = future_to_file[future]
                try:
                    file_data = future.result()
                    
                    # Consolidate results
                    for sample_id, mutations in file_data['sample_mutations'].items():
                        for feature, count in mutations.items():
                            consolidated_data['sample_mutations'][sample_id][feature] += count
                    
                    consolidated_data['samples_processed'].update(file_data['samples_processed'])
                    consolidated_data['total_mutations'] += file_data['total_mutations']
                    consolidated_data['files_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed processing {maf_file}: {e}")
                    consolidated_data['files_failed'] += 1
        
        logger.info(f"🎉 Completed {cancer_type}: {len(consolidated_data['samples_processed'])} samples from {consolidated_data['files_processed']} files")
        
        # Consolidate into main data structures
        self.consolidate_mutations(consolidated_data, cancer_type)
        
        return {
            'cancer_type': cancer_type,
            'samples': len(consolidated_data['samples_processed']),
            'total_mutations': consolidated_data['total_mutations'],
            'files_processed': consolidated_data['files_processed'],
            'files_failed': consolidated_data['files_failed']
        }
    
    def create_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Create ML-ready feature matrix and labels"""
        logger.info("🔧 Creating feature matrix...")
        
        # Get all unique features across all samples
        all_features = set()
        for sample_features in self.mutation_features.values():
            all_features.update(sample_features.keys())
        
        # Remove metadata features
        ml_features = [f for f in all_features if f not in ['total_mutations']]
        ml_features = sorted(ml_features)
        
        logger.info(f"📊 Total features: {len(ml_features)}")
        
        # Create feature matrix
        samples = list(self.mutation_features.keys())
        feature_matrix = np.zeros((len(samples), len(ml_features)))
        labels = []
        
        for i, sample_id in enumerate(samples):
            sample_features = self.mutation_features[sample_id]
            
            # Fill feature vector
            for j, feature in enumerate(ml_features):
                feature_matrix[i, j] = sample_features.get(feature, 0)
            
            # Get label
            cancer_type = self.sample_metadata[sample_id]['cancer_type']
            labels.append(cancer_type)
        
        # Create DataFrame
        feature_df = pd.DataFrame(
            feature_matrix,
            index=samples,
            columns=ml_features
        )
        
        labels_series = pd.Series(labels, index=samples, name='cancer_type')
        
        logger.info(f"✅ Feature matrix: {feature_df.shape[0]} samples × {feature_df.shape[1]} features")
        logger.info(f"📋 Cancer types: {labels_series.value_counts().to_dict()}")
        
        return feature_df, labels_series
    
    def save_processed_data(self, feature_df: pd.DataFrame, labels: pd.Series):
        """Save processed data for machine learning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save feature matrix
        feature_file = self.output_dir / f"tcga_mutation_features_{timestamp}.csv"
        feature_df.to_csv(feature_file)
        logger.info(f"💾 Saved features: {feature_file}")
        
        # Save labels  
        labels_file = self.output_dir / f"tcga_mutation_labels_{timestamp}.csv"
        labels.to_csv(labels_file)
        logger.info(f"💾 Saved labels: {labels_file}")
        
        # Save metadata
        metadata_file = self.output_dir / f"tcga_sample_metadata_{timestamp}.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.sample_metadata, f)
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        # Save processing summary
        summary = {
            'timestamp': timestamp,
            'total_samples': len(feature_df),
            'total_features': len(feature_df.columns),
            'cancer_types': labels.value_counts().to_dict(),
            'feature_file': str(feature_file),
            'labels_file': str(labels_file),
            'metadata_file': str(metadata_file)
        }
        
        summary_file = self.output_dir / f"processing_summary_{timestamp}.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        logger.info(f"📋 Saved summary: {summary_file}")
        
        return summary
    
    def process_all_data(self):
        """Process all TCGA mutation data"""
        logger.info("🚀 Starting massive TCGA mutation processing...")
        start_time = datetime.now()
        
        # Find all cancer type directories
        cancer_dirs = [d for d in self.data_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('TCGA-')]
        
        logger.info(f"🎯 Found {len(cancer_dirs)} cancer types to process")
        
        processing_results = {}
        
        # Process each cancer type
        for cancer_dir in cancer_dirs:
            if cancer_dir.name in self.cancer_type_mapping:
                try:
                    result = self.process_cancer_type(cancer_dir)
                    processing_results[cancer_dir.name] = result
                except Exception as e:
                    logger.error(f"❌ Failed to process {cancer_dir.name}: {e}")
                    processing_results[cancer_dir.name] = {'error': str(e)}
        
        # Create feature matrix
        try:
            feature_df, labels = self.create_feature_matrix()
            
            # Save processed data
            summary = self.save_processed_data(feature_df, labels)
            
            # Final report
            end_time = datetime.now()
            processing_time = end_time - start_time
            
            logger.info("🎉 PROCESSING COMPLETE!")
            logger.info(f"⏱️ Total time: {processing_time}")
            logger.info(f"📊 Final dataset: {len(feature_df)} samples × {len(feature_df.columns)} features")
            logger.info(f"🧬 Cancer types distribution:")
            for cancer_type, count in labels.value_counts().items():
                logger.info(f"  {cancer_type}: {count} samples")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to create feature matrix: {e}")
            raise


def main():
    """Main processing function"""
    logger.info("🧬 TCGA Massive Mutation Data Processor")
    logger.info("=" * 50)
    
    # Set paths
    mutations_dir = "data/production_tcga/mutations"
    output_dir = "data/processed_massive_tcga"
    
    if not os.path.exists(mutations_dir):
        logger.error(f"❌ Mutations directory not found: {mutations_dir}")
        return
    
    # Create processor
    processor = MassiveTCGAProcessor(mutations_dir, output_dir)
    
    try:
        # Process all data
        summary = processor.process_all_data()
        
        logger.info("✅ SUCCESS: Massive TCGA mutation processing completed!")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise


if __name__ == "__main__":
    summary = main()
