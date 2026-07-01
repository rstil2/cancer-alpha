#!/usr/bin/env python3
"""
Scale Real TCGA Processing
==========================
Process a large, balanced sample of authentic TCGA data for model training.
Ensures 100% real data with zero synthetic contamination.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
import random
from process_authentic_tcga_55k import AuthenticTCGAProcessor

# Set environment for real data only
os.environ['ONCURA_REAL_DATA_ONLY'] = '1'

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScalableRealProcessor(AuthenticTCGAProcessor):
    """Enhanced processor for large-scale real TCGA data"""
    
    def discover_balanced_files(self, samples_per_cancer=200, focus_cancers=None):
        """Discover files with balanced sampling for scaling"""
        logger.info("🔍 Discovering files for large-scale processing...")
        
        if focus_cancers is None:
            # Focus on the 8 main cancer types from WARP.md
            focus_cancers = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'HNSC', 'LUSC', 'LIHC']
        
        cancer_files = {}
        total_available = {}
        
        # Walk through data directory
        for root, dirs, files in os.walk(self.raw_data_dir):
            for file in files:
                if file.endswith('.tsv') and 'augmented_star_gene_counts' in file:
                    file_path = Path(root) / file
                    cancer_type = self.extract_cancer_type_from_path(file_path)
                    
                    if cancer_type:
                        if cancer_type not in total_available:
                            total_available[cancer_type] = 0
                        total_available[cancer_type] += 1
                        
                        if cancer_type in focus_cancers:
                            if cancer_type not in cancer_files:
                                cancer_files[cancer_type] = []
                            cancer_files[cancer_type].append(file_path)
        
        logger.info("📊 Available files by cancer type:")
        for cancer in sorted(total_available.keys()):
            available = total_available[cancer]
            focused = len(cancer_files.get(cancer, []))
            status = '✅' if cancer in focus_cancers else '⚪'
            logger.info(f"  {status} {cancer}: {available:,} total ({focused:,} selected)")
        
        # Sample balanced files
        selected_files = []
        cancer_sample_counts = {}
        
        for cancer_type in focus_cancers:
            if cancer_type in cancer_files:
                available_files = cancer_files[cancer_type]
                n_sample = min(len(available_files), samples_per_cancer)
                sampled = random.sample(available_files, n_sample)
                selected_files.extend(sampled)
                cancer_sample_counts[cancer_type] = len(sampled)
                logger.info(f"📋 {cancer_type}: Selected {len(sampled)} files")
        
        random.shuffle(selected_files)
        
        logger.info(f"🎯 Total files selected: {len(selected_files)}")
        logger.info(f"📊 Expected samples: ~{len(selected_files)} (after processing)")
        
        return selected_files, cancer_sample_counts

def main():
    """Main processing function"""
    print('🚀 SCALING UP REAL TCGA PROCESSING')
    print('=' * 50)
    print(f'Environment: ONCURA_REAL_DATA_ONLY = {os.environ.get("ONCURA_REAL_DATA_ONLY", "NOT SET")}')
    
    # Create output directory
    output_dir = Path('data/real_tcga_large')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processor
    logger.info('🔧 Initializing scaled processor...')
    processor = ScalableRealProcessor()
    
    # Discover balanced files for large dataset
    target_files, expected_counts = processor.discover_balanced_files(
        samples_per_cancer=150,  # 150 samples per cancer type = ~1200 total
        focus_cancers=['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'HNSC', 'LUSC', 'LIHC']  # 8 cancer types
    )
    
    logger.info(f'\n🚀 Processing {len(target_files)} real TCGA files...')
    
    # Process files with progress tracking
    results = []
    cancer_counts = {}
    processed_count = 0
    failed_count = 0
    
    for i, file_path in enumerate(target_files):
        try:
            result = processor.process_single_tsv_file(file_path)
            if result:
                results.append(result)
                ct = result['cancer_type']
                cancer_counts[ct] = cancer_counts.get(ct, 0) + 1
                processed_count += 1
                
                if processed_count % 50 == 0:
                    progress = (i + 1) / len(target_files) * 100
                    logger.info(f'  ✅ Processed {processed_count} samples ({progress:.1f}% complete)')
                    
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Show first 5 errors only
                logger.warning(f'  ⚠️ Error processing {file_path.name}: {str(e)[:100]}')
    
    logger.info(f'\n✅ PROCESSING COMPLETE!')
    logger.info(f'📊 Successfully processed: {processed_count} samples')
    logger.info(f'❌ Failed processing: {failed_count} files')
    logger.info(f'📈 Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%')
    
    logger.info(f'\n🧬 Final cancer type distribution:')
    for cancer, count in sorted(cancer_counts.items()):
        logger.info(f'  {cancer}: {count} samples')
    
    if len(results) >= 100:  # Need minimum samples for ML
        logger.info(f'\n📊 Creating large-scale ML dataset...')
        X, y, feature_names = processor.create_ml_dataset(results)
        
        logger.info(f'✅ Dataset created: {X.shape[0]} samples × {X.shape[1]} features')
        logger.info(f'🎯 Cancer types: {len(set(y))} unique types')
        
        # Save the large dataset
        dataset_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        labels_df = pd.DataFrame({
            'cancer_type_encoded': y,
            'cancer_type': [list(cancer_counts.keys())[yi] for yi in y]
        })
        
        # Save to CSV files
        dataset_path = output_dir / 'real_tcga_features.csv'
        labels_path = output_dir / 'real_tcga_labels.csv'
        metadata_path = output_dir / 'dataset_metadata.json'
        
        dataset_df.to_csv(dataset_path, index=False)
        labels_df.to_csv(labels_path, index=False)
        
        # Save metadata
        metadata = {
            'dataset_creation_date': datetime.now().isoformat(),
            'total_samples': len(results),
            'total_features': X.shape[1],
            'cancer_type_distribution': cancer_counts,
            'n_cancer_types': len(set(y)),
            'processing_success_rate': processed_count/(processed_count+failed_count),
            'data_source': 'authentic_tcga_only',
            'synthetic_data_used': False,
            'oncura_real_data_only': True
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f'\n💾 SAVED LARGE REAL DATASET:')
        logger.info(f'   Features: {dataset_path}')
        logger.info(f'   Labels: {labels_path}')
        logger.info(f'   Metadata: {metadata_path}')
        logger.info(f'\n🎉 READY FOR MODEL TRAINING!')
        
        return True
        
    else:
        logger.error(f'❌ Insufficient samples ({len(results)}) for ML dataset creation')
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)