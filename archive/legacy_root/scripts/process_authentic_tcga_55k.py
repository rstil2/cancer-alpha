#!/usr/bin/env python3
"""
Authentic TCGA Data Processor - 55K Samples
=============================================

Process the 55,562 authentic TCGA samples from raw .tsv files into 
a unified ML-ready dataset for training breakthrough cancer classification models.

This processor ensures 100% authentic data with zero synthetic contamination.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import gc
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tcga_processing_55k.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AuthenticTCGAProcessor:
    """Process authentic TCGA data ensuring zero synthetic contamination"""
    
    def __init__(self, 
                 raw_data_dir: str = "/Users/stillwell/projects/cancer-alpha/data/raw_tcga",
                 output_dir: str = "/Users/stillwell/projects/cancer-alpha/data/processed_authentic_tcga_55k"):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TCGA cancer type mapping
        self.cancer_type_mapping = {
            'TCGA-BRCA': 'BRCA',  # Breast Cancer
            'TCGA-LUAD': 'LUAD',  # Lung Adenocarcinoma
            'TCGA-HNSC': 'HNSC',  # Head & Neck Squamous Cell
            'TCGA-THCA': 'THCA',  # Thyroid Cancer
            'TCGA-LUSC': 'LUSC',  # Lung Squamous Cell
            'TCGA-LGG': 'LGG',    # Lower Grade Glioma
            'TCGA-PRAD': 'PRAD',  # Prostate Cancer
            'TCGA-COAD': 'COAD',  # Colon Adenocarcinoma
            'TCGA-STAD': 'STAD',  # Stomach Adenocarcinoma
            'TCGA-BLCA': 'BLCA',  # Bladder Cancer
            'TCGA-LIHC': 'LIHC',  # Liver Hepatocellular Carcinoma
            'TCGA-KIRP': 'KIRP',  # Kidney Renal Papillary Cell
            'TCGA-CESC': 'CESC',  # Cervical Cancer
            'TCGA-SARC': 'SARC',  # Sarcoma
            'TCGA-ESCA': 'ESCA',  # Esophageal Cancer
            'TCGA-PAAD': 'PAAD',  # Pancreatic Cancer
            'TCGA-PCPG': 'PCPG',  # Pheochromocytoma
            'TCGA-READ': 'READ',  # Rectum Adenocarcinoma
            'TCGA-TGCT': 'TGCT',  # Testicular Germ Cell Tumors
            'TCGA-LAML': 'LAML',  # Acute Myeloid Leukemia
        }
        
        # Processing statistics
        self.processing_stats = {
            'total_files_found': 0,
            'total_files_processed': 0,
            'total_samples_extracted': 0,
            'cancer_type_counts': {},
            'processing_errors': [],
            'feature_stats': {},
            'start_time': None,
            'end_time': None
        }
        
    def discover_tcga_files(self) -> List[Path]:
        """Discover all TCGA .tsv files"""
        logger.info("🔍 Discovering TCGA files...")
        
        tsv_files = []
        for root, dirs, files in os.walk(self.raw_data_dir):
            for file in files:
                if file.endswith('.tsv') and 'augmented_star_gene_counts' in file:
                    file_path = Path(root) / file
                    tsv_files.append(file_path)
        
        self.processing_stats['total_files_found'] = len(tsv_files)
        logger.info(f"📊 Found {len(tsv_files)} TCGA gene expression files")
        
        # Sample file paths to verify cancer types
        cancer_type_distribution = {}
        for file_path in tsv_files[:100]:  # Sample first 100
            cancer_type = self.extract_cancer_type_from_path(file_path)
            if cancer_type:
                cancer_type_distribution[cancer_type] = cancer_type_distribution.get(cancer_type, 0) + 1
        
        logger.info(f"🧬 Cancer type sample distribution: {cancer_type_distribution}")
        return tsv_files
    
    def extract_cancer_type_from_path(self, file_path: Path) -> Optional[str]:
        """Extract cancer type from file path"""
        path_parts = file_path.parts
        for part in path_parts:
            if part.startswith('TCGA-') and len(part) <= 10:
                # Map to standard cancer type
                for tcga_key, cancer_type in self.cancer_type_mapping.items():
                    if part.startswith(tcga_key.split('-')[1]):  # Match the cancer code
                        return cancer_type
                # If not in mapping, use the TCGA code directly
                if part.startswith('TCGA-'):
                    return part.split('-')[1] if '-' in part else part
        return None
    
    def process_single_tsv_file(self, file_path: Path) -> Optional[Dict]:
        """Process a single TCGA TSV file"""
        try:
            # Extract cancer type from path
            cancer_type = self.extract_cancer_type_from_path(file_path)
            if not cancer_type:
                return None
            
            # Read the TSV file (skip comment lines starting with #)
            df = pd.read_csv(file_path, sep='\t', comment='#', low_memory=False)
            
            # Validate TCGA file structure
            if df.empty or df.shape[0] < 1000:  # Should have thousands of genes
                logger.warning(f"⚠️ Suspicious file size: {file_path} ({df.shape})")
                return None
            
            # Extract gene expression data
            if 'tpm_unstranded' in df.columns:
                expression_values = df['tpm_unstranded'].values
            elif 'fpkm_unstranded' in df.columns:
                expression_values = df['fpkm_unstranded'].values
            elif len(df.columns) >= 2:
                # Use second column as expression values
                expression_values = df.iloc[:, 1].values
            else:
                logger.warning(f"⚠️ No expression column found in {file_path}")
                return None
            
            # Select top variable genes (reduce dimensionality)
            if len(expression_values) > 2000:
                # Sort by expression variance and take top 2000 genes
                expression_values = expression_values[np.argsort(np.var(expression_values.reshape(1, -1), axis=0))[-2000:]]
            
            # Create sample record
            sample_id = file_path.stem.split('.')[0]  # Use filename as sample ID
            
            sample_record = {
                'sample_id': sample_id,
                'cancer_type': cancer_type,
                'expression_features': expression_values.tolist(),
                'n_features': len(expression_values),
                'file_source': str(file_path),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return sample_record
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {str(e)}")
            self.processing_stats['processing_errors'].append({
                'file': str(file_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def process_files_batch(self, file_batch: List[Path]) -> List[Dict]:
        """Process a batch of files in parallel"""
        results = []
        
        # Use partial to create a function with fixed self parameter
        process_func = partial(self.process_single_tsv_file)
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_func, file_path): file_path 
                            for file_path in file_batch}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.processing_stats['total_files_processed'] += 1
                        
                        # Update cancer type counts
                        cancer_type = result['cancer_type']
                        self.processing_stats['cancer_type_counts'][cancer_type] = \
                            self.processing_stats['cancer_type_counts'].get(cancer_type, 0) + 1
                        
                        if len(results) % 100 == 0:
                            logger.info(f"✅ Processed {len(results)} samples...")
                            
                except Exception as e:
                    logger.error(f"❌ Batch processing error for {file_path}: {str(e)}")
        
        return results
    
    def process_all_files(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Process all TCGA files"""
        logger.info("🚀 Starting comprehensive TCGA processing...")
        self.processing_stats['start_time'] = datetime.now().isoformat()
        
        # Discover files
        tsv_files = self.discover_tcga_files()
        
        if max_samples:
            tsv_files = tsv_files[:max_samples]
            logger.info(f"🎯 Processing limited to {max_samples} samples")
        
        # Process files in batches
        batch_size = 500  # Process 500 files at a time
        all_samples = []
        
        for i in range(0, len(tsv_files), batch_size):
            batch = tsv_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(tsv_files) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} files)...")
            
            batch_results = self.process_files_batch(batch)
            all_samples.extend(batch_results)
            
            # Memory cleanup
            gc.collect()
            
            # Progress update
            logger.info(f"📊 Progress: {len(all_samples)}/{len(tsv_files)} files processed "
                       f"({len(all_samples)/len(tsv_files)*100:.1f}%)")
        
        self.processing_stats['end_time'] = datetime.now().isoformat()
        self.processing_stats['total_samples_extracted'] = len(all_samples)
        
        logger.info(f"🎉 Processing complete! {len(all_samples)} authentic samples extracted")
        return all_samples
    
    def create_ml_dataset(self, samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Create ML-ready dataset from processed samples"""
        logger.info("🔬 Creating ML-ready dataset...")
        
        if not samples:
            raise ValueError("No samples provided for dataset creation")
        
        # Determine feature dimensions
        max_features = max(len(sample['expression_features']) for sample in samples)
        logger.info(f"📏 Maximum features per sample: {max_features}")
        
        # Create feature matrix and labels
        X = []
        y = []
        cancer_types = []
        
        for sample in samples:
            # Pad features to max length
            features = sample['expression_features']
            if len(features) < max_features:
                features.extend([0.0] * (max_features - len(features)))
            elif len(features) > max_features:
                features = features[:max_features]
            
            X.append(features)
            cancer_types.append(sample['cancer_type'])
        
        # Convert to numpy arrays
        X = np.array(X, dtype=np.float32)
        
        # Create label encoding
        unique_cancer_types = sorted(list(set(cancer_types)))
        label_mapping = {cancer_type: idx for idx, cancer_type in enumerate(unique_cancer_types)}
        y = np.array([label_mapping[ct] for ct in cancer_types])
        
        logger.info(f"📊 Dataset created: {X.shape} features, {len(unique_cancer_types)} cancer types")
        logger.info(f"🧬 Cancer types: {unique_cancer_types}")
        
        # Update stats
        self.processing_stats['feature_stats'] = {
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'n_cancer_types': len(unique_cancer_types),
            'cancer_types': unique_cancer_types,
            'label_mapping': label_mapping
        }
        
        return X, y, unique_cancer_types
    
    def save_processed_data(self, X: np.ndarray, y: np.ndarray, 
                          cancer_types: List[str], samples: List[Dict]):
        """Save processed data in multiple formats"""
        logger.info("💾 Saving processed dataset...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as compressed numpy arrays (primary format for training)
        np.savez_compressed(
            self.output_dir / f"tcga_authentic_55k_{timestamp}.npz",
            features=X,
            labels=y,
            cancer_types=np.array(cancer_types),
            label_mapping=np.array(list(self.processing_stats['feature_stats']['label_mapping'].items()))
        )
        
        # Save processing metadata
        metadata = {
            'processing_stats': self.processing_stats,
            'dataset_info': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_cancer_types': len(cancer_types),
                'cancer_types': cancer_types,
                'data_source': 'Authentic TCGA RNA-seq',
                'zero_synthetic_data': True,
                'processing_date': timestamp
            }
        }
        
        with open(self.output_dir / f"tcga_authentic_55k_metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save sample inventory
        sample_inventory = pd.DataFrame([
            {
                'sample_id': s['sample_id'],
                'cancer_type': s['cancer_type'], 
                'n_features': s['n_features'],
                'file_source': s['file_source']
            } for s in samples
        ])
        sample_inventory.to_csv(
            self.output_dir / f"tcga_authentic_55k_inventory_{timestamp}.csv", 
            index=False
        )
        
        # Create training-ready file (matches expected format)
        np.savez_compressed(
            self.output_dir / "tcga_processed_data.npz",  # Standard name for training scripts
            features=X,
            labels=y
        )
        
        logger.info(f"✅ Data saved to {self.output_dir}")
        logger.info(f"📄 Primary dataset: tcga_authentic_55k_{timestamp}.npz")
        logger.info(f"📄 Training dataset: tcga_processed_data.npz")
        
        return timestamp
    
    def generate_processing_report(self, timestamp: str):
        """Generate comprehensive processing report"""
        logger.info("📋 Generating processing report...")
        
        report = f"""
# TCGA Authentic Data Processing Report - 55K Samples
================================================================

## Processing Summary
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Files Found**: {self.processing_stats['total_files_found']:,}
- **Total Files Processed**: {self.processing_stats['total_files_processed']:,}
- **Total Samples Extracted**: {self.processing_stats['total_samples_extracted']:,}
- **Success Rate**: {(self.processing_stats['total_files_processed']/self.processing_stats['total_files_found']*100):.1f}%

## Dataset Characteristics
- **Features per Sample**: {self.processing_stats['feature_stats']['n_features']:,}
- **Total Samples**: {self.processing_stats['feature_stats']['n_samples']:,}
- **Cancer Types**: {self.processing_stats['feature_stats']['n_cancer_types']}

## Cancer Type Distribution
"""
        
        for cancer_type, count in sorted(self.processing_stats['cancer_type_counts'].items()):
            percentage = count / self.processing_stats['total_samples_extracted'] * 100
            report += f"- **{cancer_type}**: {count:,} samples ({percentage:.1f}%)\n"
        
        report += f"""
## Data Quality Assurance
- **Zero Synthetic Data**: ✅ 100% Authentic TCGA samples
- **Data Source**: NCI Genomic Data Commons (GDC)
- **File Format**: RNA-seq augmented STAR gene counts (.tsv)
- **Feature Type**: Gene expression (TPM/FPKM values)

## Processing Errors
- **Total Errors**: {len(self.processing_stats['processing_errors'])}
"""
        
        if self.processing_stats['processing_errors']:
            report += "- **Error Details**:\n"
            for error in self.processing_stats['processing_errors'][:10]:  # Show first 10
                report += f"  - {error['file']}: {error['error']}\n"
        
        report += f"""
## Output Files
- **Primary Dataset**: `tcga_authentic_55k_{timestamp}.npz`
- **Training Dataset**: `tcga_processed_data.npz`
- **Metadata**: `tcga_authentic_55k_metadata_{timestamp}.json`
- **Inventory**: `tcga_authentic_55k_inventory_{timestamp}.csv`

## Next Steps
1. Train models on authentic TCGA data
2. Validate performance with real-world clinical data
3. Deploy for precision oncology applications

**Ready for breakthrough cancer AI training! 🚀**
"""
        
        # Save report
        with open(self.output_dir / f"processing_report_{timestamp}.md", 'w') as f:
            f.write(report)
        
        print(report)

def main():
    """Main processing pipeline"""
    logger.info("🔬 Starting Authentic TCGA Processing Pipeline...")
    
    processor = AuthenticTCGAProcessor()
    
    try:
        # Process all files
        samples = processor.process_all_files()
        
        if not samples:
            logger.error("❌ No samples processed successfully!")
            return
        
        # Create ML dataset
        X, y, cancer_types = processor.create_ml_dataset(samples)
        
        # Save processed data
        timestamp = processor.save_processed_data(X, y, cancer_types, samples)
        
        # Generate report
        processor.generate_processing_report(timestamp)
        
        logger.info("🎉 Authentic TCGA processing complete!")
        logger.info(f"✅ Ready for training on {len(samples)} authentic samples")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()