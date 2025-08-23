#!/usr/bin/env python3
"""
🧬 ONCURA - Next-Generation Cancer Genomics Platform
📄 Filename-Based TCGA Sample Processor

Direct sample ID extraction from filenames for reliable multi-omics integration.
Avoids GDC API dependency issues by using filename patterns.
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import logging
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import concurrent.futures
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TCGAFilenameProcessor')

class TCGAFilenameProcessor:
    """Direct TCGA sample extraction from filenames"""
    
    def __init__(self, output_dir: str = "data/filename_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Sample patterns for different data types
        self.sample_patterns = {
            'protein': re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]?)'),
            'mutations': re.compile(r'\.maf'),  # MAF files - need content-based extraction
            'expression': re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]?)'),
            'copy_number': re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]?)'),
            'methylation': re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]?)'),
        }
        
        # Data directories
        self.data_dirs = [
            'data/production_tcga',
            'data/tcga_ultra_massive', 
            'data/tcga_real_fixed'
        ]
        
        self.sample_mappings = defaultdict(lambda: defaultdict(list))
        
    def extract_tcga_id_from_filename(self, filepath: str) -> str:
        """Extract TCGA sample ID from filename"""
        filename = os.path.basename(filepath)
        
        # Direct TCGA pattern matching
        tcga_match = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]?', filename)
        if tcga_match:
            return tcga_match.group(1) if tcga_match.lastindex else tcga_match.group(0)
        
        return None
    
    def extract_from_maf_content(self, maf_file: str, max_samples: int = 20) -> List[str]:
        """Extract TCGA IDs from MAF file content"""
        samples = set()
        
        try:
            # Read MAF file (handle .gz compression)
            if maf_file.endswith('.gz'):
                import gzip
                with gzip.open(maf_file, 'rt') as f:
                    for i, line in enumerate(f):
                        if i == 0:  # Header
                            continue
                        if i > 10000:  # Limit processing
                            break
                            
                        parts = line.strip().split('\t')
                        if len(parts) > 15:  # MAF format check
                            # Tumor_Sample_Barcode is typically column 15
                            sample_id = parts[15] if len(parts) > 15 else parts[0]
                            if sample_id.startswith('TCGA-'):
                                samples.add(sample_id[:15])  # Standard TCGA format
                                if len(samples) >= max_samples:
                                    break
            else:
                with open(maf_file, 'r') as f:
                    for i, line in enumerate(f):
                        if i == 0:  # Header
                            continue
                        if i > 10000:  # Limit processing
                            break
                            
                        parts = line.strip().split('\t')
                        if len(parts) > 15:
                            sample_id = parts[15] if len(parts) > 15 else parts[0]
                            if sample_id.startswith('TCGA-'):
                                samples.add(sample_id[:15])
                                if len(samples) >= max_samples:
                                    break
                                    
        except Exception as e:
            logger.warning(f"Failed to process MAF {maf_file}: {e}")
            
        return list(samples)
    
    def process_data_directory(self, data_dir: str) -> Dict:
        """Process a single data directory"""
        results = defaultdict(lambda: defaultdict(list))
        
        logger.info(f"📂 Processing directory: {data_dir}")
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                filepath = os.path.join(root, file)
                
                # Determine data type from path
                data_type = None
                if 'protein' in root.lower():
                    data_type = 'protein'
                elif 'mutation' in root.lower() or file.endswith('.maf') or file.endswith('.maf.gz'):
                    data_type = 'mutations'
                elif 'expression' in root.lower() or 'rna' in root.lower():
                    data_type = 'expression'
                elif 'copy' in root.lower() or 'cnv' in root.lower():
                    data_type = 'copy_number'
                elif 'methylation' in root.lower():
                    data_type = 'methylation'
                else:
                    continue
                
                # Extract cancer type from path
                cancer_type = 'UNKNOWN'
                path_parts = root.split(os.sep)
                for part in path_parts:
                    if part.startswith('TCGA-'):
                        cancer_type = part
                        break
                
                # Extract sample IDs
                sample_ids = []
                
                if data_type == 'mutations':
                    # MAF files need content-based extraction
                    sample_ids = self.extract_from_maf_content(filepath)
                else:
                    # Other types use filename extraction
                    sample_id = self.extract_tcga_id_from_filename(filepath)
                    if sample_id:
                        sample_ids = [sample_id]
                
                # Store mappings
                for sample_id in sample_ids:
                    if sample_id:
                        results[cancer_type][sample_id].append({
                            'data_type': data_type,
                            'file_path': filepath,
                            'file_size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
                        })
        
        return results
    
    def integrate_multi_omics_data(self, max_samples_per_type: int = 1000) -> pd.DataFrame:
        """Integrate all available multi-omics data"""
        logger.info("🔗 Starting multi-omics integration")
        
        all_results = defaultdict(lambda: defaultdict(list))
        
        # Process all data directories
        for data_dir in self.data_dirs:
            if os.path.exists(data_dir):
                results = self.process_data_directory(data_dir)
                
                # Merge results
                for cancer_type, samples in results.items():
                    for sample_id, files in samples.items():
                        all_results[cancer_type][sample_id].extend(files)
        
        # Create integration matrix
        integration_data = []
        
        for cancer_type, samples in all_results.items():
            logger.info(f"📊 {cancer_type}: {len(samples)} samples")
            
            sample_count = 0
            for sample_id, files in samples.items():
                if sample_count >= max_samples_per_type:
                    break
                    
                # Organize by data type
                data_types = defaultdict(list)
                for file_info in files:
                    data_types[file_info['data_type']].append(file_info)
                
                # Create record
                record = {
                    'sample_id': sample_id,
                    'cancer_type': cancer_type,
                    'num_omics_types': len(data_types),
                    'total_files': len(files),
                    'has_mutations': 'mutations' in data_types,
                    'has_expression': 'expression' in data_types,
                    'has_protein': 'protein' in data_types,
                    'has_copy_number': 'copy_number' in data_types,
                    'has_methylation': 'methylation' in data_types,
                }
                
                # Add file counts
                for dtype in ['mutations', 'expression', 'protein', 'copy_number', 'methylation']:
                    record[f'{dtype}_files'] = len(data_types.get(dtype, []))
                    
                    # Add first file path for each type
                    files_of_type = data_types.get(dtype, [])
                    if files_of_type:
                        record[f'{dtype}_file'] = files_of_type[0]['file_path']
                    else:
                        record[f'{dtype}_file'] = None
                
                integration_data.append(record)
                sample_count += 1
        
        # Create DataFrame
        df = pd.DataFrame(integration_data)
        
        logger.info(f"✅ Integration complete: {len(df)} samples across {df['cancer_type'].nunique()} cancer types")
        
        # Summary statistics
        logger.info("📈 Multi-omics coverage:")
        for omics_type in ['mutations', 'expression', 'protein', 'copy_number', 'methylation']:
            count = df[f'has_{omics_type}'].sum()
            percentage = (count / len(df)) * 100 if len(df) > 0 else 0
            logger.info(f"   {omics_type}: {count} samples ({percentage:.1f}%)")
        
        # Save results
        output_file = self.output_dir / f"tcga_multi_omics_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Saved integration data to: {output_file}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description="TCGA Filename-Based Sample Processor")
    parser.add_argument('--max-samples', type=int, default=2000,
                       help='Maximum samples per cancer type')
    parser.add_argument('--output-dir', type=str, default='data/filename_processed',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🧬 ONCURA - Next-Generation Cancer Genomics Platform")
    print("📄 Filename-Based TCGA Sample Processor")
    print("=" * 60)
    
    processor = TCGAFilenameProcessor(output_dir=args.output_dir)
    
    # Run integration
    df = processor.integrate_multi_omics_data(max_samples_per_type=args.max_samples)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Total samples: {len(df)}")
    print(f"🎯 Cancer types: {df['cancer_type'].nunique()}")
    print(f"💾 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
