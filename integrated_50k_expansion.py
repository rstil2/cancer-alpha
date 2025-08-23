#!/usr/bin/env python3
"""
Integrated 50k Sample Expansion
Uses existing TCGA downloader infrastructure to efficiently reach 50k samples

This script:
1. Analyzes current data to identify gaps
2. Uses existing ultra_massive_tcga_downloader to get additional samples
3. Processes all available raw data for comprehensive integration
4. Creates the final 50k dataset

Author: Oncura AI
Date: 2025-08-22
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sqlite3
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
import subprocess
import hashlib
import re

# Add current directory to path to import existing modules
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_50k_expansion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Integrated50kExpansion:
    """Integrated approach using existing TCGA infrastructure for 50k expansion"""
    
    def __init__(self, base_path: str = "/Users/stillwell/projects/cancer-alpha/data"):
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "tcga_ultra_massive_50k"
        self.output_path = self.base_path / "tcga_50k_final"
        self.output_path.mkdir(exist_ok=True)
        
        # Current dataset info
        self.current_csv = self.base_path / "processed_50k" / "oncura_comprehensive_multi_omics_50k.csv"
        self.current_samples = 9660
        self.target_samples = 50000
        
        # Cancer types and their current status
        self.current_cancer_types = [
            'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-GBM', 'TCGA-KIRC',
            'TCGA-KIRP', 'TCGA-OV', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-UCEC'
        ]
        
        # Additional cancer types to download (based on existing downloader priorities)
        self.additional_cancer_types = [
            'TCGA-LUAD',  # Lung Adenocarcinoma - high yield
            'TCGA-LUSC',  # Lung Squamous Cell Carcinoma
            'TCGA-COAD',  # Colon Adenocarcinoma
            'TCGA-PRAD',  # Prostate Adenocarcinoma
            'TCGA-THCA',  # Thyroid Carcinoma
            'TCGA-HNSC',  # Head and Neck Squamous Cell Carcinoma
            'TCGA-LGG',   # Brain Lower Grade Glioma
            'TCGA-LIHC',  # Liver Hepatocellular Carcinoma
            'TCGA-STAD',  # Stomach Adenocarcinoma
            'TCGA-PAAD',  # Pancreatic Adenocarcinoma
            'TCGA-READ',  # Rectum Adenocarcinoma
            'TCGA-LAML',  # Acute Myeloid Leukemia
            'TCGA-PCPG',  # Pheochromocytoma and Paraganglioma
            'TCGA-TGCT',  # Testicular Germ Cell Tumors
            'TCGA-ESCA',  # Esophageal Carcinoma
            'TCGA-THYM',  # Thymoma
            'TCGA-MESO',  # Mesothelioma
            'TCGA-UCS',   # Uterine Carcinosarcoma
            'TCGA-ACC',   # Adrenocortical Carcinoma
            'TCGA-UVM',   # Uveal Melanoma
            'TCGA-DLBC',  # Lymphoma
            'TCGA-KICH',  # Kidney Chromophobe
            'TCGA-CHOL'   # Cholangiocarcinoma
        ]
        
        # Omics data type mapping (matching your existing structure)
        self.omics_types = {
            'expression': 'Gene Expression Quantification',
            'copy_number': 'Copy Number Segment', 
            'methylation': 'Methylation Beta Value',
            'mirna': 'miRNA Expression Quantification',
            'protein': 'Protein Expression Quantification',
            'mutations': 'Masked Somatic Mutation',
            'clinical': 'Clinical Supplement'
        }
        
        # Track processed samples
        self.processed_samples = {}
        self.sample_stats = {}
        
    def analyze_current_dataset(self) -> Dict:
        """Analyze current dataset to understand coverage and identify gaps"""
        logger.info("Analyzing current dataset...")
        
        # Read current CSV
        try:
            df = pd.read_csv(self.current_csv)
            logger.info(f"Current CSV has {len(df)} samples")
            
            # Analyze cancer type distribution
            cancer_counts = df['cancer_type'].value_counts().to_dict()
            
            # Analyze omics coverage
            omics_coverage = {}
            for omics_type in self.omics_types.keys():
                if omics_type in df.columns:
                    non_empty = (df[omics_type] != '').sum()
                    omics_coverage[omics_type] = {
                        'samples_with_data': non_empty,
                        'coverage_percent': (non_empty / len(df)) * 100
                    }
            
            analysis = {
                'total_samples': len(df),
                'cancer_type_distribution': cancer_counts,
                'omics_coverage': omics_coverage,
                'samples_needed': self.target_samples - len(df)
            }
            
            logger.info(f"Analysis complete. Need {analysis['samples_needed']} additional samples")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing current dataset: {e}")
            return {}
    
    def analyze_available_raw_data(self) -> Dict:
        """Analyze all available raw data to estimate potential sample yield"""
        logger.info("Analyzing available raw data...")
        
        raw_stats = {}
        total_files = 0
        potential_samples = 0
        
        # Check current cancer types
        for cancer_type in self.current_cancer_types:
            cancer_path = self.raw_data_path / cancer_type
            if cancer_path.exists():
                cancer_files = 0
                for omics_name, omics_dir in self.omics_types.items():
                    omics_path = cancer_path / omics_dir
                    if omics_path.exists():
                        files = list(omics_path.glob("*.tsv")) + list(omics_path.glob("*.txt")) + list(omics_path.glob("*.seg"))
                        cancer_files += len(files)
                
                raw_stats[cancer_type] = cancer_files
                total_files += cancer_files
        
        # Estimate potential samples (conservative estimate: 4 files per sample across omics)
        potential_samples = total_files // 4
        
        logger.info(f"Raw data analysis: {total_files} files, ~{potential_samples} potential samples")
        return {
            'cancer_type_files': raw_stats,
            'total_files': total_files,
            'potential_samples': potential_samples
        }
    
    def run_additional_downloads(self, cancer_types: List[str]) -> bool:
        """Use existing ultra_massive downloader to get additional cancer types"""
        logger.info(f"Running additional downloads for {len(cancer_types)} cancer types...")
        
        # Use the existing ultra_massive_tcga_downloader
        downloader_script = Path(__file__).parent / "ultra_massive_tcga_downloader_50k.py"
        
        if not downloader_script.exists():
            logger.error(f"Downloader script not found: {downloader_script}")
            return False
        
        # Create custom download configuration
        download_config = {
            "target_cancer_types": cancer_types,
            "samples_per_cancer": 2000,  # Aggressive target
            "output_dir": str(self.raw_data_path),
            "max_concurrent_downloads": 8,
            "priority_data_types": [
                "Gene Expression Quantification",
                "Masked Somatic Mutation", 
                "Copy Number Segment",
                "Methylation Beta Value",
                "Clinical Supplement",
                "miRNA Expression Quantification",
                "Protein Expression Quantification"
            ]
        }
        
        config_file = self.output_path / "download_config.json"
        with open(config_file, 'w') as f:
            json.dump(download_config, f, indent=2)
        
        logger.info(f"Download configuration saved to {config_file}")
        
        # Create modified downloader script for these specific cancer types
        custom_downloader = self.create_custom_downloader(cancer_types)
        
        logger.info(f"Custom downloader created: {custom_downloader}")
        logger.info("Please run the custom downloader to fetch additional cancer types")
        
        return True
    
    def create_custom_downloader(self, cancer_types: List[str]) -> Path:
        """Create a custom downloader script for specific cancer types"""
        custom_script = self.output_path / "custom_downloader_50k.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Custom TCGA Downloader for 50k Expansion
Generated: {datetime.now().isoformat()}

Downloads specific cancer types needed to reach 50k samples
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from ultra_massive_tcga_downloader_50k import UltraMassiveTCGADownloader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Download specific cancer types for 50k expansion"""
    
    # Target cancer types for expansion
    target_cancer_types = {cancer_types}
    
    logger.info(f"🚀 Custom 50k Expansion Download")
    logger.info(f"Target cancer types: {{len(target_cancer_types)}}")
    logger.info(f"Cancer types: {{target_cancer_types}}")
    
    # Initialize downloader
    output_dir = "{self.raw_data_path}"
    downloader = UltraMassiveTCGADownloader(output_dir)
    
    # Download each cancer type
    total_downloaded = 0
    
    for cancer_type in target_cancer_types:
        logger.info(f"\\n{'='*60}")
        logger.info(f"Downloading {{cancer_type}}")
        logger.info(f"{'='*60}")
        
        try:
            downloaded = downloader.download_cancer_type(cancer_type, target_samples_per_type=2000)
            total_downloaded += downloaded
            logger.info(f"✅ {{cancer_type}}: {{downloaded}} files downloaded")
            
        except Exception as e:
            logger.error(f"❌ Error downloading {{cancer_type}}: {{e}}")
            continue
    
    logger.info(f"\\n🎉 Custom download complete!")
    logger.info(f"Total files downloaded: {{total_downloaded:,}}")
    logger.info(f"Ready for 50k sample integration!")

if __name__ == "__main__":
    main()
'''
        
        with open(custom_script, 'w') as f:
            f.write(script_content)
        
        custom_script.chmod(0o755)
        return custom_script
    
    def extract_sample_id_from_filename(self, file_path: Path) -> Optional[str]:
        """Extract TCGA sample ID from filename using robust pattern matching"""
        filename = file_path.name
        
        # TCGA sample ID patterns
        patterns = [
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}[A-Z]-[A-Z0-9]{4}-[0-9]{2})',  # Full barcode
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z])',  # Sample barcode
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})',  # Patient barcode
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return None
    
    def process_all_raw_data_parallel(self, max_workers: int = 8) -> List[Dict]:
        """Process all available raw data in parallel to create comprehensive sample list"""
        logger.info(f"Processing all raw data with {max_workers} workers...")
        
        # Get all cancer types (current + additional if available)
        all_cancer_types = []
        for cancer_type in (self.current_cancer_types + self.additional_cancer_types):
            cancer_path = self.raw_data_path / cancer_type
            if cancer_path.exists():
                all_cancer_types.append(cancer_type)
        
        logger.info(f"Processing {len(all_cancer_types)} cancer types")
        
        # Process in parallel
        all_samples = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cancer = {
                executor.submit(self.process_cancer_type, cancer_type): cancer_type
                for cancer_type in all_cancer_types
            }
            
            for future in as_completed(future_to_cancer):
                cancer_type = future_to_cancer[future]
                try:
                    cancer_samples = future.result()
                    all_samples.extend(cancer_samples)
                    logger.info(f"Processed {cancer_type}: {len(cancer_samples)} samples")
                except Exception as e:
                    logger.error(f"Error processing {cancer_type}: {e}")
        
        logger.info(f"Total samples processed: {len(all_samples)}")
        return all_samples
    
    def process_cancer_type(self, cancer_type: str) -> List[Dict]:
        """Process all samples for a single cancer type"""
        cancer_path = self.raw_data_path / cancer_type
        if not cancer_path.exists():
            return []
        
        # Find all unique sample IDs by scanning expression files (most common)
        sample_files = {}
        expr_path = cancer_path / self.omics_types['expression']
        
        if expr_path.exists():
            for expr_file in expr_path.glob("*.tsv"):
                sample_id = self.extract_sample_id_from_filename(expr_file)
                if sample_id:
                    if sample_id not in sample_files:
                        sample_files[sample_id] = {
                            'sample_id': sample_id,
                            'cancer_type': cancer_type,
                            'omics_data': {}
                        }
        
        # Find files for each sample across all omics types
        samples = []
        for sample_id, sample_data in sample_files.items():
            omics_found = 0
            
            for omics_name, omics_dir in self.omics_types.items():
                omics_path = cancer_path / omics_dir
                if not omics_path.exists():
                    continue
                
                # Find files for this sample
                sample_files_list = []
                for ext in ['*.tsv', '*.txt', '*.seg']:
                    pattern = f"*{sample_id}*{ext[1:]}"
                    matching_files = list(omics_path.glob(pattern))
                    sample_files_list.extend(matching_files)
                
                if sample_files_list:
                    sample_data['omics_data'][omics_name] = [str(f) for f in sample_files_list]
                    omics_found += 1
            
            # Only include samples with at least 2 omics types
            if omics_found >= 2:
                samples.append(sample_data)
        
        return samples
    
    def create_comprehensive_50k_dataset(self, processed_samples: List[Dict]) -> str:
        """Create the final comprehensive 50k dataset"""
        logger.info(f"Creating comprehensive 50k dataset from {len(processed_samples)} processed samples...")
        
        # Convert to DataFrame format
        rows = []
        
        for sample in processed_samples:
            row = {
                'sample_id': sample['sample_id'],
                'cancer_type': sample['cancer_type']
            }
            
            # Add omics file paths (semicolon-separated)
            for omics_name in self.omics_types.keys():
                if omics_name in sample.get('omics_data', {}):
                    file_paths = sample['omics_data'][omics_name]
                    row[omics_name] = ';'.join(file_paths)
                else:
                    row[omics_name] = ''
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Remove duplicates based on sample_id
        df = df.drop_duplicates(subset=['sample_id'], keep='first')
        
        # Limit to target if we have more than needed
        if len(df) > self.target_samples:
            # Sample stratified by cancer type to maintain balance
            df_sampled = df.groupby('cancer_type').apply(
                lambda x: x.sample(n=min(len(x), self.target_samples // df['cancer_type'].nunique()))
            ).reset_index(drop=True)
            df = df_sampled.iloc[:self.target_samples]
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_path / f"tcga_comprehensive_50k_final_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Generate comprehensive metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': len(df),
            'cancer_types': df['cancer_type'].value_counts().to_dict(),
            'omics_coverage': {},
            'data_sources': {
                'raw_data_path': str(self.raw_data_path),
                'processing_method': 'comprehensive_multi_omics_integration'
            }
        }
        
        # Calculate omics coverage
        for omics_name in self.omics_types.keys():
            if omics_name in df.columns:
                non_empty = (df[omics_name] != '').sum()
                metadata['omics_coverage'][omics_name] = {
                    'samples_with_data': int(non_empty),
                    'coverage_percent': float((non_empty / len(df)) * 100)
                }
        
        # Save metadata
        metadata_file = self.output_path / f"tcga_50k_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"50k dataset created: {output_file}")
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Cancer types: {len(df['cancer_type'].unique())}")
        logger.info(f"Metadata saved: {metadata_file}")
        
        return str(output_file)
    
    def run_integrated_expansion(self) -> str:
        """Run the complete integrated 50k expansion"""
        logger.info("🚀 Starting Integrated 50k Sample Expansion...")
        
        try:
            # Step 1: Analyze current dataset
            current_analysis = self.analyze_current_dataset()
            
            # Step 2: Analyze available raw data
            raw_analysis = self.analyze_available_raw_data()
            
            # Step 3: Determine if we need additional downloads
            available_samples = raw_analysis.get('potential_samples', 0)
            needed_samples = self.target_samples - self.current_samples
            
            if available_samples < needed_samples:
                logger.info(f"Need additional downloads: {needed_samples - available_samples} more samples")
                success = self.run_additional_downloads(self.additional_cancer_types[:15])  # Top 15 additional types
                
                if success:
                    logger.info("Download script created. Please run it first, then re-run this expansion.")
                    return "download_required"
            
            # Step 4: Process all available raw data
            logger.info("Processing all available raw data...")
            processed_samples = self.process_all_raw_data_parallel()
            
            if len(processed_samples) < 40000:
                logger.warning(f"Only {len(processed_samples)} samples available, may need additional data")
            
            # Step 5: Create final comprehensive dataset
            final_dataset = self.create_comprehensive_50k_dataset(processed_samples)
            
            # Step 6: Validate final dataset
            validation_results = self.validate_final_dataset(final_dataset)
            
            logger.info("🎉 Integrated 50k expansion complete!")
            logger.info(f"Final dataset: {final_dataset}")
            
            return final_dataset
            
        except Exception as e:
            logger.error(f"Integrated expansion failed: {e}")
            raise
    
    def validate_final_dataset(self, dataset_path: str) -> Dict:
        """Validate the final 50k dataset"""
        logger.info("Validating final 50k dataset...")
        
        df = pd.read_csv(dataset_path)
        
        validation_results = {
            'total_samples': len(df),
            'unique_samples': df['sample_id'].nunique(),
            'cancer_types': df['cancer_type'].nunique(),
            'cancer_distribution': df['cancer_type'].value_counts().to_dict(),
            'omics_coverage': {},
            'quality_metrics': {}
        }
        
        # Check omics coverage
        for omics_name in self.omics_types.keys():
            if omics_name in df.columns:
                non_empty = (df[omics_name] != '').sum()
                validation_results['omics_coverage'][omics_name] = {
                    'samples_with_data': int(non_empty),
                    'coverage_percent': float((non_empty / len(df)) * 100)
                }
        
        # Quality metrics
        validation_results['quality_metrics'] = {
            'target_achieved': len(df) >= 45000,  # 90% of target
            'balanced_cancer_types': df['cancer_type'].nunique() >= 15,
            'good_omics_coverage': sum(1 for cov in validation_results['omics_coverage'].values() 
                                     if cov['coverage_percent'] > 50) >= 4
        }
        
        # Save validation results
        validation_file = Path(dataset_path).parent / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation complete: {validation_file}")
        return validation_results


def main():
    """Main execution function"""
    print("="*70)
    print("🚀 INTEGRATED 50K TCGA SAMPLE EXPANSION")
    print("="*70)
    print("Using existing infrastructure to efficiently reach 50k samples")
    print("100% REAL TCGA DATA - NO SYNTHETIC CONTAMINATION")
    print("="*70)
    
    expansion = Integrated50kExpansion()
    result = expansion.run_integrated_expansion()
    
    if result == "download_required":
        print("\\n" + "="*60)
        print("🔄 ADDITIONAL DOWNLOADS NEEDED")
        print("="*60)
        print("Custom downloader created. Please run:")
        print("python /Users/stillwell/projects/cancer-alpha/data/tcga_50k_final/custom_downloader_50k.py")
        print("Then re-run this script to complete the 50k expansion.")
        print("="*60)
    else:
        print("\\n" + "="*60)
        print("🎉 50K EXPANSION COMPLETE!")
        print("="*60)
        print(f"Final dataset: {result}")
        print("Your 50k multi-omics dataset is ready for analysis!")
        print("="*60)


if __name__ == "__main__":
    main()
