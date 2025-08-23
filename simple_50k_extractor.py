#!/usr/bin/env python3
"""
Simple 50k Sample Extractor
Extracts maximum possible samples from existing raw data using a simple, reliable approach

Author: Oncura AI
Date: 2025-08-22
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Simple50kExtractor:
    """Simple, reliable approach to extract maximum samples from raw data"""
    
    def __init__(self, base_path: str = "/Users/stillwell/projects/cancer-alpha/data"):
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "tcga_ultra_massive_50k"
        self.output_path = self.base_path / "simple_50k_output"
        self.output_path.mkdir(exist_ok=True)
        
        # Target samples
        self.target_samples = 50000
        
        # Omics data types
        self.omics_types = {
            'expression': 'Gene Expression Quantification',
            'copy_number': 'Copy Number Segment', 
            'methylation': 'Methylation Beta Value',
            'mirna': 'miRNA Expression Quantification',
            'protein': 'Protein Expression Quantification',
            'mutations': 'Masked Somatic Mutation',
            'clinical': 'Clinical Supplement'
        }
    
    def extract_sample_id(self, filename: str) -> Optional[str]:
        """Extract TCGA sample ID from filename"""
        patterns = [
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}[A-Z]-[A-Z0-9]{4}-[0-9]{2})',
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}[A-Z])',
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z])',
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        return None
    
    def process_cancer_type(self, cancer_type: str) -> Dict[str, Dict]:
        """Process a single cancer type to find all samples"""
        cancer_path = self.raw_data_path / cancer_type
        if not cancer_path.exists():
            logger.warning(f"Cancer type directory not found: {cancer_type}")
            return {}
        
        logger.info(f"Processing {cancer_type}...")
        samples = {}
        
        # Process each omics type
        for omics_name, omics_dir in self.omics_types.items():
            omics_path = cancer_path / omics_dir
            if not omics_path.exists():
                continue
            
            # Find all files
            file_patterns = ['*.tsv', '*.txt', '*.seg', '*.xml', '*.maf']
            files_found = []
            
            for pattern in file_patterns:
                files_found.extend(omics_path.glob(pattern))
            
            logger.info(f"  {omics_name}: Found {len(files_found)} files")
            
            # Extract sample IDs from filenames
            for file_path in files_found:
                sample_id = self.extract_sample_id(file_path.name)
                if sample_id:
                    if sample_id not in samples:
                        samples[sample_id] = {
                            'sample_id': sample_id,
                            'cancer_type': cancer_type,
                            'omics_data': {},
                            'file_count': 0
                        }
                    
                    if omics_name not in samples[sample_id]['omics_data']:
                        samples[sample_id]['omics_data'][omics_name] = []
                    
                    samples[sample_id]['omics_data'][omics_name].append(str(file_path))
                    samples[sample_id]['file_count'] += 1
        
        # Filter to keep samples with at least 1 omics type
        valid_samples = {
            sid: data for sid, data in samples.items()
            if len(data['omics_data']) >= 1
        }
        
        logger.info(f"  {cancer_type}: {len(valid_samples)} valid samples")
        return valid_samples
    
    def discover_all_samples(self) -> List[Dict]:
        """Discover all samples across all cancer types"""
        logger.info("Discovering all available cancer types...")
        
        # Find all cancer type directories
        cancer_types = []
        if self.raw_data_path.exists():
            for item in self.raw_data_path.iterdir():
                if item.is_dir() and item.name.startswith('TCGA-'):
                    cancer_types.append(item.name)
        
        logger.info(f"Found {len(cancer_types)} cancer types: {cancer_types}")
        
        # Process each cancer type
        all_samples = []
        for cancer_type in cancer_types:
            cancer_samples = self.process_cancer_type(cancer_type)
            all_samples.extend(cancer_samples.values())
        
        logger.info(f"Total samples discovered: {len(all_samples)}")
        return all_samples
    
    def score_and_select_samples(self, all_samples: List[Dict]) -> List[Dict]:
        """Score samples and select the best ones"""
        logger.info(f"Scoring and selecting from {len(all_samples)} samples...")
        
        # Score each sample
        for sample in all_samples:
            score = 0.0
            omics_data = sample.get('omics_data', {})
            
            # Base score: number of omics types
            score += len(omics_data) * 10
            
            # Bonus for file count
            score += sample.get('file_count', 0) * 0.1
            
            # Bonus for important omics types
            if 'expression' in omics_data:
                score += 5
            if 'clinical' in omics_data:
                score += 3
            if 'mutations' in omics_data:
                score += 3
            if 'copy_number' in omics_data:
                score += 2
            
            sample['score'] = score
        
        # Sort by score (descending)
        all_samples.sort(key=lambda x: x['score'], reverse=True)
        
        # Select samples with cancer type balance
        selected = []
        cancer_counts = defaultdict(int)
        max_per_cancer = max(1, self.target_samples // 50)  # Allow flexibility
        
        for sample in all_samples:
            if len(selected) >= self.target_samples:
                break
            
            cancer_type = sample['cancer_type']
            
            # Balanced selection, but allow overflow if needed
            if (cancer_counts[cancer_type] < max_per_cancer or 
                len(selected) < self.target_samples * 0.8):
                selected.append(sample)
                cancer_counts[cancer_type] += 1
        
        logger.info(f"Selected {len(selected)} samples")
        logger.info(f"Cancer type distribution: {dict(cancer_counts)}")
        return selected
    
    def create_dataset(self, selected_samples: List[Dict]) -> str:
        """Create the final dataset CSV"""
        logger.info(f"Creating dataset from {len(selected_samples)} samples...")
        
        rows = []
        for sample in selected_samples:
            row = {
                'sample_id': sample['sample_id'],
                'cancer_type': sample['cancer_type'],
                'score': sample.get('score', 0),
                'file_count': sample.get('file_count', 0),
                'omics_count': len(sample.get('omics_data', {}))
            }
            
            # Add omics file paths
            for omics_name in self.omics_types.keys():
                omics_data = sample.get('omics_data', {})
                if omics_name in omics_data:
                    row[omics_name] = ';'.join(omics_data[omics_name])
                else:
                    row[omics_name] = ''
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_path / f"tcga_simple_50k_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        # Generate statistics
        stats = {
            'timestamp': timestamp,
            'total_samples': len(df),
            'target_samples': self.target_samples,
            'achievement_rate': len(df) / self.target_samples * 100,
            'cancer_types': df['cancer_type'].value_counts().to_dict(),
            'omics_coverage': {},
            'avg_score': float(df['score'].mean()),
            'avg_file_count': float(df['file_count'].mean()),
            'avg_omics_count': float(df['omics_count'].mean())
        }
        
        # Omics coverage
        for omics_name in self.omics_types.keys():
            non_empty = (df[omics_name] != '').sum()
            stats['omics_coverage'][omics_name] = {
                'samples_with_data': int(non_empty),
                'coverage_percent': float((non_empty / len(df)) * 100)
            }
        
        # Save statistics
        stats_file = self.output_path / f"simple_50k_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset created: {output_file}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Achievement: {stats['achievement_rate']:.1f}%")
        logger.info(f"Stats saved: {stats_file}")
        
        return str(output_file)
    
    def run_extraction(self) -> str:
        """Run the complete extraction process"""
        logger.info("🚀 Starting Simple 50k Extraction...")
        
        try:
            # Step 1: Discover all samples
            all_samples = self.discover_all_samples()
            
            if not all_samples:
                logger.error("No samples found!")
                return ""
            
            # Step 2: Score and select samples
            selected_samples = self.score_and_select_samples(all_samples)
            
            # Step 3: Create final dataset
            final_dataset = self.create_dataset(selected_samples)
            
            logger.info("🎉 Extraction complete!")
            return final_dataset
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise


def main():
    print("="*60)
    print("🚀 SIMPLE 50K TCGA EXTRACTOR")
    print("="*60)
    print("Reliable extraction of maximum samples from raw data")
    print("100% REAL TCGA DATA")
    print("="*60)
    
    extractor = Simple50kExtractor()
    result = extractor.run_extraction()
    
    if result:
        print("\\n" + "="*50)
        print("✅ EXTRACTION SUCCESSFUL!")
        print("="*50)
        print(f"Dataset: {result}")
        print("Check the stats file for detailed analysis")
        print("="*50)
    else:
        print("\\n" + "="*50)
        print("❌ EXTRACTION FAILED")
        print("="*50)


if __name__ == "__main__":
    main()
