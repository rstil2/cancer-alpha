#!/usr/bin/env python3
"""
Aggressive 50k Processor
Extracts maximum possible samples from existing raw data before downloading additional data

This version:
1. Processes ALL available raw files more aggressively
2. Uses looser matching criteria to find more samples
3. Includes partial omics samples (1+ omics types instead of 2+)
4. Attempts to get closer to 50k from existing data

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
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aggressive_50k_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Aggressive50kProcessor:
    """Aggressively process existing raw data to extract maximum samples"""
    
    def __init__(self, base_path: str = "/Users/stillwell/projects/cancer-alpha/data"):
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "tcga_ultra_massive_50k"
        self.output_path = self.base_path / "aggressive_50k_output"
        self.output_path.mkdir(exist_ok=True)
        
        # Target samples
        self.target_samples = 50000
        
        # All available cancer types
        self.all_cancer_types = [
            'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-GBM', 'TCGA-KIRC',
            'TCGA-KIRP', 'TCGA-OV', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-UCEC'
        ]
        
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
        
        # Track all discovered samples
        self.all_samples = {}
        self.sample_statistics = defaultdict(lambda: defaultdict(int))
        
    def discover_all_cancer_types(self) -> List[str]:
        """Discover all available cancer types in raw data"""
        available_types = []
        
        if self.raw_data_path.exists():
            for item in self.raw_data_path.iterdir():
                if item.is_dir() and item.name.startswith('TCGA-'):
                    available_types.append(item.name)
        
        logger.info(f"Discovered {len(available_types)} cancer types: {available_types}")
        return available_types
    
    def extract_all_sample_patterns(self, file_path: Path) -> List[str]:
        """Extract all possible TCGA sample ID patterns from filename"""
        filename = file_path.name
        sample_ids = []
        
        # Multiple TCGA patterns with different levels of specificity
        patterns = [
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}[A-Z]-[A-Z0-9]{4}-[0-9]{2})',  # Full UUID
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}[A-Z])',  # Sample barcode  
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z])',  # Sample ID
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})',  # Patient barcode
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            sample_ids.extend(matches)
        
        # Remove duplicates while preserving order
        unique_samples = []
        seen = set()
        for sample_id in sample_ids:
            if sample_id not in seen:
                unique_samples.append(sample_id)
                seen.add(sample_id)
        
        return unique_samples
    
    def comprehensive_sample_discovery(self, cancer_type: str) -> Dict[str, Dict]:
        """Comprehensively discover all samples for a cancer type"""
        cancer_path = self.raw_data_path / cancer_type
        if not cancer_path.exists():
            return {}
        
        logger.info(f"Comprehensive discovery for {cancer_type}...")
        
        discovered_samples = {}
        
        # Scan all omics directories
        for omics_name, omics_dir in self.omics_types.items():
            omics_path = cancer_path / omics_dir
            if not omics_path.exists():
                continue
            
            file_count = 0
            # Look for all possible file types
            for file_pattern in ['*.tsv', '*.txt', '*.seg', '*.xml', '*.maf']:
                for file_path in omics_path.glob(file_pattern):
                    file_count += 1
                    
                    # Extract all possible sample IDs from this file
                    sample_ids = self.extract_all_sample_patterns(file_path)
                    
                    for sample_id in sample_ids:
                        if sample_id not in discovered_samples:
                            discovered_samples[sample_id] = {
                                'sample_id': sample_id,
                                'cancer_type': cancer_type,
                                'omics_data': {},
                                'file_count': 0
                            }
                        
                        # Add this file to the sample's omics data
                        if omics_name not in discovered_samples[sample_id]['omics_data']:
                            discovered_samples[sample_id]['omics_data'][omics_name] = []
                        
                        discovered_samples[sample_id]['omics_data'][omics_name].append(str(file_path))
                        discovered_samples[sample_id]['file_count'] += 1
            
            self.sample_statistics[cancer_type][f'{omics_name}_files'] = file_count
        
        # Filter samples - keep those with at least 1 omics type (very aggressive)
        valid_samples = {
            sample_id: sample_data
            for sample_id, sample_data in discovered_samples.items()
            if len(sample_data['omics_data']) >= 1
        }
        
        logger.info(f"{cancer_type}: {len(valid_samples)} valid samples from {len(discovered_samples)} discovered")
        return valid_samples
    
    def parallel_comprehensive_discovery(self, max_workers: int = 8) -> Dict[str, Dict]:
        """Run comprehensive discovery in parallel across all cancer types"""
        logger.info(f"Running parallel comprehensive discovery with {max_workers} workers...")
        
        # Discover available cancer types
        available_cancer_types = self.discover_all_cancer_types()
        
        all_discovered_samples = {}
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_cancer = {
                executor.submit(self.comprehensive_sample_discovery, cancer_type): cancer_type
                for cancer_type in available_cancer_types
            }
            
            for future in as_completed(future_to_cancer):
                cancer_type = future_to_cancer[future]
                try:
                    cancer_samples = future.result()
                    all_discovered_samples.update(cancer_samples)
                    logger.info(f"✅ {cancer_type}: {len(cancer_samples)} samples discovered")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {cancer_type}: {e}")
        
        logger.info(f"🎯 Total samples discovered: {len(all_discovered_samples)}")
        return all_discovered_samples
    
    def prioritize_and_select_samples(self, all_samples: Dict[str, Dict], target_count: int) -> List[Dict]:
        """Prioritize and select the best samples for the 50k dataset"""
        logger.info(f"Prioritizing and selecting top {target_count} samples from {len(all_samples)} discovered...")
        
        # Convert to list for sorting
        sample_list = list(all_samples.values())
        
        # Score samples based on:
        # 1. Number of omics types (higher is better)
        # 2. Total file count (higher is better)  
        # 3. Has expression data (preferred)
        # 4. Has clinical data (preferred)
        
        def score_sample(sample: Dict) -> float:
            score = 0.0
            omics_data = sample.get('omics_data', {})
            
            # Base score: number of omics types
            score += len(omics_data) * 10
            
            # Bonus for total file count
            score += sample.get('file_count', 0) * 0.1
            
            # Bonus for having key omics types
            if 'expression' in omics_data:
                score += 5
            if 'clinical' in omics_data:
                score += 3
            if 'mutations' in omics_data:
                score += 3
            if 'copy_number' in omics_data:
                score += 2
            if 'methylation' in omics_data:
                score += 2
            
            return score
        
        # Score and sort samples
        for sample in sample_list:
            sample['priority_score'] = score_sample(sample)
        
        # Sort by score (descending)
        sample_list.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Select top samples, ensuring cancer type balance
        selected_samples = []
        cancer_type_counts = defaultdict(int)
        max_per_cancer = max(1, target_count // 20)  # Allow up to target/20 per cancer type
        
        for sample in sample_list:
            cancer_type = sample['cancer_type']
            
            if len(selected_samples) >= target_count:
                break
                
            if cancer_type_counts[cancer_type] < max_per_cancer:
                selected_samples.append(sample)
                cancer_type_counts[cancer_type] += 1
            elif len(selected_samples) < target_count * 0.9:  # If we're under 90% of target, relax balance
                selected_samples.append(sample)
                cancer_type_counts[cancer_type] += 1
        
        logger.info(f"Selected {len(selected_samples)} samples with balance: {dict(cancer_type_counts)}")
        return selected_samples
    
    def create_aggressive_50k_dataset(self, selected_samples: List[Dict]) -> str:
        """Create the aggressive 50k dataset from selected samples"""
        logger.info(f"Creating aggressive 50k dataset from {len(selected_samples)} samples...")
        
        # Convert to DataFrame format
        rows = []
        
        for sample in selected_samples:
            row = {
                'sample_id': sample['sample_id'],
                'cancer_type': sample['cancer_type'],
                'priority_score': sample.get('priority_score', 0),
                'file_count': sample.get('file_count', 0),
                'omics_count': len(sample.get('omics_data', {}))
            }
            
            # Add omics file paths (semicolon-separated)
            for omics_name in self.omics_types.keys():
                omics_data = sample.get('omics_data', {})
                if omics_name in omics_data:
                    file_paths = omics_data[omics_name]
                    row[omics_name] = ';'.join(file_paths)
                else:
                    row[omics_name] = ''
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_path / f"tcga_aggressive_50k_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Generate detailed statistics
        stats = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': len(df),
            'target_samples': self.target_samples,
            'achievement_rate': len(df) / self.target_samples * 100,
            'cancer_types': df['cancer_type'].value_counts().to_dict(),
            'omics_coverage': {},
            'quality_metrics': {
                'avg_priority_score': float(df['priority_score'].mean()),
                'avg_file_count': float(df['file_count'].mean()),
                'avg_omics_count': float(df['omics_count'].mean())
            },
            'processing_method': 'aggressive_comprehensive_discovery'
        }
        
        # Calculate omics coverage
        for omics_name in self.omics_types.keys():
            if omics_name in df.columns:
                non_empty = (df[omics_name] != '').sum()
                stats['omics_coverage'][omics_name] = {
                    'samples_with_data': int(non_empty),
                    'coverage_percent': float((non_empty / len(df)) * 100)
                }
        
        # Save statistics
        stats_file = self.output_path / f"aggressive_50k_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Aggressive 50k dataset created: {output_file}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Achievement rate: {stats['achievement_rate']:.1f}%")
        logger.info(f"Statistics saved: {stats_file}")
        
        return str(output_file)
    
    def run_aggressive_processing(self) -> str:
        """Run the complete aggressive 50k processing"""
        logger.info("🔥 Starting Aggressive 50k Processing...")
        logger.info("This will extract MAXIMUM samples from existing raw data")
        
        try:
            # Step 1: Comprehensive discovery of all samples
            all_samples = self.parallel_comprehensive_discovery()
            
            if len(all_samples) == 0:
                logger.error("No samples discovered - check raw data path")
                return ""
            
            # Step 2: Prioritize and select best samples
            target_count = min(self.target_samples, len(all_samples))
            selected_samples = self.prioritize_and_select_samples(all_samples, target_count)
            
            # Step 3: Create final dataset
            final_dataset = self.create_aggressive_50k_dataset(selected_samples)
            
            logger.info("🎉 Aggressive processing complete!")
            return final_dataset
            
        except Exception as e:
            logger.error(f"Aggressive processing failed: {e}")
            raise


def main():
    """Main execution function"""
    print("="*70)
    print("🔥 AGGRESSIVE 50K TCGA PROCESSOR")
    print("="*70) 
    print("Extracting MAXIMUM samples from existing raw data")
    print("Uses aggressive discovery and looser criteria")
    print("100% REAL TCGA DATA - NO SYNTHETIC CONTAMINATION")
    print("="*70)
    
    processor = Aggressive50kProcessor()
    result = processor.run_aggressive_processing()
    
    if result:
        print("\\n" + "="*60)
        print("🎉 AGGRESSIVE PROCESSING COMPLETE!")
        print("="*60)
        print(f"Dataset created: {result}")
        print("Check the statistics file for detailed analysis")
        print("="*60)
    else:
        print("\\n" + "="*60)
        print("❌ PROCESSING FAILED")
        print("="*60)
        print("Check logs for details")
        print("="*60)


if __name__ == "__main__":
    main()
