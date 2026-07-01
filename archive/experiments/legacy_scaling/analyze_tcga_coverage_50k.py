#!/usr/bin/env python3
"""
Ultra-Massive TCGA Data Coverage Analyzer for 50K+ Sample Scaling

This script analyzes our current TCGA data coverage across all cancer types and data types
to create a strategic plan for scaling to 50,000+ authentic samples.
"""

import os
import sys
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tcga_coverage_analysis_50k.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TCGACoverageAnalyzer:
    """Analyze TCGA data coverage to plan ultra-massive scaling"""
    
    def __init__(self, data_dirs):
        self.data_dirs = data_dirs
        self.cancer_types = set()
        self.data_types = set()
        self.file_inventory = defaultdict(lambda: defaultdict(list))
        self.sample_coverage = defaultdict(set)
        
        # All 33 TCGA cancer types for comprehensive coverage
        self.all_tcga_projects = [
            'TCGA-LAML', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-LGG', 'TCGA-BRCA',
            'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA',
            'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP',
            'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV',
            'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC',
            'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM',
            'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM'
        ]
        
        # Target data types for maximum sample coverage
        self.target_data_types = [
            'mutations', 'expression', 'protein', 'copy_number', 
            'methylation', 'clinical', 'miRNA'
        ]
    
    def scan_data_directories(self):
        """Scan all data directories for comprehensive inventory"""
        logger.info("Starting comprehensive TCGA data directory scan...")
        total_files = 0
        
        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                logger.warning(f"Directory not found: {data_dir}")
                continue
                
            logger.info(f"Scanning {data_dir}...")
            
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.startswith('.'):
                        continue
                        
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, data_dir)
                    
                    # Extract data type and cancer type from path
                    path_parts = relative_path.split(os.sep)
                    
                    data_type = 'unknown'
                    cancer_type = 'unknown'
                    
                    # Identify data type from path
                    for part in path_parts:
                        if any(dt in part.lower() for dt in self.target_data_types):
                            for dt in self.target_data_types:
                                if dt in part.lower():
                                    data_type = dt
                                    break
                            break
                    
                    # Identify cancer type from path or filename
                    for project in self.all_tcga_projects:
                        if project in relative_path:
                            cancer_type = project
                            break
                    
                    # Try to extract sample ID from filename
                    sample_id = self.extract_sample_id(file)
                    
                    self.file_inventory[cancer_type][data_type].append({
                        'file_path': file_path,
                        'relative_path': relative_path,
                        'filename': file,
                        'sample_id': sample_id,
                        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    })
                    
                    if sample_id:
                        self.sample_coverage[sample_id].add(data_type)
                    
                    self.cancer_types.add(cancer_type)
                    self.data_types.add(data_type)
                    total_files += 1
        
        logger.info(f"Scanned {total_files} files across {len(self.cancer_types)} cancer types")
        logger.info(f"Identified data types: {sorted(self.data_types)}")
        logger.info(f"Cancer types found: {sorted(self.cancer_types)}")
        
    def extract_sample_id(self, filename):
        """Extract TCGA sample ID from filename using multiple patterns"""
        import re
        
        # Pattern 1: Direct TCGA sample ID in filename
        tcga_pattern = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]?)', filename)
        if tcga_pattern:
            return tcga_pattern.group(1)
        
        # Pattern 2: UUID filename (will need mapping)
        uuid_pattern = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', filename)
        if uuid_pattern:
            return f"UUID_{uuid_pattern.group(1)}"
        
        return None
    
    def analyze_coverage(self):
        """Analyze current coverage and identify gaps for 50K+ scaling"""
        logger.info("Analyzing current TCGA data coverage...")
        
        coverage_stats = {
            'total_files': sum(len(files) for cancer_files in self.file_inventory.values() 
                             for files in cancer_files.values()),
            'total_samples': len(self.sample_coverage),
            'multi_omics_samples': sum(1 for sample_types in self.sample_coverage.values() 
                                     if len(sample_types) > 1),
            'cancer_type_coverage': {},
            'data_type_coverage': {},
            'gaps_analysis': {},
            'scaling_recommendations': {}
        }
        
        # Cancer type coverage analysis
        for cancer_type in self.cancer_types:
            cancer_files = self.file_inventory[cancer_type]
            total_cancer_files = sum(len(files) for files in cancer_files.values())
            
            coverage_stats['cancer_type_coverage'][cancer_type] = {
                'total_files': total_cancer_files,
                'data_types': list(cancer_files.keys()),
                'data_type_counts': {dt: len(files) for dt, files in cancer_files.items()}
            }
        
        # Data type coverage analysis
        for data_type in self.data_types:
            type_files = []
            type_cancers = set()
            for cancer_type, cancer_files in self.file_inventory.items():
                if data_type in cancer_files:
                    type_files.extend(cancer_files[data_type])
                    type_cancers.add(cancer_type)
            
            coverage_stats['data_type_coverage'][data_type] = {
                'total_files': len(type_files),
                'cancer_types': list(type_cancers),
                'cancer_count': len(type_cancers)
            }
        
        # Gap analysis for 50K+ scaling
        missing_cancer_types = set(self.all_tcga_projects) - self.cancer_types
        
        coverage_stats['gaps_analysis'] = {
            'missing_cancer_types': list(missing_cancer_types),
            'missing_count': len(missing_cancer_types),
            'low_coverage_types': [],
            'data_type_gaps': {}
        }
        
        # Identify low coverage cancer types (< 100 files)
        for cancer_type, info in coverage_stats['cancer_type_coverage'].items():
            if info['total_files'] < 100:
                coverage_stats['gaps_analysis']['low_coverage_types'].append({
                    'cancer_type': cancer_type,
                    'file_count': info['total_files']
                })
        
        # Identify data type gaps
        for cancer_type in self.all_tcga_projects:
            if cancer_type in self.file_inventory:
                available_types = set(self.file_inventory[cancer_type].keys())
                missing_types = set(self.target_data_types) - available_types
                if missing_types:
                    coverage_stats['gaps_analysis']['data_type_gaps'][cancer_type] = list(missing_types)
        
        # Generate scaling recommendations
        current_samples = coverage_stats['total_samples']
        target_samples = 50000
        
        scaling_factor = target_samples / current_samples if current_samples > 0 else 50
        
        coverage_stats['scaling_recommendations'] = {
            'current_samples': current_samples,
            'target_samples': target_samples,
            'scaling_factor': scaling_factor,
            'priority_actions': self.generate_scaling_priorities(coverage_stats),
            'estimated_download_requirements': self.estimate_download_requirements(coverage_stats)
        }
        
        return coverage_stats
    
    def generate_scaling_priorities(self, stats):
        """Generate priority actions for scaling to 50K+ samples"""
        priorities = []
        
        # Priority 1: Download missing cancer types
        if stats['gaps_analysis']['missing_cancer_types']:
            priorities.append({
                'priority': 1,
                'action': 'Download missing cancer types',
                'targets': stats['gaps_analysis']['missing_cancer_types'][:10],  # Top 10
                'expected_samples': len(stats['gaps_analysis']['missing_cancer_types']) * 500
            })
        
        # Priority 2: Expand low-coverage cancer types
        if stats['gaps_analysis']['low_coverage_types']:
            low_coverage = sorted(stats['gaps_analysis']['low_coverage_types'], 
                                key=lambda x: x['file_count'])[:5]
            priorities.append({
                'priority': 2,
                'action': 'Expand low-coverage cancer types',
                'targets': [item['cancer_type'] for item in low_coverage],
                'expected_samples': len(low_coverage) * 1000
            })
        
        # Priority 3: Add missing data types to existing cancers
        gap_cancers = list(stats['gaps_analysis']['data_type_gaps'].keys())[:10]
        if gap_cancers:
            priorities.append({
                'priority': 3,
                'action': 'Add missing data types to existing cancers',
                'targets': gap_cancers,
                'expected_samples': len(gap_cancers) * 200
            })
        
        # Priority 4: Expand high-yield data types globally
        high_yield_types = ['expression', 'mutations', 'copy_number', 'methylation']
        priorities.append({
            'priority': 4,
            'action': 'Expand high-yield data types globally',
            'targets': high_yield_types,
            'expected_samples': 20000
        })
        
        return priorities
    
    def estimate_download_requirements(self, stats):
        """Estimate download requirements for 50K+ scaling"""
        current_files = stats['total_files']
        current_samples = stats['total_samples']
        target_samples = 50000
        
        # Estimate files per sample ratio
        files_per_sample = current_files / current_samples if current_samples > 0 else 5
        
        # Calculate required additional files
        additional_samples_needed = target_samples - current_samples
        additional_files_needed = int(additional_samples_needed * files_per_sample)
        
        # Estimate download size (assuming average 50MB per file)
        avg_file_size_mb = 50
        total_download_size_gb = (additional_files_needed * avg_file_size_mb) / 1024
        
        return {
            'additional_samples_needed': additional_samples_needed,
            'additional_files_needed': additional_files_needed,
            'estimated_download_size_gb': total_download_size_gb,
            'estimated_download_time_hours': additional_files_needed / 100,  # ~100 files/hour
            'priority_cancer_types': stats['gaps_analysis']['missing_cancer_types'][:15]
        }
    
    def save_analysis(self, stats):
        """Save comprehensive analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON analysis
        analysis_file = f"tcga_coverage_analysis_50k_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for cancer_type, info in stats['cancer_type_coverage'].items():
            summary_data.append({
                'cancer_type': cancer_type,
                'total_files': info['total_files'],
                'data_types_count': len(info['data_types']),
                'data_types': ', '.join(info['data_types'][:5])  # First 5
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f"tcga_coverage_summary_50k_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Analysis saved to {analysis_file} and {summary_file}")
        
        return analysis_file, summary_file
    
    def print_summary(self, stats):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("TCGA DATA COVERAGE ANALYSIS FOR 50K+ SAMPLE SCALING")
        print("="*80)
        
        print(f"\nCURRENT STATUS:")
        print(f"  Total Files: {stats['total_files']:,}")
        print(f"  Total Samples: {stats['total_samples']:,}")
        print(f"  Multi-omics Samples: {stats['multi_omics_samples']:,}")
        print(f"  Cancer Types: {len(stats['cancer_type_coverage'])}")
        print(f"  Data Types: {len(stats['data_type_coverage'])}")
        
        print(f"\nSCALING REQUIREMENTS:")
        reqs = stats['scaling_recommendations']['estimated_download_requirements']
        print(f"  Target Samples: {stats['scaling_recommendations']['target_samples']:,}")
        print(f"  Additional Samples Needed: {reqs['additional_samples_needed']:,}")
        print(f"  Additional Files Needed: {reqs['additional_files_needed']:,}")
        print(f"  Estimated Download Size: {reqs['estimated_download_size_gb']:.1f} GB")
        print(f"  Estimated Download Time: {reqs['estimated_download_time_hours']:.1f} hours")
        
        print(f"\nPRIORITY ACTIONS:")
        for i, action in enumerate(stats['scaling_recommendations']['priority_actions'], 1):
            print(f"  {i}. {action['action']}")
            print(f"     Targets: {', '.join(action['targets'][:3])}{'...' if len(action['targets']) > 3 else ''}")
            print(f"     Expected Samples: {action['expected_samples']:,}")
        
        print(f"\nMISSING CANCER TYPES ({len(stats['gaps_analysis']['missing_cancer_types'])}):")
        missing = stats['gaps_analysis']['missing_cancer_types']
        for i in range(0, len(missing), 6):
            print(f"  {', '.join(missing[i:i+6])}")
        
        print(f"\nTOP DATA TYPES BY COVERAGE:")
        sorted_types = sorted(stats['data_type_coverage'].items(), 
                            key=lambda x: x[1]['total_files'], reverse=True)
        for data_type, info in sorted_types[:8]:
            print(f"  {data_type}: {info['total_files']:,} files across {info['cancer_count']} cancer types")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    logger.info("Starting ultra-massive TCGA coverage analysis for 50K+ scaling...")
    
    # Define data directories to analyze
    base_dir = "/Users/stillwell/projects/cancer-alpha/data"
    data_dirs = [
        os.path.join(base_dir, "production_tcga"),
        os.path.join(base_dir, "tcga_ultra_massive"),
        os.path.join(base_dir, "tcga_real_fixed"),
        os.path.join(base_dir, "tcga"),
        os.path.join(base_dir, "tcga_massive_real")
    ]
    
    # Filter to existing directories
    existing_dirs = [d for d in data_dirs if os.path.exists(d)]
    logger.info(f"Analyzing {len(existing_dirs)} data directories")
    
    # Create analyzer and run analysis
    analyzer = TCGACoverageAnalyzer(existing_dirs)
    analyzer.scan_data_directories()
    stats = analyzer.analyze_coverage()
    
    # Save and display results
    analysis_file, summary_file = analyzer.save_analysis(stats)
    analyzer.print_summary(stats)
    
    logger.info(f"Analysis complete! Results saved to {analysis_file}")
    print(f"\nFull analysis available in: {analysis_file}")
    print(f"Summary available in: {summary_file}")

if __name__ == "__main__":
    main()
