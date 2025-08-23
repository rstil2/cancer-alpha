#!/usr/bin/env python3
"""
Complete 50K+ Sample Integrator
===============================

Process ALL 33 TCGA cancer types found in data/production_tcga 
to create the complete 50K+ sample dataset.

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import os
import pandas as pd
import json
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Complete50KIntegrator:
    """Integrate all available TCGA samples to reach 50K+"""
    
    def __init__(self, base_dir="data/production_tcga", output_dir="data/complete_50k"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_counts = defaultdict(int)
        self.total_samples = 0
        
    def extract_sample_id_from_filename(self, filename):
        """Extract TCGA sample ID from filename"""
        import re
        
        # Try different TCGA patterns
        patterns = [
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})',  # Basic TCGA pattern
            r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z])',  # Extended pattern
            r'TCGA\.([A-Z0-9]{2})\.([A-Z0-9]{4})',  # Dot notation
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                if 'TCGA.' in pattern:
                    return f"TCGA-{match.group(1)}-{match.group(2)}"
                else:
                    return match.group(1)
        
        return None
    
    def process_all_cancer_types(self):
        """Process all cancer types found in the data directory"""
        
        logger.info("🔍 Scanning for all TCGA cancer types...")
        
        # Find all cancer type directories
        cancer_types = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith('TCGA-'):
                cancer_types.append(item.name)
        
        # Also check within omics type directories
        omics_dirs = ['expression', 'mutations', 'copy_number', 'methylation', 'protein', 'clinical']
        for omics_dir in omics_dirs:
            omics_path = self.base_dir / omics_dir
            if omics_path.exists():
                for item in omics_path.iterdir():
                    if item.is_dir() and item.name.startswith('TCGA-') and item.name not in cancer_types:
                        cancer_types.append(item.name)
        
        cancer_types = sorted(set(cancer_types))
        logger.info(f"📊 Found {len(cancer_types)} TCGA cancer types: {cancer_types}")
        
        # Process each cancer type
        all_samples = []
        
        for cancer_type in tqdm(cancer_types, desc="Processing cancer types"):
            samples = self.process_cancer_type(cancer_type)
            all_samples.extend(samples)
            
        logger.info(f"✅ Total samples processed: {len(all_samples)}")
        
        return all_samples
    
    def process_cancer_type(self, cancer_type):
        """Process all files for a specific cancer type"""
        
        samples = []
        sample_files = defaultdict(lambda: defaultdict(list))
        
        # Check direct cancer type directory
        cancer_dir = self.base_dir / cancer_type
        if cancer_dir.exists():
            for file_path in cancer_dir.rglob("*"):
                if file_path.is_file():
                    sample_id = self.extract_sample_id_from_filename(file_path.name)
                    if sample_id:
                        sample_files[sample_id]['direct'].append(str(file_path))
        
        # Check within omics directories
        omics_types = ['expression', 'mutations', 'copy_number', 'methylation', 'protein', 'clinical']
        
        for omics_type in omics_types:
            omics_cancer_dir = self.base_dir / omics_type / cancer_type
            if omics_cancer_dir.exists():
                for file_path in omics_cancer_dir.rglob("*"):
                    if file_path.is_file():
                        sample_id = self.extract_sample_id_from_filename(file_path.name)
                        if sample_id:
                            sample_files[sample_id][omics_type].append(str(file_path))
        
        # Create sample records
        for sample_id, omics_data in sample_files.items():
            sample_record = {
                'sample_id': sample_id,
                'cancer_type': cancer_type,
                'file_count': sum(len(files) for files in omics_data.values()),
                'omics_types': len([ot for ot in omics_data.keys() if omics_data[ot]]),
            }
            
            # Add omics-specific file counts
            for omics_type in omics_types + ['direct']:
                sample_record[f'{omics_type}_files'] = len(omics_data[omics_type])
            
            samples.append(sample_record)
        
        self.sample_counts[cancer_type] = len(samples)
        
        return samples
    
    def create_integrated_dataset(self):
        """Create the complete integrated dataset"""
        
        logger.info("🚀 Starting complete 50K+ sample integration...")
        
        # Process all samples
        all_samples = self.process_all_cancer_types()
        
        if not all_samples:
            logger.error("❌ No samples found!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_samples)
        
        # Calculate statistics
        total_samples = len(df)
        cancer_type_counts = df['cancer_type'].value_counts().to_dict()
        
        # Multi-omics samples (samples with > 1 omics type)
        multi_omics_samples = len(df[df['omics_types'] > 1])
        
        logger.info(f"📊 INTEGRATION COMPLETE:")
        logger.info(f"   Total samples: {total_samples:,}")
        logger.info(f"   Multi-omics samples: {multi_omics_samples:,}")
        logger.info(f"   Cancer types: {len(cancer_type_counts)}")
        
        # Check if we reached 50K+
        if total_samples >= 50000:
            logger.info(f"🎉 SUCCESS! Reached {total_samples:,} samples (target: 50,000+)")
        else:
            logger.info(f"⚠️  Current: {total_samples:,} samples (need {50000 - total_samples:,} more for 50K+)")
        
        # Save results
        output_file = self.output_dir / f"complete_50k_samples_{total_samples}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Dataset saved to: {output_file}")
        
        # Save summary
        summary = {
            "total_samples": total_samples,
            "target_reached": total_samples >= 50000,
            "multi_omics_samples": multi_omics_samples,
            "cancer_type_counts": cancer_type_counts,
            "omics_coverage": {
                "expression": int(df['expression_files'].sum()),
                "mutations": int(df['mutations_files'].sum()),
                "copy_number": int(df['copy_number_files'].sum()),
                "methylation": int(df['methylation_files'].sum()),
                "protein": int(df['protein_files'].sum()),
                "clinical": int(df['clinical_files'].sum()),
            }
        }
        
        summary_file = self.output_dir / f"complete_50k_summary_{total_samples}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Summary saved to: {summary_file}")
        
        # Display top cancer types
        logger.info("🏆 TOP CANCER TYPES BY SAMPLE COUNT:")
        for cancer_type, count in sorted(cancer_type_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"   {cancer_type}: {count:,} samples")
        
        return df, summary

if __name__ == "__main__":
    integrator = Complete50KIntegrator()
    df, summary = integrator.create_integrated_dataset()
