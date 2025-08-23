#!/usr/bin/env python3
"""
Targeted High-Yield Cancer Type Expansion
=========================================

Target the top 3 cancer types (KIRC, UCEC, OV) for massive sample expansion
to contribute ~6000+ samples toward our 50K+ goal.

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import subprocess
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TargetedHighYieldExpansion:
    """Expand high-yield cancer types to maximize sample gain"""
    
    def __init__(self):
        self.high_yield_targets = [
            {
                'cancer_type': 'TCGA-KIRC',
                'target_samples': 3000,
                'priority': 1,
                'description': 'Kidney Renal Clear Cell Carcinoma'
            },
            {
                'cancer_type': 'TCGA-UCEC', 
                'target_samples': 2500,
                'priority': 2,
                'description': 'Uterine Corpus Endometrial Carcinoma'
            },
            {
                'cancer_type': 'TCGA-OV',
                'target_samples': 2000,
                'priority': 3, 
                'description': 'Ovarian Serous Cystadenocarcinoma'
            }
        ]
        
        self.total_target = sum(target['target_samples'] for target in self.high_yield_targets)
        
    def expand_cancer_type(self, cancer_info):
        """Expand a single cancer type using multiple data modalities"""
        
        cancer_type = cancer_info['cancer_type']
        target_samples = cancer_info['target_samples']
        description = cancer_info['description']
        
        logger.info(f"🎯 Expanding {cancer_type} ({description})")
        logger.info(f"   Target: {target_samples:,} additional samples")
        
        # Strategy: Download multiple data types to maximize sample coverage
        data_types = [
            'Gene Expression Quantification',
            'Masked Somatic Mutation', 
            'Copy Number Segment',
            'Methylation Beta Value',
            'Protein Expression Quantification',
            'Clinical Supplement',
            'miRNA Expression Quantification'
        ]
        
        expansion_commands = []
        
        # Create expansion commands for each data type
        for data_type in data_types:
            # Use individual data type downloads for maximum coverage
            cmd = [
                'python', 'cancer_genomics_ai_demo_minimal/scalable_tcga_downloader.py',
                '--cancer-types', cancer_type.replace('TCGA-', ''),
                '--data-types', data_type,
                '--max-samples', str(target_samples // len(data_types) + 100),  # Overlap for redundancy
                '--output-dir', f'data/targeted_expansion_{cancer_type.lower()}'
            ]
            expansion_commands.append(cmd)
        
        # Execute expansion commands
        successful_downloads = 0
        for i, cmd in enumerate(expansion_commands):
            try:
                logger.info(f"   📥 Downloading {data_types[i]}...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
                
                if result.returncode == 0:
                    successful_downloads += 1
                    logger.info(f"   ✅ Successfully downloaded {data_types[i]}")
                else:
                    logger.warning(f"   ⚠️ Issues with {data_types[i]}: {result.stderr[:200]}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"   ⏰ Timeout downloading {data_types[i]}")
            except Exception as e:
                logger.error(f"   ❌ Error downloading {data_types[i]}: {e}")
        
        logger.info(f"   📊 Completed {successful_downloads}/{len(data_types)} data type downloads for {cancer_type}")
        return successful_downloads
    
    def execute_targeted_expansion(self):
        """Execute the complete targeted expansion strategy"""
        
        logger.info("🚀 TARGETED HIGH-YIELD EXPANSION STRATEGY")
        logger.info("=" * 60)
        logger.info(f"🎯 Target: {self.total_target:,} additional samples")
        logger.info(f"📊 Cancer types: {len(self.high_yield_targets)}")
        
        total_successful = 0
        
        for target in self.high_yield_targets:
            logger.info(f"\n📍 Processing {target['cancer_type']} (Priority {target['priority']})")
            
            successful = self.expand_cancer_type(target)
            total_successful += successful
            
            # Small delay between cancer types to avoid overwhelming servers
            time.sleep(10)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 TARGETED EXPANSION COMPLETE!")
        logger.info(f"📊 Successful downloads: {total_successful}")
        logger.info(f"🎯 Expected sample boost: ~{self.total_target:,}")
        
        return total_successful >= len(self.high_yield_targets) * 5  # At least 5 data types per cancer

if __name__ == "__main__":
    expander = TargetedHighYieldExpansion()
    success = expander.execute_targeted_expansion()
