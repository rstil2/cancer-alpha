#!/usr/bin/env python3
"""
Extended TCGA Sampling Strategy
===============================

Scale up sampling from all cancer types using our proven infrastructure
to contribute the final samples needed for 50K+ achievement.

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import subprocess
import logging
import time
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtendedTCGASampling:
    """Extended TCGA sampling across all cancer types for final push to 50K+"""
    
    def __init__(self):
        # All 33 TCGA cancer types with their current estimated sample counts
        self.cancer_expansion_targets = {
            'TCGA-BRCA': {'current': 2090, 'target': 3000, 'priority': 'high'},
            'TCGA-KIRC': {'current': 949, 'target': 2000, 'priority': 'high'},
            'TCGA-UCEC': {'current': 1067, 'target': 2500, 'priority': 'high'},
            'TCGA-GBM': {'current': 1064, 'target': 1800, 'priority': 'high'},
            'TCGA-OV': {'current': 1051, 'target': 2000, 'priority': 'high'},
            'TCGA-SKCM': {'current': 940, 'target': 1500, 'priority': 'medium'},
            'TCGA-BLCA': {'current': 827, 'target': 1400, 'priority': 'medium'},
            'TCGA-LUAD': {'current': 500, 'target': 1200, 'priority': 'medium'},
            'TCGA-LUSC': {'current': 500, 'target': 1200, 'priority': 'medium'},
            'TCGA-COAD': {'current': 500, 'target': 1100, 'priority': 'medium'},
            'TCGA-HNSC': {'current': 399, 'target': 1000, 'priority': 'medium'},
            'TCGA-LGG': {'current': 458, 'target': 1000, 'priority': 'medium'},
            'TCGA-PRAD': {'current': 500, 'target': 1200, 'priority': 'medium'},
            'TCGA-THCA': {'current': 414, 'target': 800, 'priority': 'medium'},
            'TCGA-STAD': {'current': 434, 'target': 800, 'priority': 'medium'},
            'TCGA-LIHC': {'current': 372, 'target': 700, 'priority': 'low'},
            'TCGA-KIRP': {'current': 575, 'target': 800, 'priority': 'low'},
            'TCGA-CESC': {'current': 597, 'target': 800, 'priority': 'low'},
            'TCGA-SARC': {'current': 500, 'target': 700, 'priority': 'low'},
            'TCGA-LAML': {'current': 151, 'target': 400, 'priority': 'low'},
            'TCGA-PAAD': {'current': 182, 'target': 400, 'priority': 'low'},
            'TCGA-PCPG': {'current': 187, 'target': 400, 'priority': 'low'},
            'TCGA-READ': {'current': 177, 'target': 400, 'priority': 'low'},
            'TCGA-TGCT': {'current': 156, 'target': 350, 'priority': 'low'},
            'TCGA-THYM': {'current': 122, 'target': 300, 'priority': 'low'},
            'TCGA-ESCA': {'current': 197, 'target': 400, 'priority': 'low'},
            'TCGA-KICH': {'current': 91, 'target': 200, 'priority': 'low'},
            'TCGA-MESO': {'current': 87, 'target': 200, 'priority': 'low'},
            'TCGA-ACC': {'current': 79, 'target': 150, 'priority': 'low'},
            'TCGA-UCS': {'current': 57, 'target': 120, 'priority': 'low'},
            'TCGA-UVM': {'current': 80, 'target': 150, 'priority': 'low'},
            'TCGA-CHOL': {'current': 44, 'target': 100, 'priority': 'low'},
            'TCGA-DLBC': {'current': 48, 'target': 100, 'priority': 'low'},
        }
        
        self.total_additional_target = sum(
            max(0, info['target'] - info['current']) 
            for info in self.cancer_expansion_targets.values()
        )
        
    def calculate_expansion_needed(self, cancer_type, info):
        """Calculate how many additional samples needed for this cancer type"""
        current = info['current']
        target = info['target']
        expansion_needed = max(0, target - current)
        return expansion_needed
    
    def expand_cancer_type_comprehensive(self, cancer_type, expansion_info):
        """Comprehensively expand a cancer type using all available data modalities"""
        
        expansion_needed = self.calculate_expansion_needed(cancer_type, expansion_info)
        if expansion_needed <= 0:
            logger.info(f"✅ {cancer_type}: Already at target ({expansion_info['current']} samples)")
            return True
        
        logger.info(f"🎯 Expanding {cancer_type}: +{expansion_needed:,} samples needed")
        logger.info(f"   Priority: {expansion_info['priority'].upper()}")
        
        # Use comprehensive data type collection
        data_types = [
            'Gene Expression Quantification',
            'Masked Somatic Mutation',
            'Copy Number Segment', 
            'Methylation Beta Value',
            'Protein Expression Quantification',
            'Clinical Supplement',
            'miRNA Expression Quantification'
        ]
        
        successful_expansions = 0
        
        # Download each data type with aggressive sampling
        for data_type in data_types:
            try:
                # Calculate samples per data type (with overlap for redundancy)
                samples_per_type = max(50, expansion_needed // len(data_types) + 50)
                
                cmd = [
                    'python', 'cancer_genomics_ai_demo_minimal/scalable_tcga_downloader.py',
                    '--cancer-types', cancer_type.replace('TCGA-', ''),
                    '--data-types', data_type,
                    '--max-samples', str(samples_per_type),
                    '--output-dir', f'data/extended_sampling_{cancer_type.lower()}'
                ]
                
                logger.info(f"   📥 Downloading {data_type} ({samples_per_type} samples)...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 min timeout
                
                if result.returncode == 0:
                    successful_expansions += 1
                    logger.info(f"   ✅ Successfully expanded with {data_type}")
                else:
                    logger.warning(f"   ⚠️ Limited success with {data_type}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"   ⏰ Timeout downloading {data_type}")
            except Exception as e:
                logger.error(f"   ❌ Error downloading {data_type}: {e}")
        
        success_rate = successful_expansions / len(data_types)
        logger.info(f"   📊 {cancer_type}: {successful_expansions}/{len(data_types)} expansions successful ({success_rate:.1%})")
        
        return success_rate >= 0.5  # At least 50% success rate
    
    def execute_extended_sampling(self):
        """Execute extended sampling strategy across all cancer types"""
        
        logger.info("🚀 EXTENDED TCGA SAMPLING STRATEGY")
        logger.info("=" * 80)
        logger.info(f"🎯 Total additional samples target: {self.total_additional_target:,}")
        logger.info(f"📊 Cancer types to expand: {len(self.cancer_expansion_targets)}")
        
        # Process by priority for maximum impact
        priority_groups = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for cancer_type, info in self.cancer_expansion_targets.items():
            priority_groups[info['priority']].append((cancer_type, info))
        
        total_successful = 0
        
        for priority in ['high', 'medium', 'low']:
            cancer_list = priority_groups[priority]
            if not cancer_list:
                continue
                
            logger.info(f"\n🎯 Processing {priority.upper()} PRIORITY cancer types ({len(cancer_list)} types):")
            
            for cancer_type, info in cancer_list:
                logger.info(f"\n📍 {cancer_type} ({priority} priority)")
                
                success = self.expand_cancer_type_comprehensive(cancer_type, info)
                if success:
                    total_successful += 1
                
                # Brief delay between cancer types
                time.sleep(5)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 EXTENDED SAMPLING COMPLETE!")
        logger.info(f"📊 Successful expansions: {total_successful}/{len(self.cancer_expansion_targets)}")
        logger.info(f"🎯 Expected total sample boost: ~{self.total_additional_target:,}")
        
        return total_successful

if __name__ == "__main__":
    sampler = ExtendedTCGASampling()
    success = sampler.execute_extended_sampling()
