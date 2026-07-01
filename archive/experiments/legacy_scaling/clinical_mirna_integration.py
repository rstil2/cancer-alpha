#!/usr/bin/env python3
"""
Clinical and miRNA Data Integration
===================================

Add clinical and miRNA data across all 33 TCGA cancer types to boost 
sample counts by ~3000-4000 additional samples.

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import subprocess
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinicalMiRNAIntegrator:
    """Integrate clinical and miRNA data across all cancer types"""
    
    def __init__(self):
        # All 33 TCGA cancer types
        self.all_cancer_types = [
            'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL',
            'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC',
            'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG',
            'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV',
            'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC',
            'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM',
            'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM'
        ]
        
        self.additional_data_types = [
            'Clinical Supplement',
            'miRNA Expression Quantification'
        ]
        
        # Estimate sample boost per cancer type
        self.estimated_boost_per_type = 100  # Conservative estimate
        self.total_estimated_boost = len(self.all_cancer_types) * len(self.additional_data_types) * self.estimated_boost_per_type
        
    def download_additional_data_for_cancer(self, cancer_type):
        """Download clinical and miRNA data for a specific cancer type"""
        
        logger.info(f"🔬 Processing {cancer_type}...")
        
        successful_downloads = 0
        
        for data_type in self.additional_data_types:
            try:
                cmd = [
                    'python', 'cancer_genomics_ai_demo_minimal/scalable_tcga_downloader.py',
                    '--cancer-types', cancer_type.replace('TCGA-', ''),
                    '--data-types', data_type,
                    '--max-samples', '200',  # Get more samples for each type
                    '--output-dir', f'data/clinical_mirna_expansion'
                ]
                
                logger.info(f"   📥 Downloading {data_type} for {cancer_type}...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout
                
                if result.returncode == 0:
                    successful_downloads += 1
                    logger.info(f"   ✅ Successfully downloaded {data_type}")
                else:
                    logger.warning(f"   ⚠️ Issues with {data_type}: Limited availability")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"   ⏰ Timeout downloading {data_type}")
            except Exception as e:
                logger.error(f"   ❌ Error downloading {data_type}: {e}")
        
        return successful_downloads
    
    def execute_clinical_mirna_integration(self):
        """Execute clinical and miRNA integration across all cancer types"""
        
        logger.info("🧬 CLINICAL & miRNA DATA INTEGRATION STRATEGY")
        logger.info("=" * 70)
        logger.info(f"🎯 Target cancer types: {len(self.all_cancer_types)}")
        logger.info(f"📊 Additional data types: {len(self.additional_data_types)}")
        logger.info(f"🚀 Estimated sample boost: ~{self.total_estimated_boost:,}")
        
        # Process cancer types in parallel for efficiency
        successful_integrations = 0
        
        # Process in batches to avoid overwhelming servers
        batch_size = 5
        cancer_batches = [self.all_cancer_types[i:i+batch_size] 
                         for i in range(0, len(self.all_cancer_types), batch_size)]
        
        for batch_num, batch in enumerate(cancer_batches):
            logger.info(f"\n📦 Processing batch {batch_num + 1}/{len(cancer_batches)}: {batch}")
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_cancer = {
                    executor.submit(self.download_additional_data_for_cancer, cancer_type): cancer_type
                    for cancer_type in batch
                }
                
                for future in as_completed(future_to_cancer):
                    cancer_type = future_to_cancer[future]
                    try:
                        successful_downloads = future.result()
                        if successful_downloads > 0:
                            successful_integrations += 1
                        logger.info(f"   📊 {cancer_type}: {successful_downloads}/2 data types successful")
                    except Exception as e:
                        logger.error(f"   ❌ Error processing {cancer_type}: {e}")
            
            # Delay between batches
            time.sleep(30)
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 CLINICAL & miRNA INTEGRATION COMPLETE!")
        logger.info(f"📊 Successful integrations: {successful_integrations}/{len(self.all_cancer_types)}")
        logger.info(f"🎯 Expected sample boost: ~{successful_integrations * self.estimated_boost_per_type * len(self.additional_data_types):,}")
        
        return successful_integrations

if __name__ == "__main__":
    integrator = ClinicalMiRNAIntegrator()
    success = integrator.execute_clinical_mirna_integration()
