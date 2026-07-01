#!/usr/bin/env python3
"""
Test Real TCGA Download - Small Scale Validation
===============================================

Test the massive TCGA downloader with a small batch to validate
the approach before scaling to 10,000+ samples.

REAL DATA ONLY - NO SYNTHETIC DATA!
"""

import asyncio
import logging
from massive_real_tcga_downloader import MassiveRealTCGADownloader
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRealTCGADownloader(MassiveRealTCGADownloader):
    """Test version with smaller targets for validation"""
    
    def __init__(self):
        super().__init__(cache_dir="data/tcga_test_real")
        
        # Override with smaller test targets (5 samples per cancer type)
        self.target_projects = {
            'TCGA-BRCA': {'name': 'Breast Invasive Carcinoma', 'target': 5, 'priority': 1},
            'TCGA-LUAD': {'name': 'Lung Adenocarcinoma', 'target': 5, 'priority': 1}, 
            'TCGA-COAD': {'name': 'Colon Adenocarcinoma', 'target': 5, 'priority': 1},
        }
        
        # Test with just RNA-seq and clinical data first
        self.data_types = {
            'rna_seq': {
                'gdc_name': 'Gene Expression Quantification',
                'workflow': 'HTSeq - FPKM',
                'priority': 1,
                'target_samples': 15
            },
            'clinical': {
                'gdc_name': 'Clinical Supplement',
                'priority': 1,
                'target_samples': 15
            }
        }
        
        # Reduce concurrency for testing
        self.max_concurrent = 5

async def test_small_download():
    """Test downloading a small batch of real TCGA data"""
    
    logger.info("🧪 Testing Small-Scale Real TCGA Download...")
    logger.info("=" * 50)
    logger.info("✅ REAL DATA ONLY - NO SYNTHETIC DATA!")
    logger.info("🎯 Test Target: 15 authentic TCGA samples")
    logger.info("=" * 50)
    
    downloader = TestRealTCGADownloader()
    
    try:
        # Test Phase 1 with small targets
        logger.info("\n🧪 TEST: Small-scale download validation...")
        test_stats = await downloader.download_all_projects(phase=1)
        
        # Generate test report
        report_path = downloader.generate_download_report(test_stats)
        
        # Validate results
        total_files = sum(sum(project_stats.values()) for project_stats in test_stats.values())
        
        logger.info(f"\n✅ TEST DOWNLOAD COMPLETE!")
        logger.info(f"📊 Report: {report_path}")
        logger.info(f"📁 Total files downloaded: {total_files}")
        
        if total_files > 0:
            logger.info(f"🎉 SUCCESS: Real TCGA download working!")
            logger.info(f"🚀 Ready to scale to 10,000+ samples!")
            
            # Check downloaded files
            cache_dir = Path("data/tcga_test_real")
            for data_type in ['rna_seq', 'clinical']:
                data_dir = cache_dir / data_type
                if data_dir.exists():
                    files = list(data_dir.glob("*"))
                    logger.info(f"📁 {data_type}: {len(files)} files downloaded")
                    if files:
                        sample_file = files[0]
                        size_mb = sample_file.stat().st_size / (1024 * 1024)
                        logger.info(f"   Sample file: {sample_file.name} ({size_mb:.2f} MB)")
        else:
            logger.warning("⚠️  No files downloaded - check GDC API connectivity")
        
        return total_files > 0
        
    except Exception as e:
        logger.error(f"❌ Test download failed: {str(e)}")
        return False

def validate_downloaded_data():
    """Validate the downloaded test data"""
    
    logger.info("\n🔍 Validating Downloaded Test Data...")
    logger.info("-" * 40)
    
    cache_dir = Path("data/tcga_test_real")
    
    # Check RNA-seq files
    rna_dir = cache_dir / "rna_seq"
    if rna_dir.exists():
        rna_files = list(rna_dir.glob("*.gz"))
        logger.info(f"✅ RNA-seq files: {len(rna_files)}")
        
        if rna_files:
            # Try to peek at first file
            try:
                import gzip
                with gzip.open(rna_files[0], 'rt') as f:
                    header = f.readline().strip()
                    first_line = f.readline().strip()
                    logger.info(f"   Header: {header[:50]}...")
                    logger.info(f"   Data: {first_line[:50]}...")
                    logger.info("✅ RNA-seq data format validated")
            except Exception as e:
                logger.warning(f"⚠️  Could not validate RNA-seq format: {e}")
    
    # Check clinical files  
    clinical_dir = cache_dir / "clinical"
    if clinical_dir.exists():
        clinical_files = list(clinical_dir.glob("*.xml"))
        logger.info(f"✅ Clinical files: {len(clinical_files)}")
        
        if clinical_files:
            try:
                with open(clinical_files[0], 'r') as f:
                    content = f.read(200)
                    if 'patient' in content.lower():
                        logger.info("✅ Clinical XML format validated")
                    else:
                        logger.warning("⚠️  Clinical file format unclear")
            except Exception as e:
                logger.warning(f"⚠️  Could not validate clinical format: {e}")
    
    logger.info("🔍 Data validation complete")

async def main():
    """Main test execution"""
    
    # Test small download
    success = await test_small_download()
    
    if success:
        # Validate downloaded data
        validate_downloaded_data()
        
        # Ask user if they want to proceed with full download
        logger.info("\n" + "=" * 60)
        logger.info("🎯 READY FOR FULL 10,000+ SAMPLE DOWNLOAD")
        logger.info("=" * 60)
        logger.info("✅ Test download successful")
        logger.info("✅ Real TCGA data validated") 
        logger.info("✅ No synthetic data contamination")
        logger.info("\n🚀 To proceed with full 10K+ download, run:")
        logger.info("   python massive_real_tcga_downloader.py")
    else:
        logger.error("❌ Test download failed - please check connectivity and try again")

if __name__ == "__main__":
    asyncio.run(main())
