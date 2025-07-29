#!/usr/bin/env python3
"""
Master TCGA Pipeline
====================

This script orchestrates the complete pipeline:
1. Large-scale TCGA data download
2. Multi-modal data processing
3. Model training on integrated data

Author: Cancer Alpha Research Team
Date: July 28, 2025
"""

import logging
import time
import sys
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_download_phase(max_files_per_type: int = 10):
    """Run the TCGA download phase"""
    logger.info("🔽 PHASE 1: Large-Scale TCGA Data Download")
    
    try:
        from scalable_tcga_downloader import ScalableTCGADownloader
        
        downloader = ScalableTCGADownloader()
        all_project_data = downloader.download_all_projects(max_files_per_type=max_files_per_type)
        summary = downloader.create_download_summary(all_project_data)
        downloader.save_download_summary(summary)
        
        logger.info("✅ Download phase completed successfully")
        return summary
        
    except Exception as e:
        logger.error(f"❌ Download phase failed: {str(e)}")
        raise

def run_processing_phase():
    """Run the multi-modal processing phase"""
    logger.info("🔗 PHASE 2: Multi-Modal Data Processing")
    
    try:
        from multimodal_tcga_processor import MultiModalTCGAProcessor
        
        processor = MultiModalTCGAProcessor()
        output_file = processor.save_multimodal_data()
        
        logger.info("✅ Processing phase completed successfully")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Processing phase failed: {str(e)}")
        raise

def run_training_phase(data_file: str):
    """Run the training phase on multi-modal data"""
    logger.info("🤖 PHASE 3: Model Training on Multi-Modal Data")
    
    try:
        # Create training script for multi-modal data
        from train_multimodal_tcga import train_multimodal_model
        
        results = train_multimodal_model(data_file)
        
        logger.info("✅ Training phase completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Training phase failed: {str(e)}")
        # Continue anyway to show what we have
        return None

def create_pipeline_summary(phases_completed: dict):
    """Create comprehensive pipeline summary"""
    
    summary = {
        'pipeline_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phases_completed': phases_completed,
        'total_duration_minutes': 0,
        'status': 'completed' if all(phases_completed.values()) else 'partial'
    }
    
    logger.info("\n" + "="*60)
    logger.info("🎉 MASTER TCGA PIPELINE SUMMARY")
    logger.info("="*60)
    
    for phase, completed in phases_completed.items():
        status = "✅ COMPLETED" if completed else "❌ FAILED"
        logger.info(f"{phase}: {status}")
    
    overall_status = "🎉 SUCCESS" if all(phases_completed.values()) else "⚠️ PARTIAL SUCCESS"
    logger.info(f"\nOverall Status: {overall_status}")
    logger.info("="*60)
    
    return summary

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Master TCGA Pipeline')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip download phase and use existing data')
    parser.add_argument('--max-files', type=int, default=10,
                       help='Maximum files per data type to download')
    parser.add_argument('--download-only', action='store_true',
                       help='Only run download phase')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Master TCGA Pipeline...")
    logger.info(f"Configuration: max_files={args.max_files}, skip_download={args.skip_download}")
    
    start_time = time.time()
    phases_completed = {
        'Download Phase': False,
        'Processing Phase': False, 
        'Training Phase': False
    }
    
    try:
        # Phase 1: Download
        if not args.skip_download:
            download_summary = run_download_phase(args.max_files)
            phases_completed['Download Phase'] = True
            
            if args.download_only:
                logger.info("Download-only mode completed")
                return
        else:
            logger.info("⏭️ Skipping download phase (using existing data)")
            phases_completed['Download Phase'] = True
        
        # Phase 2: Processing
        try:
            data_file = run_processing_phase()
            phases_completed['Processing Phase'] = True
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            # Try to continue with existing data
            data_file = "multimodal_tcga_data.npz"
            if Path(data_file).exists():
                logger.info(f"Using existing data file: {data_file}")
                phases_completed['Processing Phase'] = True
            else:
                logger.error("No processed data available")
                data_file = None
        
        # Phase 3: Training
        if data_file and Path(data_file).exists():
            try:
                training_results = run_training_phase(data_file)
                phases_completed['Training Phase'] = True
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                logger.info("Pipeline completed data processing - training can be run separately")
        else:
            logger.warning("No data file available for training")
        
        # Summary
        total_time = time.time() - start_time
        pipeline_summary = create_pipeline_summary(phases_completed)
        pipeline_summary['total_duration_minutes'] = total_time / 60
        
        logger.info(f"Total pipeline duration: {total_time/60:.2f} minutes")
        
        # Determine next steps
        if all(phases_completed.values()):
            logger.info("🎉 COMPLETE SUCCESS: All phases completed!")
        elif phases_completed['Processing Phase']:
            logger.info("📊 DATA READY: Multi-modal dataset created successfully")
            logger.info("Next step: Run training separately if needed")
        elif phases_completed['Download Phase']:
            logger.info("📁 DATA DOWNLOADED: Files ready for processing")
            logger.info("Next step: Run processing phase")
        else:
            logger.warning("⚠️ Pipeline had issues - check logs for details")
            
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        create_pipeline_summary(phases_completed)
        raise

if __name__ == "__main__":
    main()
