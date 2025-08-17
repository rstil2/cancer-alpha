#!/usr/bin/env python3
"""
Test Real TCGA Data Integration
===============================

This script tests the enhanced TCGA data integration pipeline with real data access.

Author: Oncura Research Team
Date: July 28, 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from data_integration.tcga_data_processor import TCGADataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tcga_api_connection():
    """Test basic TCGA API connectivity"""
    logger.info("Testing TCGA API connection...")
    
    processor = TCGADataProcessor()
    
    # Test getting available projects
    projects = processor.get_available_projects()
    
    if projects:
        logger.info(f"✅ Successfully connected to TCGA API")
        logger.info(f"Found {len(projects)} relevant TCGA projects:")
        for project in projects[:5]:  # Show first 5
            logger.info(f"  - {project['project_id']}: {project['name']} ({project['cases_count']} cases)")
        return True, projects
    else:
        logger.error("❌ Failed to connect to TCGA API")
        return False, []

def test_case_queries(processor, projects):
    """Test querying cases for specific projects"""
    logger.info("Testing case queries...")
    
    if not projects:
        logger.warning("No projects available for case queries")
        return False
    
    # Test with first available project
    test_project = projects[0]['project_id']
    logger.info(f"Testing case query for project: {test_project}")
    
    cases = processor.query_cases_by_project(test_project, limit=10)
    
    if cases:
        logger.info(f"✅ Successfully retrieved {len(cases)} cases for {test_project}")
        # Show sample case information
        if cases:
            sample_case = cases[0]
            logger.info(f"Sample case ID: {sample_case.get('submitter_id', 'Unknown')}")
        return True
    else:
        logger.error(f"❌ Failed to retrieve cases for {test_project}")
        return False

def test_file_queries(processor, projects):
    """Test querying files by data type"""
    logger.info("Testing file queries...")
    
    if not projects:
        logger.warning("No projects available for file queries")
        return False, {}
    
    # Test with first 2 projects
    test_projects = [p['project_id'] for p in projects[:2]]
    logger.info(f"Testing file queries for projects: {test_projects}")
    
    data_types = [
        'Methylation Beta Value',
        'Masked Somatic Mutation',
        'Copy Number Segment',
        'Clinical Supplement'
    ]
    
    files_by_type = processor.query_files_by_data_type(test_projects, data_types, limit=5)
    
    total_files = sum(len(files) for files in files_by_type.values())
    
    if total_files > 0:
        logger.info(f"✅ Successfully found {total_files} files across data types:")
        for data_type, files in files_by_type.items():
            if files:
                logger.info(f"  - {data_type}: {len(files)} files")
                # Show sample file info
                sample_file = files[0]
                logger.info(f"    Sample file: {sample_file.get('file_name', 'Unknown')} ({sample_file.get('file_size', 0)} bytes)")
        return True, files_by_type
    else:
        logger.warning("⚠️ No files found for the specified data types")
        return False, {}

def test_synthetic_data_generation(processor):
    """Test synthetic data generation as fallback"""
    logger.info("Testing synthetic TCGA-like data generation...")
    
    try:
        X, y = processor.create_synthetic_tcga_like_data(500)
        
        logger.info(f"✅ Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        # Validate data quality
        quality_metrics = processor.validate_data_quality(X, y)
        logger.info(f"Data quality score: {quality_metrics['quality_score']:.2f}/10")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to generate synthetic data: {str(e)}")
        return False

def test_small_real_data_download(processor, files_by_type):
    """Test downloading a small amount of real data"""
    logger.info("Testing small real data download...")
    
    if not files_by_type or not any(files_by_type.values()):
        logger.warning("No files available for download test")
        return False
    
    # Find the smallest available file for testing
    smallest_file = None
    smallest_size = float('inf')
    
    for data_type, files in files_by_type.items():
        for file_info in files:
            file_size = file_info.get('file_size', 0)
            if 0 < file_size < smallest_size:
                smallest_size = file_size
                smallest_file = file_info
    
    if smallest_file:
        logger.info(f"Testing download of smallest file: {smallest_file.get('file_name', 'Unknown')} ({smallest_size} bytes)")
        
        try:
            # Download just one small file
            downloaded_files = processor.download_files_by_uuid([smallest_file['id']])
            
            if downloaded_files:
                logger.info(f"✅ Successfully downloaded {len(downloaded_files)} file(s)")
                
                # Test file extraction
                processed_data = processor.extract_and_process_files(downloaded_files)
                logger.info(f"Extracted data types: {list(processed_data.keys())}")
                
                return True
            else:
                logger.error("❌ Failed to download files")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during download test: {str(e)}")
            return False
    else:
        logger.warning("No suitable file found for download test")
        return False

def test_full_pipeline(processor, projects):
    """Test the full data processing pipeline"""
    logger.info("Testing full TCGA data processing pipeline...")
    
    if not projects:
        logger.warning("No projects available for full pipeline test")
        return False
    
    # Test with first 2 projects
    test_projects = [p['project_id'] for p in projects[:2]]
    
    try:
        # Test with real data attempt (will fallback to synthetic if needed)
        success = processor.process_real_tcga_data(
            project_ids=test_projects,
            output_file="test_tcga_data.npz",
            use_real_data=True,
            max_files_per_type=2  # Limit for testing
        )
        
        if success:
            logger.info("✅ Full pipeline completed successfully")
            
            # Verify output file
            if Path("test_tcga_data.npz").exists():
                data = np.load("test_tcga_data.npz", allow_pickle=True)
                logger.info(f"Output data shape: {data['features'].shape}")
                logger.info(f"Labels shape: {data['labels'].shape}")
                logger.info(f"Cancer types: {data['cancer_types']}")
                logger.info(f"Project IDs: {data['project_ids']}")
                return True
            else:
                logger.error("❌ Output file not created")
                return False
        else:
            logger.error("❌ Full pipeline failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in full pipeline test: {str(e)}")
        return False

def main():
    """Run comprehensive TCGA integration tests"""
    logger.info("🚀 Starting Real TCGA Data Integration Tests")
    logger.info("=" * 60)
    
    results = {
        'api_connection': False,
        'case_queries': False,
        'file_queries': False,
        'synthetic_data': False,
        'small_download': False,
        'full_pipeline': False
    }
    
    # Initialize processor
    processor = TCGADataProcessor()
    
    # Test 1: API Connection
    logger.info("\n📡 Test 1: TCGA API Connection")
    results['api_connection'], projects = test_tcga_api_connection()
    
    # Test 2: Case Queries
    logger.info("\n👥 Test 2: Case Queries")
    results['case_queries'] = test_case_queries(processor, projects)
    
    # Test 3: File Queries
    logger.info("\n📁 Test 3: File Queries")
    results['file_queries'], files_by_type = test_file_queries(processor, projects)
    
    # Test 4: Synthetic Data Generation
    logger.info("\n🧬 Test 4: Synthetic Data Generation")
    results['synthetic_data'] = test_synthetic_data_generation(processor)
    
    # Test 5: Small Real Data Download (if files available)
    logger.info("\n⬇️ Test 5: Small Real Data Download")
    if results['file_queries']:
        results['small_download'] = test_small_real_data_download(processor, files_by_type)
    else:
        logger.warning("Skipping download test - no files available")
    
    # Test 6: Full Pipeline
    logger.info("\n🔄 Test 6: Full Pipeline Test")
    results['full_pipeline'] = test_full_pipeline(processor, projects)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests >= 4:  # At least basic functionality working
        logger.info("🎉 TCGA integration setup is functional!")
        if passed_tests == total_tests:
            logger.info("🌟 All tests passed - Real data integration is ready!")
        else:
            logger.info("⚠️ Some advanced features may need attention")
    else:
        logger.warning("⚠️ TCGA integration needs troubleshooting")
    
    # Cleanup
    test_files = ["test_tcga_data.npz"]
    for test_file in test_files:
        if Path(test_file).exists():
            Path(test_file).unlink()
            logger.info(f"Cleaned up test file: {test_file}")

if __name__ == '__main__':
    main()
