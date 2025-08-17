#!/usr/bin/env python3
"""
Scalable TCGA Data Downloader
=============================

This script downloads and processes hundreds of TCGA samples across multiple
cancer types and data modalities for large-scale training.

Author: Oncura Research Team
Date: July 28, 2025
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import gzip
import tarfile
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScalableTCGADownloader:
    """Scalable TCGA data downloader and processor"""
    
    def __init__(self, cache_dir: str = "data_integration/tcga_large_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # TCGA API endpoints
        self.api_base = "https://api.gdc.cancer.gov"
        
        # Target cancer types for large-scale processing
        self.target_projects = {
            'TCGA-BRCA': {'name': 'Breast Invasive Carcinoma', 'target_samples': 100},
            'TCGA-LUAD': {'name': 'Lung Adenocarcinoma', 'target_samples': 100}, 
            'TCGA-COAD': {'name': 'Colon Adenocarcinoma', 'target_samples': 80},
            'TCGA-PRAD': {'name': 'Prostate Adenocarcinoma', 'target_samples': 80},
            'TCGA-STAD': {'name': 'Stomach Adenocarcinoma', 'target_samples': 60},
            'TCGA-KIRC': {'name': 'Kidney Renal Clear Cell Carcinoma', 'target_samples': 60},
            'TCGA-HNSC': {'name': 'Head and Neck Squamous Cell Carcinoma', 'target_samples': 60},
            'TCGA-LIHC': {'name': 'Liver Hepatocellular Carcinoma', 'target_samples': 60}
        }
        
        # Data types to download
        self.data_types = {
            'mutations': 'Masked Somatic Mutation',
            'methylation': 'Methylation Beta Value', 
            'expression': 'Gene Expression Quantification',
            'clinical': 'Clinical Supplement',
            'copy_number': 'Copy Number Segment'
        }
        
        # Cancer type mapping
        self.cancer_mapping = {
            'TCGA-BRCA': 0, 'TCGA-LUAD': 1, 'TCGA-COAD': 2, 'TCGA-PRAD': 3,
            'TCGA-STAD': 4, 'TCGA-KIRC': 5, 'TCGA-HNSC': 6, 'TCGA-LIHC': 7
        }

    def query_project_files(self, project_id: str, data_type: str, limit: int = 200) -> List[Dict]:
        """Query files for a specific project and data type"""
        
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [project_id]
                    }
                },
                {
                    "op": "in", 
                    "content": {
                        "field": "files.data_type",
                        "value": [data_type]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "format": "json",
            "size": str(limit),
            "expand": "cases.submitter_id"
        }
        
        try:
            response = requests.get(f"{self.api_base}/files", params=params)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('data', {}).get('hits', [])
            
            logger.info(f"Found {len(files)} {data_type} files for {project_id}")
            return files
            
        except Exception as e:
            logger.error(f"Error querying files for {project_id} {data_type}: {str(e)}")
            return []

    def download_file(self, file_info: Dict, data_type: str) -> Optional[Path]:
        """Download a single file"""
        
        file_id = file_info['id']
        file_name = file_info.get('file_name', f"{file_id}.dat")
        
        # Create data type specific directory
        data_dir = self.cache_dir / data_type
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / file_name
        
        # Skip if already downloaded
        if file_path.exists():
            return file_path
        
        try:
            # Download file
            response = requests.get(f"{self.api_base}/data/{file_id}", stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.debug(f"Downloaded {file_name}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading {file_name}: {str(e)}")
            return None

    def download_project_data(self, project_id: str, max_files_per_type: int = 50) -> Dict[str, List[Path]]:
        """Download data for a specific project"""
        
        logger.info(f"🔽 Downloading data for {project_id}...")
        project_files = defaultdict(list)
        
        for data_key, data_type in self.data_types.items():
            logger.info(f"  Querying {data_type} files...")
            
            files = self.query_project_files(project_id, data_type, limit=max_files_per_type)
            
            if files:
                logger.info(f"  Downloading {len(files)} {data_type} files...")
                
                # Download files with progress bar
                with tqdm(total=len(files), desc=f"{data_key}") as pbar:
                    for file_info in files:
                        file_path = self.download_file(file_info, data_key)
                        if file_path:
                            project_files[data_key].append(file_path)
                        pbar.update(1)
                        
                        # Rate limiting
                        time.sleep(0.1)
            
            logger.info(f"  ✅ Downloaded {len(project_files[data_key])} {data_type} files")
        
        return dict(project_files)

    def download_all_projects(self, max_files_per_type: int = 30) -> Dict[str, Dict[str, List[Path]]]:
        """Download data for all target projects"""
        
        logger.info("🚀 Starting large-scale TCGA data download...")
        logger.info(f"Target projects: {list(self.target_projects.keys())}")
        logger.info(f"Max files per type: {max_files_per_type}")
        
        all_project_data = {}
        
        for project_id, project_info in self.target_projects.items():
            logger.info(f"\n📂 Processing {project_id} - {project_info['name']}")
            
            project_files = self.download_project_data(project_id, max_files_per_type)
            all_project_data[project_id] = project_files
            
            # Summary
            total_files = sum(len(files) for files in project_files.values())
            logger.info(f"✅ {project_id}: {total_files} total files downloaded")
        
        # Overall summary
        total_projects = len(all_project_data)
        total_files_all = sum(
            sum(len(files) for files in project_files.values())
            for project_files in all_project_data.values()
        )
        
        logger.info(f"\n🎉 Large-scale download complete!")
        logger.info(f"  📊 Projects processed: {total_projects}")
        logger.info(f"  📁 Total files downloaded: {total_files_all}")
        
        return all_project_data

    def create_download_summary(self, all_project_data: Dict) -> Dict:
        """Create summary of downloaded data"""
        
        summary = {
            'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'projects': {},
            'totals': defaultdict(int)
        }
        
        for project_id, project_files in all_project_data.items():
            project_summary = {
                'name': self.target_projects[project_id]['name'],
                'files_by_type': {},
                'total_files': 0
            }
            
            for data_type, files in project_files.items():
                count = len(files)
                project_summary['files_by_type'][data_type] = count
                project_summary['total_files'] += count
                summary['totals'][data_type] += count
                summary['totals']['total_files'] += count
            
            summary['projects'][project_id] = project_summary
        
        summary['totals']['total_projects'] = len(all_project_data)
        
        return summary

    def save_download_summary(self, summary: Dict):
        """Save download summary to file"""
        
        summary_file = self.cache_dir / 'download_summary.json'
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Download summary saved to {summary_file}")
        
        # Print summary
        logger.info("\n📊 DOWNLOAD SUMMARY:")
        logger.info(f"  Total projects: {summary['totals']['total_projects']}")
        logger.info(f"  Total files: {summary['totals']['total_files']}")
        logger.info("  Files by type:")
        for data_type, count in summary['totals'].items():
            if data_type not in ['total_files', 'total_projects']:
                logger.info(f"    {data_type}: {count}")

def main():
    """Main function to run large-scale TCGA download"""
    
    logger.info("🚀 Starting Large-Scale TCGA Data Download Pipeline...")
    
    # Initialize downloader
    downloader = ScalableTCGADownloader()
    
    try:
        # Download all project data
        all_project_data = downloader.download_all_projects(max_files_per_type=20)
        
        # Create and save summary
        summary = downloader.create_download_summary(all_project_data)
        downloader.save_download_summary(summary)
        
        logger.info("✅ Large-scale TCGA download pipeline completed successfully!")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error in download pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
