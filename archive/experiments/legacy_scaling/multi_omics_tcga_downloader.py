#!/usr/bin/env python3
"""
Multi-Omics TCGA Downloader
===========================

Download complementary omics data types (clinical, expression, copy number, methylation, protein)
to build a comprehensive multi-omics cancer classification system.

This script focuses on collecting the missing omics data types to complement our existing
4,761 mutation files for a complete multi-omics analysis.

Key Features:
- Downloads clinical, expression, copy number, methylation, and protein data
- Matches samples with existing mutation data for integrated analysis
- Production-grade error handling and progress tracking
- Optimized for large-scale downloads
- Resume capability for interrupted downloads

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import requests
import json
import gzip
import os
import time
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_omics_tcga_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DownloadProgress:
    """Track download progress for each data type and project"""
    clinical: Dict[str, Set[str]] = field(default_factory=lambda: {})
    expression: Dict[str, Set[str]] = field(default_factory=lambda: {})
    copy_number: Dict[str, Set[str]] = field(default_factory=lambda: {})
    methylation: Dict[str, Set[str]] = field(default_factory=lambda: {})
    protein: Dict[str, Set[str]] = field(default_factory=lambda: {})

class MultiOmicsTCGADownloader:
    """Production-grade multi-omics TCGA data downloader"""
    
    def __init__(self, base_dir: str = "data/production_tcga"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # GDC API endpoints
        self.gdc_api_base = "https://api.gdc.cancer.gov"
        self.files_endpoint = f"{self.gdc_api_base}/files"
        self.data_endpoint = f"{self.gdc_api_base}/data"
        
        # Target TCGA projects (same as mutation data)
        self.target_projects = [
            'TCGA-BRCA', 'TCGA-COAD', 'TCGA-HNSC', 'TCGA-LGG', 'TCGA-LIHC',
            'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-PRAD', 'TCGA-STAD', 'TCGA-THCA'
        ]
        
        # Data type configurations for multi-omics
        self.data_type_configs = {
            'clinical': {
                'data_category': 'Clinical',
                'data_type': 'Clinical Supplement',
                'data_format': 'BCR XML',
                'directory': 'clinical'
            },
            'expression': {
                'data_category': 'Transcriptome Profiling',
                'data_type': 'Gene Expression Quantification',
                'data_format': 'TXT',
                'analysis_workflow_type': 'STAR - Counts',
                'directory': 'expression'
            },
            'copy_number': {
                'data_category': 'Copy Number Variation',
                'data_type': 'Copy Number Segment',
                'data_format': 'TXT',
                'directory': 'copy_number'
            },
            'methylation': {
                'data_category': 'DNA Methylation',
                'data_type': 'Methylation Beta Value',
                'data_format': 'TXT',
                'platform': 'Illumina Human Methylation 450',
                'directory': 'methylation'
            },
            'protein': {
                'data_category': 'Proteome Profiling',
                'data_type': 'Protein Expression Quantification',
                'data_format': 'TXT',
                'directory': 'protein'
            }
        }
        
        # Progress tracking
        self.progress_file = self.base_dir / "multi_omics_progress.pkl"
        self.progress = self.load_progress()
        self.lock = threading.Lock()
        
        # Rate limiting
        self.request_delay = 0.5  # Conservative 0.5 second delay
        self.max_workers = 3      # Conservative threading
        
        logger.info(f"🧬 Multi-Omics TCGA Downloader initialized")
        logger.info(f"📁 Base directory: {base_dir}")
        logger.info(f"🎯 Target projects: {len(self.target_projects)}")
        logger.info(f"🔬 Data types: {list(self.data_type_configs.keys())}")
    
    def load_progress(self) -> DownloadProgress:
        """Load previous download progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
        return DownloadProgress()
    
    def save_progress(self):
        """Save current download progress"""
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(self.progress, f)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def query_gdc_files(self, project: str, data_type: str, config: Dict) -> List[Dict]:
        """Query GDC API for files matching criteria"""
        try:
            # Build query filters
            filters = {
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "cases.project.project_id",
                            "value": [project]
                        }
                    },
                    {
                        "op": "in",
                        "content": {
                            "field": "data_category",
                            "value": [config['data_category']]
                        }
                    },
                    {
                        "op": "in", 
                        "content": {
                            "field": "data_type",
                            "value": [config['data_type']]
                        }
                    }
                ]
            }
            
            # Add additional filters if specified
            if 'data_format' in config:
                filters['content'].append({
                    "op": "in",
                    "content": {
                        "field": "data_format", 
                        "value": [config['data_format']]
                    }
                })
            
            if 'analysis_workflow_type' in config:
                filters['content'].append({
                    "op": "in",
                    "content": {
                        "field": "analysis.workflow_type",
                        "value": [config['analysis_workflow_type']]
                    }
                })
            
            if 'platform' in config:
                filters['content'].append({
                    "op": "in",
                    "content": {
                        "field": "platform",
                        "value": [config['platform']]
                    }
                })
            
            # Query parameters
            params = {
                "filters": json.dumps(filters),
                "fields": "id,file_name,file_size,data_category,data_type,cases.submitter_id",
                "format": "json",
                "size": "2000"  # Get up to 2000 files per request
            }
            
            response = requests.get(self.files_endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('data', [])
            
            logger.info(f"📊 Found {len(files)} {data_type} files for {project}")
            return files
            
        except Exception as e:
            logger.error(f"❌ Failed to query {data_type} files for {project}: {e}")
            return []
    
    def download_file(self, file_info: Dict, output_dir: Path, data_type: str) -> bool:
        """Download a single file from GDC"""
        file_id = file_info['id']
        file_name = file_info['file_name']
        output_path = output_dir / file_name
        
        # Skip if already downloaded
        if output_path.exists():
            logger.debug(f"⏭️ Skipping existing {file_name}")
            return True
        
        try:
            # Download file
            response = requests.get(f"{self.data_endpoint}/{file_id}", timeout=60)
            response.raise_for_status()
            
            # Save file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Verify file was written correctly
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.debug(f"✅ Downloaded {file_name} ({file_info.get('file_size', 0)} bytes)")
                return True
            else:
                logger.warning(f"⚠️ Downloaded file {file_name} appears to be empty")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to download {file_name}: {e}")
            if output_path.exists():
                output_path.unlink()  # Clean up partial download
            return False
    
    def download_data_type(self, data_type: str, config: Dict):
        """Download all files for a specific data type across all projects"""
        logger.info(f"🔥 Starting {data_type.upper()} download...")
        
        # Create output directory
        output_dir = self.base_dir / config['directory']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_files_downloaded = 0
        total_files_found = 0
        
        for project in self.target_projects:
            logger.info(f"🧬 Processing {data_type} for {project}...")
            
            # Create project subdirectory
            project_dir = output_dir / project
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Query for files
            files = self.query_gdc_files(project, data_type, config)
            if not files:
                logger.warning(f"⚠️ No {data_type} files found for {project}")
                continue
            
            total_files_found += len(files)
            
            # Get already downloaded files for this project
            progress_dict = getattr(self.progress, data_type)
            if project not in progress_dict:
                progress_dict[project] = set()
            
            # Filter out already downloaded files
            new_files = []
            for file_info in files:
                file_name = file_info['file_name']
                if file_name not in progress_dict[project]:
                    new_files.append(file_info)
            
            logger.info(f"📥 Downloading {len(new_files)} new {data_type} files for {project} (found {len(files)} total)")
            
            if not new_files:
                logger.info(f"✅ All {data_type} files already downloaded for {project}")
                continue
            
            # Download files with threading
            download_success_count = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit download tasks
                future_to_file = {
                    executor.submit(self.download_file, file_info, project_dir, data_type): file_info
                    for file_info in new_files
                }
                
                # Process completed downloads
                for future in as_completed(future_to_file):
                    file_info = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            download_success_count += 1
                            total_files_downloaded += 1
                            
                            # Update progress
                            with self.lock:
                                progress_dict[project].add(file_info['file_name'])
                                self.save_progress()
                        
                        # Rate limiting
                        time.sleep(self.request_delay)
                        
                    except Exception as e:
                        logger.error(f"❌ Download task failed for {file_info['file_name']}: {e}")
            
            logger.info(f"✅ {project} {data_type}: {download_success_count}/{len(new_files)} files downloaded")
        
        logger.info(f"🎉 {data_type.upper()} COMPLETE: {total_files_downloaded}/{total_files_found} files downloaded")
        return total_files_downloaded
    
    def get_current_status(self) -> Dict:
        """Get current download status for all data types"""
        status = {}
        
        for data_type, config in self.data_type_configs.items():
            data_dir = self.base_dir / config['directory'] 
            
            # Count actual files on disk
            file_count = 0
            if data_dir.exists():
                for project_dir in data_dir.iterdir():
                    if project_dir.is_dir() and project_dir.name.startswith('TCGA-'):
                        project_files = list(project_dir.glob("*"))
                        file_count += len([f for f in project_files if f.is_file()])
            
            # Count progress tracking
            progress_dict = getattr(self.progress, data_type)
            progress_count = sum(len(files) for files in progress_dict.values())
            
            status[data_type] = {
                'files_on_disk': file_count,
                'progress_tracked': progress_count,
                'projects_started': len(progress_dict)
            }
        
        return status
    
    def run_multi_omics_download(self):
        """Execute complete multi-omics download"""
        logger.info("🚀 Starting Multi-Omics TCGA Download...")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Show initial status
        logger.info("📊 Initial status:")
        initial_status = self.get_current_status()
        for data_type, status in initial_status.items():
            logger.info(f"  {data_type}: {status['files_on_disk']} files on disk")
        
        # Download each data type sequentially for stability
        download_results = {}
        
        for data_type, config in self.data_type_configs.items():
            logger.info(f"\n🔬 === {data_type.upper()} DOWNLOAD ===")
            try:
                files_downloaded = self.download_data_type(data_type, config)
                download_results[data_type] = files_downloaded
                
                # Brief pause between data types
                logger.info(f"⏸️ Pausing 30 seconds before next data type...")
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Failed to download {data_type}: {e}")
                download_results[data_type] = 0
        
        # Final report
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info("\n🎉 MULTI-OMICS DOWNLOAD COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"⏱️ Total time: {total_time}")
        
        final_status = self.get_current_status()
        total_new_files = sum(download_results.values())
        
        logger.info("📊 Final Results:")
        for data_type, count in download_results.items():
            final_count = final_status[data_type]['files_on_disk']
            logger.info(f"  {data_type}: +{count} new files (total: {final_count})")
        
        logger.info(f"🎯 Total new files downloaded: {total_new_files}")
        
        # Show comprehensive summary
        logger.info("\n🧬 Complete Multi-Omics Dataset Status:")
        for data_type, status in final_status.items():
            logger.info(f"  {data_type}: {status['files_on_disk']} files across {status['projects_started']} projects")
        
        return download_results


def main():
    """Main execution function"""
    logger.info("🧬 Multi-Omics TCGA Downloader")
    logger.info("=" * 60)
    
    # Initialize downloader
    downloader = MultiOmicsTCGADownloader()
    
    try:
        # Run complete download
        results = downloader.run_multi_omics_download()
        
        logger.info("✅ SUCCESS: Multi-omics download completed!")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("⏸️ Download interrupted by user")
        return None
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise


if __name__ == "__main__":
    results = main()
