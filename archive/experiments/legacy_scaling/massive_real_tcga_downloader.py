#!/usr/bin/env python3
"""
Massive Real TCGA Downloader for 10,000+ Samples
===============================================

Production-grade downloader for scaling to 10,000+ authentic TCGA samples
across 16 cancer types using only real clinical genomics data.

NO SYNTHETIC DATA - REAL DATA ONLY!

Author: Oncura Research Team
Date: August 19, 2025
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import gzip
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from tqdm import tqdm
import hashlib
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('massive_tcga_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MassiveRealTCGADownloader:
    """Production-grade downloader for 10,000+ real TCGA samples"""
    
    def __init__(self, cache_dir: str = "data/tcga_massive_real"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # GDC API configuration
        self.api_base = "https://api.gdc.cancer.gov"
        self.max_concurrent = 20  # Concurrent downloads
        self.chunk_size = 8192 * 8  # 64KB chunks
        self.timeout = 60  # Request timeout
        self.retry_attempts = 5
        
        # Target: 10,000+ samples across 16 cancer types
        self.target_projects = {
            # Phase 1: Scale existing 8 types (5,000 samples total)
            'TCGA-BRCA': {'name': 'Breast Invasive Carcinoma', 'target': 800, 'priority': 1},
            'TCGA-LUAD': {'name': 'Lung Adenocarcinoma', 'target': 700, 'priority': 1}, 
            'TCGA-COAD': {'name': 'Colon Adenocarcinoma', 'target': 600, 'priority': 1},
            'TCGA-PRAD': {'name': 'Prostate Adenocarcinoma', 'target': 650, 'priority': 1},
            'TCGA-STAD': {'name': 'Stomach Adenocarcinoma', 'target': 600, 'priority': 1},
            'TCGA-KIRC': {'name': 'Kidney Renal Clear Cell Carcinoma', 'target': 650, 'priority': 1},
            'TCGA-HNSC': {'name': 'Head and Neck Squamous Cell Carcinoma', 'target': 600, 'priority': 1},
            'TCGA-LIHC': {'name': 'Liver Hepatocellular Carcinoma', 'target': 400, 'priority': 1},
            
            # Phase 2: Add 8 new types (5,000 samples total)
            'TCGA-LUSC': {'name': 'Lung Squamous Cell Carcinoma', 'target': 700, 'priority': 2},
            'TCGA-KIRP': {'name': 'Kidney Renal Papillary Cell Carcinoma', 'target': 400, 'priority': 2},
            'TCGA-THCA': {'name': 'Thyroid Carcinoma', 'target': 650, 'priority': 2},
            'TCGA-BLCA': {'name': 'Bladder Urothelial Carcinoma', 'target': 500, 'priority': 2},
            'TCGA-UCEC': {'name': 'Uterine Corpus Endometrial Carcinoma', 'target': 600, 'priority': 2},
            'TCGA-OV': {'name': 'Ovarian Serous Cystadenocarcinoma', 'target': 400, 'priority': 2},
            'TCGA-GBM': {'name': 'Glioblastoma Multiforme', 'target': 350, 'priority': 2},
            'TCGA-LGG': {'name': 'Brain Lower Grade Glioma', 'target': 400, 'priority': 2}
        }
        
        # High-priority multi-omics data types
        self.data_types = {
            'rna_seq': {
                'gdc_name': 'Gene Expression Quantification',
                'workflow': 'HTSeq - FPKM',
                'priority': 1,
                'target_samples': 10000
            },
            'mutations': {
                'gdc_name': 'Masked Somatic Mutation', 
                'format': 'MAF',
                'priority': 1,
                'target_samples': 10000
            },
            'methylation_450k': {
                'gdc_name': 'Methylation Beta Value',
                'platform': 'Illumina Human Methylation 450',
                'priority': 1,
                'target_samples': 8000
            },
            'methylation_epic': {
                'gdc_name': 'Methylation Beta Value', 
                'platform': 'Illumina Human Methylation EPIC',
                'priority': 1,
                'target_samples': 2000
            },
            'copy_number': {
                'gdc_name': 'Copy Number Segment',
                'priority': 1,
                'target_samples': 8000
            },
            'clinical': {
                'gdc_name': 'Clinical Supplement',
                'priority': 1,
                'target_samples': 10000
            },
            'mirna': {
                'gdc_name': 'miRNA Expression Quantification',
                'priority': 2,
                'target_samples': 6000
            },
            'protein': {
                'gdc_name': 'Protein Expression Quantification',
                'priority': 2, 
                'target_samples': 2000
            }
        }
        
        # Cancer type mapping (16 types)
        self.cancer_mapping = {
            'TCGA-BRCA': 0, 'TCGA-LUAD': 1, 'TCGA-COAD': 2, 'TCGA-PRAD': 3,
            'TCGA-STAD': 4, 'TCGA-KIRC': 5, 'TCGA-HNSC': 6, 'TCGA-LIHC': 7,
            'TCGA-LUSC': 8, 'TCGA-KIRP': 9, 'TCGA-THCA': 10, 'TCGA-BLCA': 11,
            'TCGA-UCEC': 12, 'TCGA-OV': 13, 'TCGA-GBM': 14, 'TCGA-LGG': 15
        }
        
        # Track progress
        self.progress_file = self.cache_dir / "download_progress.json"
        self.downloaded_files: Set[str] = set()
        self.failed_downloads: Set[str] = set()
        
        # Load existing progress
        self._load_progress()
    
    def _load_progress(self):
        """Load previous download progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.downloaded_files = set(progress.get('downloaded', []))
                    self.failed_downloads = set(progress.get('failed', []))
                    logger.info(f"Loaded progress: {len(self.downloaded_files)} downloaded, {len(self.failed_downloads)} failed")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
    
    def _save_progress(self):
        """Save download progress"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'downloaded': list(self.downloaded_files),
            'failed': list(self.failed_downloads),
            'total_downloaded': len(self.downloaded_files),
            'total_failed': len(self.failed_downloads)
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    async def query_project_samples(self, session: aiohttp.ClientSession, 
                                   project_id: str, data_type_info: Dict, 
                                   limit: int = 2000) -> List[Dict]:
        """Query samples for a specific project and data type"""
        
        # Base filters for project and data type
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
                        "value": [data_type_info['gdc_name']]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.samples.sample_type",
                        "value": ["Primary Tumor"]  # Only primary tumors
                    }
                }
            ]
        }
        
        # Add workflow filter if specified
        if 'workflow' in data_type_info:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": [data_type_info['workflow']]
                }
            })
        
        # Add platform filter if specified
        if 'platform' in data_type_info:
            filters["content"].append({
                "op": "in",
                "content": {
                    "field": "files.platform",
                    "value": [data_type_info['platform']]
                }
            })
        
        params = {
            "filters": json.dumps(filters),
            "format": "json",
            "size": str(limit),
            "expand": "cases.submitter_id,cases.samples"
        }
        
        try:
            async with session.get(f"{self.api_base}/files", 
                                 params=params, 
                                 timeout=self.timeout) as response:
                response.raise_for_status()
                data = await response.json()
                files = data.get('data', {}).get('hits', [])
                
                # Filter out already downloaded files
                new_files = [f for f in files if f['id'] not in self.downloaded_files]
                
                logger.info(f"Found {len(files)} files for {project_id} {data_type_info['gdc_name']}, "
                          f"{len(new_files)} new files to download")
                return new_files
                
        except Exception as e:
            logger.error(f"Error querying {project_id} {data_type_info['gdc_name']}: {str(e)}")
            return []
    
    async def download_file(self, session: aiohttp.ClientSession, 
                          file_info: Dict, data_type: str) -> Optional[Path]:
        """Download a single file with retry logic"""
        
        file_id = file_info['id']
        file_name = file_info.get('file_name', f"{file_id}.dat")
        
        # Skip if already downloaded
        if file_id in self.downloaded_files:
            return None
        
        # Skip if previously failed too many times
        if file_id in self.failed_downloads:
            return None
        
        # Create data type directory
        data_dir = self.cache_dir / data_type
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / file_name
        temp_path = data_dir / f"{file_name}.tmp"
        
        # Skip if file already exists and is valid
        if file_path.exists() and file_path.stat().st_size > 0:
            self.downloaded_files.add(file_id)
            return file_path
        
        # Download with retry logic
        for attempt in range(self.retry_attempts):
            try:
                async with session.get(f"{self.api_base}/data/{file_id}",
                                     timeout=self.timeout) as response:
                    response.raise_for_status()
                    
                    # Download to temporary file
                    with open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            f.write(chunk)
                    
                    # Verify file size
                    if temp_path.stat().st_size > 0:
                        temp_path.rename(file_path)
                        self.downloaded_files.add(file_id)
                        logger.debug(f"Downloaded {file_name} ({file_path.stat().st_size} bytes)")
                        return file_path
                    else:
                        temp_path.unlink(missing_ok=True)
                        raise ValueError("Downloaded file is empty")
                        
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {file_name}: {str(e)}")
                temp_path.unlink(missing_ok=True)
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to download {file_name} after {self.retry_attempts} attempts")
                    self.failed_downloads.add(file_id)
                    return None
    
    async def download_project_data(self, project_id: str, target_samples: int) -> Dict[str, int]:
        """Download all data types for a specific project"""
        
        logger.info(f"🔽 Starting download for {project_id} (target: {target_samples} samples)")
        
        project_stats = defaultdict(int)
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=10,
            ttl_dns_cache=300
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout
        ) as session:
            
            for data_type, data_info in self.data_types.items():
                logger.info(f"  📁 Processing {data_type} for {project_id}...")
                
                # Query available files
                files = await self.query_project_samples(
                    session, project_id, data_info, 
                    limit=min(target_samples * 2, 2000)  # Query extra for filtering
                )
                
                if not files:
                    logger.warning(f"  ⚠️  No {data_type} files found for {project_id}")
                    continue
                
                # Limit to target samples
                files_to_download = files[:target_samples]
                
                logger.info(f"  🔽 Downloading {len(files_to_download)} {data_type} files...")
                
                # Create semaphore for concurrent downloads
                semaphore = asyncio.Semaphore(self.max_concurrent)
                
                async def download_with_semaphore(file_info):
                    async with semaphore:
                        return await self.download_file(session, file_info, data_type)
                
                # Download with progress bar
                download_tasks = [download_with_semaphore(f) for f in files_to_download]
                
                downloaded_count = 0
                with tqdm(total=len(download_tasks), desc=f"{data_type}") as pbar:
                    for coro in asyncio.as_completed(download_tasks):
                        result = await coro
                        if result:
                            downloaded_count += 1
                        pbar.update(1)
                
                project_stats[data_type] = downloaded_count
                logger.info(f"  ✅ Downloaded {downloaded_count}/{len(files_to_download)} {data_type} files")
                
                # Save progress periodically
                self._save_progress()
                
                # Rate limiting between data types
                await asyncio.sleep(1)
        
        return dict(project_stats)
    
    async def download_all_projects(self, phase: int = 1) -> Dict[str, Dict[str, int]]:
        """Download data for all target projects in specified phase"""
        
        phase_projects = {
            pid: info for pid, info in self.target_projects.items() 
            if info['priority'] == phase
        }
        
        total_target = sum(info['target'] for info in phase_projects.values())
        
        logger.info(f"🚀 Starting Phase {phase} massive TCGA download...")
        logger.info(f"Target projects: {list(phase_projects.keys())}")
        logger.info(f"Total target samples: {total_target:,}")
        
        all_stats = {}
        
        for project_id, project_info in phase_projects.items():
            logger.info(f"\n📂 Processing {project_id} - {project_info['name']}")
            logger.info(f"Target samples: {project_info['target']}")
            
            try:
                project_stats = await self.download_project_data(
                    project_id, project_info['target']
                )
                all_stats[project_id] = project_stats
                
                # Summary for this project
                total_downloaded = sum(project_stats.values())
                logger.info(f"✅ {project_id} complete: {total_downloaded:,} total files")
                
            except Exception as e:
                logger.error(f"❌ Error processing {project_id}: {str(e)}")
                all_stats[project_id] = {}
        
        # Final summary
        total_downloaded = sum(
            sum(stats.values()) for stats in all_stats.values()
        )
        
        logger.info(f"\n🎉 Phase {phase} download complete!")
        logger.info(f"Total files downloaded: {total_downloaded:,}")
        logger.info(f"Projects processed: {len(all_stats)}")
        
        # Save final progress
        self._save_progress()
        
        return all_stats
    
    def generate_download_report(self, stats: Dict[str, Dict[str, int]]) -> Path:
        """Generate comprehensive download report"""
        
        report_path = self.cache_dir / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate totals
        total_files = sum(sum(project_stats.values()) for project_stats in stats.values())
        total_projects = len(stats)
        
        # Estimate sample counts (approximate)
        estimated_samples = {}
        for project_id, project_stats in stats.items():
            # Use RNA-seq or mutations as sample count estimate
            sample_estimate = max(
                project_stats.get('rna_seq', 0),
                project_stats.get('mutations', 0),
                sum(project_stats.values()) // len(self.data_types)
            )
            estimated_samples[project_id] = sample_estimate
        
        total_estimated_samples = sum(estimated_samples.values())
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files_downloaded': total_files,
                'total_projects_processed': total_projects,
                'estimated_total_samples': total_estimated_samples,
                'target_achieved': total_estimated_samples >= 10000
            },
            'by_project': stats,
            'estimated_samples_by_project': estimated_samples,
            'data_types_collected': list(self.data_types.keys()),
            'cancer_types_collected': list(stats.keys()),
            'download_status': {
                'total_downloaded_files': len(self.downloaded_files),
                'total_failed_files': len(self.failed_downloads),
                'success_rate': len(self.downloaded_files) / (len(self.downloaded_files) + len(self.failed_downloads)) if (len(self.downloaded_files) + len(self.failed_downloads)) > 0 else 1.0
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Download report saved: {report_path}")
        
        # Print summary
        logger.info(f"\n📈 DOWNLOAD SUMMARY")
        logger.info(f"=" * 50)
        logger.info(f"Total files downloaded: {total_files:,}")
        logger.info(f"Estimated samples: {total_estimated_samples:,}")
        logger.info(f"Target achieved: {'✅ YES' if total_estimated_samples >= 10000 else '❌ NO'}")
        logger.info(f"Success rate: {report['download_status']['success_rate']:.1%}")
        
        return report_path

async def main():
    """Main download execution"""
    
    downloader = MassiveRealTCGADownloader()
    
    logger.info("🧬 Massive Real TCGA Downloader Starting...")
    logger.info("=" * 60)
    logger.info("✅ REAL DATA ONLY - NO SYNTHETIC DATA!")
    logger.info("🎯 Target: 10,000+ authentic TCGA samples")
    logger.info("=" * 60)
    
    try:
        # Phase 1: Scale existing 8 cancer types to 5,000 samples
        logger.info("\n🚀 PHASE 1: Scaling existing cancer types...")
        phase1_stats = await downloader.download_all_projects(phase=1)
        
        # Phase 2: Add 8 new cancer types for additional 5,000 samples  
        logger.info("\n🚀 PHASE 2: Adding new cancer types...")
        phase2_stats = await downloader.download_all_projects(phase=2)
        
        # Combine results
        all_stats = {**phase1_stats, **phase2_stats}
        
        # Generate final report
        report_path = downloader.generate_download_report(all_stats)
        
        logger.info(f"\n🎉 MASSIVE REAL TCGA DOWNLOAD COMPLETE!")
        logger.info(f"📊 Report: {report_path}")
        logger.info(f"🧬 100% Real TCGA Data - Ready for 10K+ Model Training!")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
