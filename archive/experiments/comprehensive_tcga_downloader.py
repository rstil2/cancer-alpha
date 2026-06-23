#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE TCGA DOWNLOADER - ALL 33 CANCER TYPES 🚀
========================================================

Extends the production downloader to target ALL available TCGA cancer types
for maximum dataset coverage toward 50,000+ real samples.

Features:
- All 33 TCGA cancer project codes
- Conservative rate limiting for API stability
- Resume capability from existing progress
- Prioritizes high-yield cancer types first
- 100% real TCGA data - ZERO synthetic contamination

Author: Cancer Alpha AI
Date: 2025-01-19
"""

import requests
import json
import gzip
import os
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import signal
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DownloadProgress:
    """Track download progress across sessions"""
    project: str
    data_type: str
    total_files: int
    downloaded_files: int
    failed_files: int
    start_time: datetime
    last_update: datetime

class RateLimitedSession:
    """Thread-safe rate-limited HTTP session"""
    
    def __init__(self, requests_per_second: float = 2.0, max_retries: int = 3):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = threading.Lock()
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'Cancer-Alpha-Comprehensive-Downloader/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Rate-limited GET request"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            return self.session.get(url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Rate-limited POST request"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            return self.session.post(url, **kwargs)

class ComprehensiveTCGADownloader:
    """Comprehensive TCGA downloader targeting all 33 cancer types"""
    
    def __init__(self, base_dir: str = "data/production_tcga", max_workers: int = 3):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # GDC API endpoints
        self.gdc_base = "https://api.gdc.cancer.gov"
        self.files_endpoint = f"{self.gdc_base}/files"
        self.data_endpoint = f"{self.gdc_base}/data"
        
        # Rate-limited session (2 requests per second max)
        self.session = RateLimitedSession(requests_per_second=1.5)
        
        # Progress tracking
        self.progress_file = self.base_dir / "download_progress.pkl"
        self.progress: Dict[str, DownloadProgress] = {}
        self.load_progress()
        
        # Data types to download
        self.data_types = {
            "mutations": "Masked Somatic Mutation",
            "clinical": "Clinical Supplement", 
            "expression": "Gene Expression Quantification",
            "methylation": "Methylation Beta Value",
            "copy_number": "Copy Number Segment",
            "protein": "Protein Expression Quantification"
        }
        
        # ALL 33 TCGA projects - prioritized by expected yield
        self.tcga_projects = [
            # Already downloaded (high-yield)
            "TCGA-BRCA",  # Breast - 1098 cases
            "TCGA-LUAD",  # Lung Adenocarcinoma - 585 cases
            "TCGA-LUSC",  # Lung Squamous - 504 cases
            "TCGA-COAD",  # Colon - 512 cases
            "TCGA-PRAD",  # Prostate - 550 cases
            "TCGA-THCA",  # Thyroid - 572 cases
            "TCGA-HNSC",  # Head and Neck - 566 cases
            "TCGA-LGG",   # Low Grade Glioma - 534 cases
            "TCGA-LIHC",  # Liver - 424 cases
            "TCGA-STAD",  # Stomach - 478 cases
            
            # Medium-yield projects
            "TCGA-UCEC",  # Endometrial - 548 cases
            "TCGA-KIRC",  # Kidney Clear Cell - 537 cases
            "TCGA-BLCA",  # Bladder - 433 cases
            "TCGA-OV",    # Ovarian - 379 cases
            "TCGA-GBM",   # Glioblastoma - 617 cases
            "TCGA-KIRP",  # Kidney Papillary - 321 cases
            "TCGA-CESC",  # Cervical - 307 cases
            "TCGA-SARC",  # Sarcoma - 261 cases
            "TCGA-PAAD",  # Pancreatic - 185 cases
            "TCGA-PCPG",  # Pheochromocytoma - 179 cases
            
            # Lower-yield but complete coverage projects
            "TCGA-READ",  # Rectum - 172 cases
            "TCGA-TGCT",  # Testicular - 156 cases
            "TCGA-ESCA",  # Esophageal - 185 cases
            "TCGA-THYM",  # Thymoma - 124 cases
            "TCGA-MESO",  # Mesothelioma - 87 cases
            "TCGA-UCS",   # Uterine Carcinosarcoma - 57 cases
            "TCGA-ACC",   # Adrenocortical - 92 cases
            "TCGA-UVM",   # Uveal Melanoma - 80 cases
            "TCGA-DLBC",  # Lymphoma - 58 cases
            "TCGA-KICH",  # Kidney Chromophobe - 113 cases
            "TCGA-SKCM",  # Skin Melanoma - 472 cases
            "TCGA-CHOL",  # Cholangiocarcinoma - 45 cases
            "TCGA-LAML"   # Acute Myeloid Leukemia - 200 cases
        ]
        
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Comprehensive TCGA Downloader initialized")
        logger.info(f"📂 Base directory: {self.base_dir}")
        logger.info(f"🧵 Max workers: {self.max_workers}")
        logger.info(f"⚡ Rate limit: 1.5 requests/second")
        logger.info(f"🎯 Target projects: {len(self.tcga_projects)} (ALL TCGA)")
        logger.info(f"📊 Data types: {len(self.data_types)}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Shutdown signal received ({signum})")
        self.shutdown_requested = True
        self.save_progress()
    
    def load_progress(self):
        """Load progress from disk"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    self.progress = pickle.load(f)
                logger.info(f"📈 Loaded progress for {len(self.progress)} downloads")
            except Exception as e:
                logger.warning(f"Failed to load progress: {e}")
                self.progress = {}
    
    def save_progress(self):
        """Save progress to disk"""
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(self.progress, f)
            logger.info("💾 Progress saved")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def get_project_files(self, project: str, data_type: str, max_files: int = 2000) -> List[Dict]:
        """Get files for a specific project and data type"""
        logger.info(f"🔍 Querying files: {project} - {data_type}")
        
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
                {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
                {"op": "in", "content": {"field": "data_format", "value": ["MAF", "TXT", "TSV"]}},
                {"op": "in", "content": {"field": "access", "value": ["open"]}}
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "format": "json",
            "size": str(max_files),
            "fields": "id,file_name,file_size,data_type,cases.project.project_id"
        }
        
        try:
            response = self.session.get(self.files_endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            files = data.get("data", {}).get("hits", [])
            logger.info(f"✅ Found {len(files)} files for {project} - {data_type}")
            
            return files
            
        except Exception as e:
            logger.error(f"❌ Error querying files for {project} - {data_type}: {e}")
            return []
    
    def download_file(self, file_info: Dict, output_dir: Path) -> bool:
        """Download a single file with proper error handling"""
        file_id = file_info["id"]
        file_name = file_info["file_name"]
        file_path = output_dir / file_name
        
        # Skip if already exists and has correct size
        if file_path.exists():
            expected_size = file_info.get("file_size")
            if expected_size and file_path.stat().st_size == expected_size:
                return True
        
        try:
            logger.info(f"⬇️ Downloading: {file_name}")
            
            response = self.session.get(f"{self.data_endpoint}/{file_id}", 
                                      stream=True, timeout=60)
            response.raise_for_status()
            
            # Write file in chunks
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✅ Downloaded: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {file_name}: {e}")
            # Clean up partial file
            if file_path.exists():
                file_path.unlink()
            return False
    
    def download_project_data(self, project: str, data_type_key: str, max_files: int = 2000):
        """Download data for a specific project and data type"""
        data_type = self.data_types[data_type_key]
        progress_key = f"{project}_{data_type_key}"
        
        # Skip if already completed
        if progress_key in self.progress:
            p = self.progress[progress_key]
            if p.downloaded_files > 0 and p.failed_files == 0:
                logger.info(f"⏭️ Skipping {project} - {data_type_key} (already complete)")
                return
        
        # Create output directory
        output_dir = self.base_dir / data_type_key / project
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get files list
        files = self.get_project_files(project, data_type, max_files)
        if not files:
            logger.warning(f"⚠️ No files found for {project} - {data_type}")
            return
        
        # Initialize progress
        if progress_key not in self.progress:
            self.progress[progress_key] = DownloadProgress(
                project=project,
                data_type=data_type_key,
                total_files=len(files),
                downloaded_files=0,
                failed_files=0,
                start_time=datetime.now(),
                last_update=datetime.now()
            )
        
        progress = self.progress[progress_key]
        logger.info(f"📊 {project} - {data_type_key}: {progress.downloaded_files}/{progress.total_files} complete")
        
        # Download files with conservative concurrency
        downloaded = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=2) as executor:  # Very conservative
            future_to_file = {}
            
            for file_info in files:
                if self.shutdown_requested:
                    break
                    
                future = executor.submit(self.download_file, file_info, output_dir)
                future_to_file[future] = file_info
            
            for future in as_completed(future_to_file):
                if self.shutdown_requested:
                    break
                    
                file_info = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        downloaded += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"❌ Download task failed: {e}")
                    failed += 1
                
                # Update progress periodically
                if (downloaded + failed) % 10 == 0:
                    progress.downloaded_files = downloaded
                    progress.failed_files = failed
                    progress.last_update = datetime.now()
                    self.save_progress()
        
        # Final progress update
        progress.downloaded_files = downloaded
        progress.failed_files = failed
        progress.last_update = datetime.now()
        self.save_progress()
        
        logger.info(f"🎯 {project} - {data_type_key} complete: {downloaded} downloaded, {failed} failed")
    
    def download_all_data(self):
        """Download all data with proper prioritization"""
        logger.info("🚀 Starting comprehensive TCGA download - ALL 33 CANCER TYPES")
        logger.info("🌟 100% REAL DATA - ZERO SYNTHETIC CONTAMINATION")
        
        total_combinations = len(self.tcga_projects) * len(self.data_types)
        completed = 0
        
        # Process each project-datatype combination
        for project in self.tcga_projects:
            if self.shutdown_requested:
                break
                
            logger.info(f"\\n🔥 Processing project: {project}")
            
            # Start with most important data types
            priority_order = ["mutations", "clinical", "expression", "methylation", "copy_number", "protein"]
            
            for data_type_key in priority_order:
                if self.shutdown_requested:
                    break
                
                if data_type_key in self.data_types:
                    logger.info(f"📁 Downloading {data_type_key} data...")
                    self.download_project_data(project, data_type_key)
                    
                    completed += 1
                    progress_pct = (completed / total_combinations) * 100
                    logger.info(f"📈 Overall progress: {completed}/{total_combinations} ({progress_pct:.1f}%)")
                    
                    # Brief pause between data types
                    if not self.shutdown_requested:
                        time.sleep(2)
        
        if not self.shutdown_requested:
            logger.info("🎉 Comprehensive download completed successfully!")
        else:
            logger.info("🛑 Download stopped by user request")
        
        self.save_progress()
    
    def print_summary(self):
        """Print download summary"""
        logger.info("\\n" + "="*60)
        logger.info("📊 COMPREHENSIVE DOWNLOAD SUMMARY")
        logger.info("="*60)
        
        total_files = 0
        total_downloaded = 0
        total_failed = 0
        
        for key, progress in self.progress.items():
            logger.info(f"{progress.project} - {progress.data_type}:")
            logger.info(f"  📁 Total: {progress.total_files}")
            logger.info(f"  ✅ Downloaded: {progress.downloaded_files}")
            logger.info(f"  ❌ Failed: {progress.failed_files}")
            logger.info(f"  ⏱️ Duration: {progress.last_update - progress.start_time}")
            
            total_files += progress.total_files
            total_downloaded += progress.downloaded_files
            total_failed += progress.failed_files
        
        logger.info("-" * 40)
        logger.info(f"🎯 OVERALL TOTALS:")
        logger.info(f"  📁 Total files: {total_files}")
        logger.info(f"  ✅ Downloaded: {total_downloaded}")
        logger.info(f"  ❌ Failed: {total_failed}")
        logger.info(f"  📊 Success rate: {(total_downloaded/total_files*100):.1f}%" if total_files > 0 else "  📊 Success rate: 0%")
        logger.info("="*60)

def main():
    """Main entry point"""
    print("🚀 COMPREHENSIVE TCGA DOWNLOADER - ALL 33 CANCER TYPES 🚀")
    print("=========================================================")
    print("Conservative, rate-limited downloader targeting maximum coverage")
    print("100% REAL DATA - ZERO SYNTHETIC CONTAMINATION")
    print()
    
    downloader = ComprehensiveTCGADownloader()
    
    try:
        downloader.download_all_data()
    except KeyboardInterrupt:
        logger.info("🛑 Download interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
    finally:
        downloader.print_summary()

if __name__ == "__main__":
    main()
