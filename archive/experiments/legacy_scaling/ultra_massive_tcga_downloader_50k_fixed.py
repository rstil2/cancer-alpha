#!/usr/bin/env python3
"""
Ultra-Massive TCGA Strategic Downloader for 50K+ Authentic Samples

Priority-driven downloader targeting missing cancer types and high-yield data types
to scale from current 19K to 50K+ authentic TCGA samples.
"""

import os
import sys
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import sqlite3

# Setup production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_massive_tcga_download_50k.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltraMassiveTCGADownloader:
    """Strategic downloader for scaling to 50K+ authentic TCGA samples"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Priority cancer types based on sample availability (highest first)
        self.priority_cancer_types = [
            'TCGA-BRCA',  # Breast cancer - typically >1000 samples
            'TCGA-KIRC',  # Kidney clear cell - typically >500 samples  
            'TCGA-OV',    # Ovarian - typically >300 samples
            'TCGA-UCEC',  # Endometrial - typically >500 samples
            'TCGA-GBM',   # Glioblastoma - typically >150 samples
            'TCGA-BLCA',  # Bladder - typically >400 samples
            'TCGA-CESC',  # Cervical - typically >300 samples
            'TCGA-SARC',  # Sarcoma - typically >250 samples
            'TCGA-PAAD',  # Pancreatic - typically >180 samples
            'TCGA-READ',  # Rectal - typically >160 samples
            'TCGA-LAML',  # Acute myeloid leukemia - typically >150 samples
            'TCGA-SKCM',  # Skin melanoma - typically >450 samples
            'TCGA-KICH',  # Kidney chromophobe - typically >65 samples
            'TCGA-KIRP',  # Kidney papillary - typically >280 samples
            'TCGA-ESCA',  # Esophageal - typically >180 samples
            'TCGA-CHOL',  # Cholangiocarcinoma - typically >35 samples
            'TCGA-DLBC',  # Lymphoid neoplasm - typically >45 samples
            'TCGA-MESO',  # Mesothelioma - typically >85 samples
            'TCGA-UVM',   # Uveal melanoma - typically >80 samples
            'TCGA-THYM',  # Thymoma - typically >120 samples
            'TCGA-TGCT',  # Testicular cancer - typically >150 samples
            'TCGA-UCS',   # Uterine carcinosarcoma - typically >55 samples
            'TCGA-PCPG',  # Pheochromocytoma - typically >180 samples
            'TCGA-ACC'    # Adrenocortical - typically >75 samples
        ]
        
        # High-yield data types for maximum multi-omics coverage
        self.priority_data_types = [
            'Gene Expression Quantification',    # RNA-seq
            'Masked Somatic Mutation',          # Mutations
            'Copy Number Segment',              # CNV
            'Methylation Beta Value',           # Methylation
            'Protein Expression Quantification', # Protein
            'Clinical Supplement',              # Clinical
            'miRNA Expression Quantification'   # miRNA
        ]
        
        # GDC API endpoints
        self.gdc_base_url = "https://api.gdc.cancer.gov"
        self.files_endpoint = f"{self.gdc_base_url}/files"
        self.data_endpoint = f"{self.gdc_base_url}/data"
        
        # Download session and rate limiting
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # Conservative rate limiting
        self.batch_size = 50
        self.max_concurrent_downloads = 8
        
        # Progress tracking
        self.download_stats = {
            'total_files_requested': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'bytes_downloaded': 0,
            'start_time': None
        }
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        # Setup database for tracking
        self.setup_tracking_database()
    
    def setup_tracking_database(self):
        """Setup SQLite database for download tracking"""
        self.db_path = self.output_dir / "ultra_massive_download_tracking.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_tracking (
                    file_id TEXT PRIMARY KEY,
                    cancer_type TEXT,
                    data_type TEXT,
                    filename TEXT,
                    file_size INTEGER,
                    download_status TEXT,
                    download_path TEXT,
                    timestamp TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS download_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    target_samples INTEGER,
                    files_downloaded INTEGER,
                    total_size_bytes INTEGER
                )
            """)
    
    def query_gdc_files(self, cancer_type, data_type, limit=2000):
        """Query GDC API for files of specific cancer type and data type"""
        
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [cancer_type]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": [data_type]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "format": "json",
            "fields": "id,file_name,file_size,data_type,cases.submitter_id",
            "size": str(limit)
        }
        
        try:
            response = self.session.get(self.files_endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('data', {}).get('pagination', {}).get('total', 0)
            
            logger.info(f"Found {len(data.get('data', {}).get('hits', []))} {data_type} files for {cancer_type}")
            return data.get('data', {}).get('hits', [])
            
        except Exception as e:
            logger.error(f"Failed to query {cancer_type}/{data_type}: {e}")
            return []
    
    def download_file(self, file_info, cancer_type, data_type):
        """Download single file with error handling and progress tracking"""
        file_id = file_info['id']
        filename = file_info['file_name']
        file_size = file_info.get('file_size', 0)
        
        # Create cancer type and data type subdirectories
        cancer_dir = self.output_dir / cancer_type / data_type
        cancer_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = cancer_dir / filename
        
        # Skip if already downloaded
        if file_path.exists() and file_path.stat().st_size > 0:
            logger.debug(f"Skipping existing file: {filename}")
            with self.stats_lock:
                self.download_stats['successful_downloads'] += 1
            return True
        
        try:
            # Download file
            download_url = f"{self.data_endpoint}/{file_id}"
            
            with self.session.get(download_url, stream=True, timeout=60) as response:
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    downloaded_bytes = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
                
                # Verify download
                if downloaded_bytes > 0:
                    with self.stats_lock:
                        self.download_stats['successful_downloads'] += 1
                        self.download_stats['bytes_downloaded'] += downloaded_bytes
                    
                    # Update tracking database
                    self.update_download_tracking(file_id, cancer_type, data_type, 
                                                filename, downloaded_bytes, "success", str(file_path))
                    
                    logger.info(f"Downloaded {filename} ({downloaded_bytes:,} bytes)")
                    return True
                else:
                    logger.warning(f"Empty download for {filename}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            with self.stats_lock:
                self.download_stats['failed_downloads'] += 1
            
            # Update tracking database
            self.update_download_tracking(file_id, cancer_type, data_type, 
                                        filename, 0, "failed", "")
            return False
        
        finally:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
    
    def update_download_tracking(self, file_id, cancer_type, data_type, filename, size, status, path):
        """Update download tracking database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO download_tracking 
                    (file_id, cancer_type, data_type, filename, file_size, download_status, download_path, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (file_id, cancer_type, data_type, filename, size, status, path, datetime.now().isoformat()))
        except Exception as e:
            logger.error(f"Database update failed: {e}")
    
    def download_cancer_type(self, cancer_type, target_samples_per_type=2000):
        """Download all priority data types for a specific cancer type"""
        logger.info(f"Starting download for {cancer_type} (target: {target_samples_per_type} samples)")
        
        total_files = []
        
        for data_type in self.priority_data_types:
            logger.info(f"Querying {cancer_type} / {data_type}...")
            
            files = self.query_gdc_files(cancer_type, data_type, limit=target_samples_per_type)
            
            if files:
                total_files.extend([(f, cancer_type, data_type) for f in files])
                logger.info(f"Added {len(files)} {data_type} files for {cancer_type}")
            
            time.sleep(0.5)  # API courtesy delay
        
        if not total_files:
            logger.warning(f"No files found for {cancer_type}")
            return 0
        
        logger.info(f"Downloading {len(total_files)} files for {cancer_type}...")
        
        # Parallel download with conservative concurrency
        successful_downloads = 0
        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
            future_to_file = {
                executor.submit(self.download_file, file_info, cancer_type, data_type): (file_info, cancer_type, data_type)
                for file_info, cancer_type, data_type in total_files
            }
            
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    if result:
                        successful_downloads += 1
                        
                    # Progress update every 50 downloads
                    if successful_downloads % 50 == 0:
                        logger.info(f"Progress for {cancer_type}: {successful_downloads}/{len(total_files)} files")
                        
                except Exception as e:
                    logger.error(f"Download task failed: {e}")
        
        logger.info(f"Completed {cancer_type}: {successful_downloads}/{len(total_files)} files downloaded")
        return successful_downloads
    
    def execute_ultra_massive_download(self, target_cancer_types=None, samples_per_cancer=2000):
        """Execute strategic download for multiple cancer types"""
        
        if target_cancer_types is None:
            target_cancer_types = self.priority_cancer_types[:15]  # Top 15 priorities
        
        logger.info(f"Starting ultra-massive download for {len(target_cancer_types)} cancer types")
        logger.info(f"Target: {samples_per_cancer} samples per cancer type")
        logger.info(f"Total target samples: {len(target_cancer_types) * samples_per_cancer:,}")
        
        self.download_stats['start_time'] = datetime.now()
        session_id = self.download_stats['start_time'].strftime("%Y%m%d_%H%M%S")
        
        total_downloaded = 0
        
        for i, cancer_type in enumerate(target_cancer_types, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing cancer type {i}/{len(target_cancer_types)}: {cancer_type}")
            logger.info(f"{'='*60}")
            
            downloaded = self.download_cancer_type(cancer_type, samples_per_cancer)
            total_downloaded += downloaded
            
            # Progress summary
            elapsed = (datetime.now() - self.download_stats['start_time']).total_seconds() / 3600
            rate = total_downloaded / elapsed if elapsed > 0 else 0
            eta_hours = (len(target_cancer_types) - i) * elapsed / i if i > 0 else 0
            
            logger.info(f"Session progress: {i}/{len(target_cancer_types)} cancer types")
            logger.info(f"Total files downloaded: {total_downloaded:,}")
            logger.info(f"Download rate: {rate:.1f} files/hour")
            logger.info(f"ETA: {eta_hours:.1f} hours")
            
            # Save session progress
            self.save_session_progress(session_id, target_cancer_types, i, total_downloaded)
            
            # Brief pause between cancer types
            time.sleep(2)
        
        # Final summary
        final_stats = self.generate_download_summary(session_id, total_downloaded)
        self.save_final_summary(final_stats)
        
        logger.info(f"\nUltra-massive download complete!")
        logger.info(f"Total files downloaded: {total_downloaded:,}")
        logger.info(f"Total size: {self.download_stats['bytes_downloaded'] / (1024**3):.2f} GB")
        
        return final_stats
    
    def save_session_progress(self, session_id, target_cancers, completed_cancers, files_downloaded):
        """Save current session progress"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO download_sessions 
                    (session_id, start_time, end_time, target_samples, files_downloaded, total_size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    self.download_stats['start_time'].isoformat(),
                    datetime.now().isoformat(),
                    len(target_cancers) * 2000,  # Target samples
                    files_downloaded,
                    self.download_stats['bytes_downloaded']
                ))
        except Exception as e:
            logger.error(f"Failed to save session progress: {e}")
    
    def generate_download_summary(self, session_id, total_downloaded):
        """Generate comprehensive download summary"""
        end_time = datetime.now()
        duration = end_time - self.download_stats['start_time']
        
        summary = {
            'session_id': session_id,
            'start_time': self.download_stats['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'duration_hours': duration.total_seconds() / 3600,
            'total_files_downloaded': total_downloaded,
            'successful_downloads': self.download_stats['successful_downloads'],
            'failed_downloads': self.download_stats['failed_downloads'],
            'total_size_gb': self.download_stats['bytes_downloaded'] / (1024**3),
            'download_rate_files_per_hour': total_downloaded / (duration.total_seconds() / 3600) if duration.total_seconds() > 0 else 0,
            'estimated_new_samples': total_downloaded * 0.7,  # Conservative estimate
            'output_directory': str(self.output_dir)
        }
        
        return summary
    
    def save_final_summary(self, summary):
        """Save final download summary"""
        summary_file = self.output_dir / f"ultra_massive_download_summary_{summary['session_id']}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Download summary saved to {summary_file}")
        
        # Also create human-readable summary
        readable_summary = self.output_dir / f"download_summary_{summary['session_id']}.txt"
        
        with open(readable_summary, 'w') as f:
            f.write("ULTRA-MASSIVE TCGA DOWNLOAD SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Session ID: {summary['session_id']}\n")
            f.write(f"Duration: {summary['duration_hours']:.2f} hours\n")
            f.write(f"Files Downloaded: {summary['total_files_downloaded']:,}\n")
            f.write(f"Success Rate: {summary['successful_downloads']}/{summary['successful_downloads'] + summary['failed_downloads']}\n")
            f.write(f"Total Size: {summary['total_size_gb']:.2f} GB\n")
            f.write(f"Download Rate: {summary['download_rate_files_per_hour']:.1f} files/hour\n")
            f.write(f"Estimated New Samples: {summary['estimated_new_samples']:,.0f}\n")
            f.write(f"Output Directory: {summary['output_directory']}\n")
        
        print(f"\nDownload summary: {readable_summary}")

def main():
    """Main execution function"""
    logger.info("Initializing Ultra-Massive TCGA Strategic Downloader...")
    
    # Output directory for new downloads
    output_dir = "/Users/stillwell/projects/cancer-alpha/data/tcga_ultra_massive_50k"
    
    # Create downloader
    downloader = UltraMassiveTCGADownloader(output_dir)
    
    # Phase 1: Download top 10 highest-yield cancer types
    phase1_cancers = [
        'TCGA-BRCA', 'TCGA-KIRC', 'TCGA-OV', 'TCGA-UCEC', 'TCGA-BLCA',
        'TCGA-SKCM', 'TCGA-CESC', 'TCGA-SARC', 'TCGA-KIRP', 'TCGA-GBM'
    ]
    
    print(f"\nPhase 1: Downloading {len(phase1_cancers)} high-yield cancer types")
    print(f"Target: 2,000 samples per cancer type")
    print(f"Expected total: {len(phase1_cancers) * 2000:,} new samples")
    print(f"\nThis will significantly boost our dataset toward the 50K target!")
    
    # Execute download
    summary = downloader.execute_ultra_massive_download(
        target_cancer_types=phase1_cancers,
        samples_per_cancer=2000
    )
    
    print(f"\nPhase 1 Complete!")
    print(f"Files Downloaded: {summary['total_files_downloaded']:,}")
    print(f"Estimated New Samples: {summary['estimated_new_samples']:,.0f}")
    print(f"Total Data Size: {summary['total_size_gb']:.2f} GB")
    
    # Recommendations for next phases
    print(f"\nNext Steps:")
    print(f"1. Process the new data with enhanced multi-omics integration")
    print(f"2. Run Phase 2 with remaining 13 cancer types")
    print(f"3. Train breakthrough models on 50K+ sample dataset")
    print(f"4. Deploy production cancer genomics AI system")

if __name__ == "__main__":
    main()
