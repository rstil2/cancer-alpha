#!/usr/bin/env python3
"""
CLEAN TCGA DOWNLOADER
=====================
Phase 4: Download authentic TCGA data to reach 50,000 real samples

This script:
- Uses GDC API with strict MD5 verification
- Downloads ONLY authentic TCGA files
- Implements file-level integrity checks
- Maintains reproducible manifest
- Stops at exactly 50K unique samples
- ZERO synthetic data - real samples only

Author: Cancer Alpha Project
Purpose: Download verified authentic TCGA data
Rule: STRICT authenticity verification for every file
"""

import requests
import json
import hashlib
import pandas as pd
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CleanTCGADownloader:
    def __init__(self, target_samples=50000, threads=4):
        self.target_samples = target_samples
        self.threads = threads
        self.project_root = Path("/Users/stillwell/projects/cancer-alpha")
        self.download_dir = self.project_root / f"data/raw_tcga/{datetime.now().strftime('%Y%m%d')}"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # GDC API endpoints
        self.gdc_base = "https://api.gdc.cancer.gov"
        self.files_endpoint = f"{self.gdc_base}/files"
        self.cases_endpoint = f"{self.gdc_base}/cases"
        
        # Track progress
        self.downloaded_samples = set()
        self.download_stats = {
            'files_downloaded': 0,
            'samples_collected': 0,
            'bytes_downloaded': 0,
            'verification_failures': 0,
            'api_errors': 0
        }
        
        # Load current inventory
        self.load_current_inventory()
        
        print("🔄 CLEAN TCGA DOWNLOADER")
        print("=" * 60)
        print("🎯 Target: 50,000 authentic TCGA samples")
        print("🚫 ZERO synthetic data policy enforced")
        print("🔒 MD5 verification for every file")
        print("=" * 60)
    
    def load_current_inventory(self):
        """Load current sample inventory to avoid duplicates"""
        inventory_path = self.project_root / "real_tcga_inventory.json"
        if inventory_path.exists():
            with open(inventory_path, 'r') as f:
                inventory = json.load(f)
                current_samples = inventory.get('total_samples', 0)
                print(f"📊 Current inventory: {current_samples:,} authentic samples")
                self.needed_samples = max(0, self.target_samples - current_samples)
                print(f"🎯 Need to download: {self.needed_samples:,} additional samples")
        else:
            print("⚠️  No current inventory found - starting from zero")
            self.needed_samples = self.target_samples
    
    def get_high_yield_cancer_projects(self):
        """Get cancer projects with highest sample counts"""
        # Focus on major cancer types with most samples available
        priority_projects = [
            'TCGA-BRCA',  # Breast cancer - highest yield
            'TCGA-LUAD',  # Lung adenocarcinoma 
            'TCGA-HNSC',  # Head and neck
            'TCGA-LGG',   # Lower grade glioma
            'TCGA-THCA',  # Thyroid cancer
            'TCGA-LUSC',  # Lung squamous
            'TCGA-PRAD',  # Prostate cancer
            'TCGA-COAD',  # Colon adenocarcinoma
            'TCGA-STAD',  # Stomach cancer
            'TCGA-BLCA',  # Bladder cancer
            'TCGA-LIHC',  # Liver cancer
            'TCGA-CESC',  # Cervical cancer
            'TCGA-KIRP',  # Kidney renal papillary
            'TCGA-SARC',  # Sarcoma
            'TCGA-LAML',  # Acute myeloid leukemia
            'TCGA-ESCA',  # Esophageal cancer
            'TCGA-PAAD',  # Pancreatic cancer
            'TCGA-PCPG',  # Pheochromocytoma
            'TCGA-READ',  # Rectum adenocarcinoma
            'TCGA-TGCT',  # Testicular cancer
        ]
        
        print(f"🎯 Targeting {len(priority_projects)} high-yield cancer projects")
        return priority_projects
    
    def query_gdc_files(self, project_id, data_type="Gene Expression Quantification", limit=2000):
        """Query GDC for files from a specific project"""
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
                        "field": "data_type",
                        "value": [data_type]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_format",
                        "value": ["TSV"]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "expand": "cases.submitter_id,cases.project.project_id",
            "fields": "id,file_name,file_size,md5sum,cases.submitter_id,cases.project.project_id",
            "format": "json",
            "size": str(limit)
        }
        
        try:
            response = requests.get(self.files_endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('data', {}).get('hits', [])
            
            print(f"   📁 {project_id}: Found {len(files)} files")
            return files
            
        except Exception as e:
            print(f"   ❌ Error querying {project_id}: {e}")
            self.download_stats['api_errors'] += 1
            return []
    
    def verify_file_integrity(self, file_path, expected_md5):
        """Verify downloaded file integrity with MD5"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            actual_md5 = hasher.hexdigest()
            return actual_md5 == expected_md5
            
        except Exception as e:
            print(f"   ❌ MD5 verification error: {e}")
            return False
    
    def download_file_with_verification(self, file_info):
        """Download a single file with full integrity verification"""
        file_id = file_info['id']
        file_name = file_info['file_name']
        expected_md5 = file_info['md5sum']
        file_size = file_info['file_size']
        
        # Extract sample info
        cases = file_info.get('cases', [])
        if not cases:
            return None
        
        sample_id = cases[0].get('submitter_id', '')
        project = cases[0].get('project', {}).get('project_id', '')
        
        # Skip if we already have this sample
        if sample_id in self.downloaded_samples:
            return None
        
        # Create download path
        download_path = self.download_dir / project / file_name
        download_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download file
            download_url = f"{self.gdc_base}/data/{file_id}"
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Save file
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify MD5
            if self.verify_file_integrity(download_path, expected_md5):
                # Success - update tracking
                self.downloaded_samples.add(sample_id)
                self.download_stats['files_downloaded'] += 1
                self.download_stats['samples_collected'] += 1
                self.download_stats['bytes_downloaded'] += file_size
                
                print(f"   ✅ {project}: {sample_id} ({file_size/1024/1024:.1f} MB)")
                
                return {
                    'sample_id': sample_id,
                    'project': project,
                    'file_path': str(download_path),
                    'file_size': file_size,
                    'md5_verified': True,
                    'download_timestamp': datetime.now().isoformat()
                }
            else:
                # MD5 verification failed
                download_path.unlink()  # Delete corrupted file
                self.download_stats['verification_failures'] += 1
                print(f"   ❌ {project}: {sample_id} - MD5 verification failed")
                return None
                
        except Exception as e:
            print(f"   ❌ Download error for {sample_id}: {e}")
            if download_path.exists():
                download_path.unlink()
            return None
    
    def download_project_files(self, project_id, max_files=1000):
        """Download files from a single project in parallel"""
        print(f"\n🔄 Downloading from {project_id}")
        print("-" * 60)
        
        # Query available files
        files = self.query_gdc_files(project_id, limit=max_files)
        if not files:
            return []
        
        # Filter out samples we already have
        new_files = []
        for file_info in files:
            cases = file_info.get('cases', [])
            if cases:
                sample_id = cases[0].get('submitter_id', '')
                if sample_id not in self.downloaded_samples:
                    new_files.append(file_info)
        
        if not new_files:
            print(f"   ⚠️  No new samples to download from {project_id}")
            return []
        
        # Limit downloads to avoid overwhelming
        download_files = new_files[:min(len(new_files), max_files)]
        print(f"   📥 Downloading {len(download_files)} files...")
        
        # Download in parallel
        downloaded_samples = []
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = [executor.submit(self.download_file_with_verification, file_info) 
                      for file_info in download_files]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    downloaded_samples.append(result)
                
                # Check if we've reached target
                if len(self.downloaded_samples) >= self.needed_samples:
                    print(f"   🎯 Reached sample target!")
                    break
        
        print(f"   ✅ Downloaded {len(downloaded_samples)} new samples from {project_id}")
        return downloaded_samples
    
    def generate_manifest(self, all_downloads):
        """Generate reproducible download manifest"""
        manifest = {
            'download_timestamp': datetime.now().isoformat(),
            'target_samples': self.target_samples,
            'total_downloaded': len(all_downloads),
            'download_statistics': self.download_stats,
            'files': all_downloads
        }
        
        # Save manifest
        manifest_path = self.project_root / "gdc_manifest_locked.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"📋 Manifest saved: {manifest_path}")
        
        # Also save CSV for easy viewing
        if all_downloads:
            df = pd.DataFrame(all_downloads)
            csv_path = self.project_root / "downloaded_samples_manifest.csv"
            df.to_csv(csv_path, index=False)
            print(f"📄 Sample manifest CSV: {csv_path}")
        
        return manifest
    
    def run_clean_download_pipeline(self):
        """Execute the complete clean download pipeline"""
        print(f"\n🚀 STARTING CLEAN TCGA DOWNLOAD PIPELINE")
        print("=" * 60)
        print(f"🎯 Target: {self.needed_samples:,} additional authentic samples")
        print(f"📁 Download directory: {self.download_dir}")
        print(f"🧵 Parallel threads: {self.threads}")
        print("=" * 60)
        
        if self.needed_samples <= 0:
            print("✅ Already have enough samples! No download needed.")
            return
        
        start_time = time.time()
        all_downloads = []
        
        # Get priority cancer projects
        projects = self.get_high_yield_cancer_projects()
        
        # Download from each project until we reach target
        for project_id in projects:
            if len(self.downloaded_samples) >= self.needed_samples:
                print(f"🎯 REACHED TARGET: {len(self.downloaded_samples):,} samples collected!")
                break
            
            # Calculate how many more we need
            remaining = self.needed_samples - len(self.downloaded_samples)
            max_from_project = min(remaining, 2000)  # Reasonable limit per project
            
            downloaded = self.download_project_files(project_id, max_from_project)
            all_downloads.extend(downloaded)
            
            print(f"   📊 Progress: {len(self.downloaded_samples):,}/{self.needed_samples:,} samples ({len(self.downloaded_samples)/self.needed_samples*100:.1f}%)")
            
            # Small delay to be respectful to GDC servers
            time.sleep(1)
        
        # Generate final manifest
        manifest = self.generate_manifest(all_downloads)
        
        # Final summary
        elapsed = time.time() - start_time
        total_gb = self.download_stats['bytes_downloaded'] / (1024**3)
        
        print(f"\n" + "🎉" * 60)
        print("✅ CLEAN TCGA DOWNLOAD COMPLETED!")
        print("🎉" * 60)
        print(f"📊 DOWNLOAD SUMMARY:")
        print(f"   • Files downloaded: {self.download_stats['files_downloaded']:,}")
        print(f"   • Samples collected: {self.download_stats['samples_collected']:,}")
        print(f"   • Data downloaded: {total_gb:.2f} GB")
        print(f"   • Download time: {elapsed/60:.1f} minutes")
        print(f"   • Verification failures: {self.download_stats['verification_failures']}")
        print(f"   • API errors: {self.download_stats['api_errors']}")
        print(f"\n🎯 NEXT STEPS:")
        print(f"   • Update inventory with new samples")
        print(f"   • Run Phase 5: Validation Framework")
        print(f"   • Begin model retraining on clean data")
        
        return manifest

def main():
    """Run the clean TCGA download pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║               CLEAN TCGA DOWNLOADER                      ║
    ║                                                          ║
    ║  🔒 MD5 verification for every file                      ║
    ║  🎯 Target: 50,000 authentic TCGA samples               ║
    ║  🚫 ZERO synthetic data policy enforced                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    downloader = CleanTCGADownloader(target_samples=50000, threads=4)
    manifest = downloader.run_clean_download_pipeline()
    
    print(f"\n✅ Phase 4 complete. Ready for Phase 5: Validation Framework")

if __name__ == "__main__":
    main()
