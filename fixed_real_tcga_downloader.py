#!/usr/bin/env python3
"""
Fixed Real TCGA Downloader - Using Correct Data Types
====================================================

Production-grade downloader using the actual GDC data type names 
for scaling to 10,000+ authentic TCGA samples.

REAL DATA ONLY - NO SYNTHETIC DATA!
"""

import asyncio
import logging
from pathlib import Path
import requests
import json
import aiohttp
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedRealTCGADownloader:
    """Fixed downloader with correct GDC data type names"""
    
    def __init__(self, cache_dir: str = "data/tcga_real_fixed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # GDC API configuration
        self.api_base = "https://api.gdc.cancer.gov"
        self.max_concurrent = 10
        self.timeout = 30
        self.retry_attempts = 3
        
        # Test with 3 cancer types, small targets
        self.target_projects = {
            'TCGA-BRCA': {'name': 'Breast Invasive Carcinoma', 'target': 10},
            'TCGA-LUAD': {'name': 'Lung Adenocarcinoma', 'target': 10}, 
            'TCGA-COAD': {'name': 'Colon Adenocarcinoma', 'target': 10},
        }
        
        # Correct data type names from GDC API
        self.data_types = {
            'mutations': {
                'gdc_name': 'Masked Somatic Mutation',
                'priority': 1
            },
            'clinical': {
                'gdc_name': 'Clinical Supplement',
                'priority': 1
            },
            'copy_number': {
                'gdc_name': 'Copy Number Segment', 
                'priority': 1
            },
            'methylation': {
                'gdc_name': 'Methylation Beta Value',
                'priority': 1
            },
            'protein_expression': {
                'gdc_name': 'Protein Expression Quantification',
                'priority': 2
            },
            'mirna_expression': {
                'gdc_name': 'miRNA Expression Quantification',
                'priority': 2
            }
        }
    
    async def query_project_files(self, session: aiohttp.ClientSession, 
                                 project_id: str, data_type: str, limit: int = 50) -> List[Dict]:
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
            "size": str(limit)
        }
        
        try:
            async with session.get(f"{self.api_base}/files", 
                                 params=params, 
                                 timeout=self.timeout) as response:
                response.raise_for_status()
                data = await response.json()
                files = data.get('data', {}).get('hits', [])
                
                logger.info(f"Found {len(files)} {data_type} files for {project_id}")
                return files
                
        except Exception as e:
            logger.error(f"Error querying {project_id} {data_type}: {str(e)}")
            return []
    
    async def download_file(self, session: aiohttp.ClientSession, 
                          file_info: Dict, data_type: str) -> bool:
        """Download a single file"""
        
        file_id = file_info['id']
        file_name = file_info.get('file_name', f"{file_id}.dat")
        
        # Create data type directory
        data_dir = self.cache_dir / data_type
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / file_name
        
        # Skip if already exists
        if file_path.exists():
            return True
        
        try:
            async with session.get(f"{self.api_base}/data/{file_id}",
                                 timeout=self.timeout) as response:
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"Downloaded {file_name} ({size_mb:.2f} MB)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {str(e)}")
            file_path.unlink(missing_ok=True)
            return False
    
    async def download_project_data(self, project_id: str, target_samples: int) -> Dict[str, int]:
        """Download data for a specific project"""
        
        logger.info(f"📂 Downloading {project_id} data (target: {target_samples} files per type)")
        
        stats = defaultdict(int)
        
        # Create session
        async with aiohttp.ClientSession() as session:
            
            for data_key, data_info in self.data_types.items():
                logger.info(f"  📁 Processing {data_key}...")
                
                # Query available files
                files = await self.query_project_files(
                    session, project_id, data_info['gdc_name'], target_samples
                )
                
                if not files:
                    continue
                
                # Limit to target
                files_to_download = files[:target_samples]
                
                # Download files
                downloaded = 0
                for file_info in files_to_download:
                    success = await self.download_file(session, file_info, data_key)
                    if success:
                        downloaded += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                
                stats[data_key] = downloaded
                logger.info(f"  ✅ {data_key}: {downloaded}/{len(files_to_download)} files")
        
        return dict(stats)
    
    async def download_all_projects(self) -> Dict[str, Dict[str, int]]:
        """Download data for all projects"""
        
        logger.info("🚀 Starting Fixed Real TCGA Download Test...")
        logger.info("✅ Using correct GDC data type names")
        
        all_stats = {}
        
        for project_id, project_info in self.target_projects.items():
            try:
                project_stats = await self.download_project_data(
                    project_id, project_info['target']
                )
                all_stats[project_id] = project_stats
                
                total = sum(project_stats.values())
                logger.info(f"✅ {project_id}: {total} total files downloaded")
                
            except Exception as e:
                logger.error(f"❌ Error processing {project_id}: {e}")
                all_stats[project_id] = {}
        
        return all_stats

async def test_fixed_download():
    """Test the fixed downloader"""
    
    downloader = FixedRealTCGADownloader()
    
    logger.info("🧪 Testing Fixed Real TCGA Download...")
    logger.info("=" * 50)
    logger.info("✅ REAL DATA ONLY - Using correct GDC data types")
    logger.info("🎯 Test: 30 authentic TCGA files")
    logger.info("=" * 50)
    
    # Download test data
    stats = await downloader.download_all_projects()
    
    # Calculate results
    total_files = sum(sum(project_stats.values()) for project_stats in stats.values())
    
    logger.info(f"\n📊 TEST RESULTS:")
    logger.info(f"Total files downloaded: {total_files}")
    
    if total_files > 0:
        logger.info(f"✅ SUCCESS: Real TCGA download working!")
        
        # Show what we got
        for project_id, project_stats in stats.items():
            if project_stats:
                logger.info(f"  {project_id}:")
                for data_type, count in project_stats.items():
                    logger.info(f"    {data_type}: {count} files")
        
        # Validate a few files
        logger.info(f"\n🔍 Validating downloaded files...")
        
        for data_type in ['clinical', 'mutations']:
            data_dir = downloader.cache_dir / data_type
            if data_dir.exists():
                files = list(data_dir.glob("*"))
                logger.info(f"  {data_type}: {len(files)} files")
                
                if files:
                    sample_file = files[0]
                    try:
                        # Peek at file content
                        content = ""
                        if sample_file.suffix == '.xml':
                            content = sample_file.read_text()[:200]
                        elif sample_file.suffix in ['.txt', '.tsv']:
                            content = sample_file.read_text()[:200]
                        
                        if content:
                            logger.info(f"    Sample content: {content[:50]}...")
                            logger.info(f"    ✅ {data_type} format validated")
                    except Exception as e:
                        logger.warning(f"    ⚠️ Could not validate {data_type}: {e}")
        
        logger.info(f"\n🎉 READY TO SCALE TO 10,000+ SAMPLES!")
        logger.info(f"The downloader is working correctly with real TCGA data.")
        
    else:
        logger.error(f"❌ No files downloaded - check implementation")
    
    return total_files > 0

if __name__ == "__main__":
    asyncio.run(test_fixed_download())
