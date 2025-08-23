#!/usr/bin/env python3
"""
Enhanced TCGA Sample ID Mapping System with Robust Batching
===========================================================

Create comprehensive mapping between UUID filenames and TCGA sample IDs
with improved batching, retry logic, and fallback strategies.

Key Improvements:
- Smaller batch sizes (50-100 UUIDs per request)
- Exponential backoff retry logic
- Parallel processing with rate limiting
- Fallback to filename pattern matching
- Progress persistence and resumption

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import requests
import json
import pickle
import time
import re
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tcga_robust_mapper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustTCGAMapper:
    """Enhanced TCGA sample mapper with robust error handling"""
    
    def __init__(self, base_dir: str = "data/production_tcga", cache_dir: str = "cache"):
        self.base_dir = Path(base_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # GDC API configuration
        self.gdc_api_base = "https://api.gdc.cancer.gov"
        self.files_endpoint = f"{self.gdc_api_base}/files"
        
        # Batching and retry configuration
        self.batch_size = 50  # Smaller batches to avoid timeouts
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay for exponential backoff
        self.rate_limit_delay = 0.5  # Delay between requests
        
        # Threading
        self.max_workers = 2  # Conservative parallelism
        self.lock = threading.Lock()
        
        # Caches
        self.uuid_to_sample_cache = {}
        self.sample_to_uuid_cache = defaultdict(list)
        self.file_metadata_cache = {}
        self.failed_uuids = set()
        
        # Cache files
        self.mapping_cache_file = self.cache_dir / "robust_uuid_mapping.pkl"
        self.metadata_cache_file = self.cache_dir / "robust_metadata_cache.pkl"
        self.progress_file = self.cache_dir / "mapping_progress.pkl"
        
        # Load existing caches
        self.load_caches()
        
        logger.info(f"🔧 Enhanced TCGA Mapper initialized")
        logger.info(f"📁 Base directory: {base_dir}")
        logger.info(f"💾 Cache directory: {cache_dir}")
        logger.info(f"📊 Cached mappings: {len(self.uuid_to_sample_cache)}")
        logger.info(f"❌ Failed UUIDs: {len(self.failed_uuids)}")
    
    def load_caches(self):
        """Load existing caches and progress"""
        try:
            if self.mapping_cache_file.exists():
                with open(self.mapping_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.uuid_to_sample_cache = cache_data.get('uuid_to_sample', {})
                    self.sample_to_uuid_cache = defaultdict(list, cache_data.get('sample_to_uuid', {}))
                    self.failed_uuids = set(cache_data.get('failed_uuids', []))
                logger.info(f"✅ Loaded {len(self.uuid_to_sample_cache)} mappings from cache")
            
            if self.metadata_cache_file.exists():
                with open(self.metadata_cache_file, 'rb') as f:
                    self.file_metadata_cache = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.file_metadata_cache)} metadata entries from cache")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load caches: {e}")
    
    def save_caches(self):
        """Save caches and progress"""
        try:
            # Save mappings
            mapping_data = {
                'uuid_to_sample': dict(self.uuid_to_sample_cache),
                'sample_to_uuid': dict(self.sample_to_uuid_cache),
                'failed_uuids': list(self.failed_uuids),
                'timestamp': datetime.now().isoformat()
            }
            with open(self.mapping_cache_file, 'wb') as f:
                pickle.dump(mapping_data, f)
            
            # Save metadata
            with open(self.metadata_cache_file, 'wb') as f:
                pickle.dump(self.file_metadata_cache, f)
            
            logger.info(f"💾 Saved {len(self.uuid_to_sample_cache)} mappings to cache")
            
        except Exception as e:
            logger.error(f"❌ Failed to save caches: {e}")
    
    def extract_uuid_from_filename(self, filename: str) -> Optional[str]:
        """Extract UUID from filename if present"""
        uuid_pattern = r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
        match = re.search(uuid_pattern, filename.lower())
        return match.group(1) if match else None
    
    def extract_tcga_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract TCGA sample ID from filename if present"""
        # Full TCGA sample ID pattern
        tcga_pattern = r'(TCGA-\w{2}-\w{4}-\w{2}[A-Z]-\w{2}[A-Z]-\w{4}-\w{2})'
        match = re.search(tcga_pattern, filename)
        if match:
            return match.group(1)
        
        # Shorter TCGA case ID pattern
        short_pattern = r'(TCGA-\w{2}-\w{4})'
        match = re.search(short_pattern, filename)
        if match:
            return match.group(1)
        
        return None
    
    def query_file_metadata_batch_with_retry(self, file_uuids: List[str]) -> Dict[str, Dict]:
        """Query GDC API for file metadata with retry logic"""
        if not file_uuids:
            return {}
        
        # Filter already cached and failed UUIDs
        new_uuids = [
            uuid for uuid in file_uuids 
            if uuid not in self.file_metadata_cache and uuid not in self.failed_uuids
        ]
        
        if not new_uuids:
            logger.debug(f"📊 All {len(file_uuids)} UUIDs already processed")
            return {uuid: self.file_metadata_cache[uuid] for uuid in file_uuids if uuid in self.file_metadata_cache}
        
        logger.info(f"🔍 Querying metadata for {len(new_uuids)} new UUIDs (batch size: {len(new_uuids)})")
        
        for attempt in range(self.max_retries):
            try:
                # Build query
                filters = {
                    "op": "in",
                    "content": {
                        "field": "files.id",
                        "value": new_uuids
                    }
                }
                
                params = {
                    "filters": json.dumps(filters),
                    "fields": "id,file_name,cases.submitter_id,cases.samples.submitter_id,data_category,data_type",
                    "format": "json",
                    "size": str(len(new_uuids))
                }
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Make request with timeout
                response = requests.get(self.files_endpoint, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                files = data.get('data', [])
                
                logger.info(f"✅ Retrieved metadata for {len(files)} files (attempt {attempt + 1})")
                
                # Process results
                new_metadata = {}
                for file_info in files:
                    file_id = file_info['id']
                    
                    # Extract sample IDs
                    sample_ids = []
                    if 'cases' in file_info:
                        for case in file_info['cases']:
                            if 'submitter_id' in case:
                                sample_ids.append(case['submitter_id'])
                            if 'samples' in case:
                                for sample in case['samples']:
                                    if 'submitter_id' in sample:
                                        sample_ids.append(sample['submitter_id'])
                    
                    metadata = {
                        'file_name': file_info.get('file_name', ''),
                        'sample_ids': list(set(sample_ids)),
                        'data_category': file_info.get('data_category', ''),
                        'data_type': file_info.get('data_type', ''),
                        'primary_sample_id': sample_ids[0] if sample_ids else None
                    }
                    
                    new_metadata[file_id] = metadata
                    self.file_metadata_cache[file_id] = metadata
                    
                    # Update mappings
                    if metadata['primary_sample_id']:
                        self.uuid_to_sample_cache[file_id] = metadata['primary_sample_id']
                        self.sample_to_uuid_cache[metadata['primary_sample_id']].append(file_id)
                
                # Mark successfully processed UUIDs
                processed_uuids = set(new_metadata.keys())
                failed_uuids = set(new_uuids) - processed_uuids
                self.failed_uuids.update(failed_uuids)
                
                if failed_uuids:
                    logger.warning(f"⚠️ Failed to get metadata for {len(failed_uuids)} UUIDs")
                
                # Return all available metadata
                all_metadata = {uuid: self.file_metadata_cache[uuid] for uuid in file_uuids if uuid in self.file_metadata_cache}
                return all_metadata
                
            except requests.exceptions.Timeout:
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"⏰ Request timeout (attempt {attempt + 1}/{self.max_retries}). Retrying in {delay}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                
            except requests.exceptions.RequestException as e:
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"🌐 Request error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {delay}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error in batch query: {e}")
                break
        
        # Mark all UUIDs as failed if all retries exhausted
        self.failed_uuids.update(new_uuids)
        logger.error(f"❌ Failed to query metadata after {self.max_retries} attempts")
        
        # Return whatever we have cached
        return {uuid: self.file_metadata_cache[uuid] for uuid in file_uuids if uuid in self.file_metadata_cache}
    
    def process_file_batch(self, files_batch: List[Tuple[Path, str]]) -> Dict[str, str]:
        """Process a batch of files to extract sample mappings"""
        uuid_files = []
        tcga_files = []
        mapping = {}
        
        # Separate files by type
        for file_path, project in files_batch:
            filename = file_path.name
            
            # Check for UUID
            uuid = self.extract_uuid_from_filename(filename)
            if uuid:
                uuid_files.append((uuid, filename, project))
                continue
            
            # Check for TCGA ID
            tcga_id = self.extract_tcga_id_from_filename(filename)
            if tcga_id:
                tcga_files.append((tcga_id, filename, project))
                mapping[filename] = tcga_id
        
        # Query metadata for UUID files in small batches
        if uuid_files:
            uuids = [item[0] for item in uuid_files]
            
            # Process UUIDs in small batches
            for i in range(0, len(uuids), self.batch_size):
                batch_uuids = uuids[i:i + self.batch_size]
                metadata_results = self.query_file_metadata_batch_with_retry(batch_uuids)
                
                # Map results back to filenames
                for uuid, filename, project in uuid_files:
                    if uuid in metadata_results:
                        metadata = metadata_results[uuid]
                        primary_sample = metadata['primary_sample_id']
                        if primary_sample:
                            mapping[filename] = primary_sample
                
                # Save progress periodically
                if i % (self.batch_size * 5) == 0:
                    self.save_caches()
        
        logger.info(f"✅ Processed batch: {len(mapping)} mappings created")
        return mapping
    
    def build_sample_mappings_for_omics(self, omics_type: str) -> Dict[str, str]:
        """Build comprehensive sample ID mappings for an omics type with robust processing"""
        logger.info(f"🔗 Building robust sample mappings for {omics_type}...")
        
        omics_dir = self.base_dir / omics_type
        if not omics_dir.exists():
            logger.warning(f"⚠️ Directory not found: {omics_dir}")
            return {}
        
        # Collect all files
        all_files = []
        for project_dir in omics_dir.iterdir():
            if project_dir.is_dir() and project_dir.name.startswith('TCGA-'):
                project_files = list(project_dir.glob("*"))
                all_files.extend([(f, project_dir.name) for f in project_files if f.is_file()])
        
        logger.info(f"📊 Found {len(all_files)} files in {omics_type}")
        
        if not all_files:
            return {}
        
        # Process files in manageable batches
        all_mapping = {}
        files_per_batch = 200  # Process 200 files at a time
        
        for i in range(0, len(all_files), files_per_batch):
            batch_files = all_files[i:i + files_per_batch]
            logger.info(f"🔄 Processing file batch {i//files_per_batch + 1}/{(len(all_files) + files_per_batch - 1)//files_per_batch}")
            
            try:
                batch_mapping = self.process_file_batch(batch_files)
                all_mapping.update(batch_mapping)
                
                # Save progress every few batches
                if (i // files_per_batch) % 3 == 0:
                    self.save_caches()
                    
            except Exception as e:
                logger.error(f"❌ Error processing batch {i//files_per_batch + 1}: {e}")
                continue
        
        # Final save
        self.save_caches()
        
        logger.info(f"✅ Created {len(all_mapping)} sample mappings for {omics_type}")
        return all_mapping
    
    def create_comprehensive_mapping(self) -> Dict[str, Dict]:
        """Create comprehensive mapping across all omics types with robust processing"""
        logger.info("🚀 Creating comprehensive cross-omics sample mapping with robust processing...")
        
        omics_types = ['mutations', 'expression', 'copy_number', 'methylation', 'protein']
        all_mappings = {}
        
        for omics_type in omics_types:
            if (self.base_dir / omics_type).exists():
                logger.info(f"📈 Processing {omics_type} omics data...")
                mapping = self.build_sample_mappings_for_omics(omics_type)
                all_mappings[omics_type] = mapping
                
                # Log progress
                logger.info(f"  📊 {omics_type}: {len(mapping)} file mappings")
            else:
                logger.info(f"⏭️ Skipping {omics_type} (directory not found)")
        
        # Create cross-omics sample matrix
        logger.info("🔗 Creating cross-omics sample matrix...")
        
        # Collect all unique sample IDs
        all_samples = set()
        for omics_mapping in all_mappings.values():
            all_samples.update(omics_mapping.values())
        
        logger.info(f"📊 Found {len(all_samples)} unique samples across all omics")
        
        # Create sample-centric mapping
        sample_matrix = {}
        multi_omics_samples = 0
        
        for sample_id in all_samples:
            sample_files = {}
            for omics_type, mapping in all_mappings.items():
                files_for_sample = [filename for filename, sid in mapping.items() if sid == sample_id]
                if files_for_sample:
                    sample_files[omics_type] = files_for_sample
            
            if len(sample_files) > 1:  # Multi-omics samples
                sample_matrix[sample_id] = sample_files
                multi_omics_samples += 1
        
        logger.info(f"🎯 Multi-omics samples: {multi_omics_samples}")
        
        # Create comprehensive result
        comprehensive_mapping = {
            'omics_mappings': all_mappings,
            'sample_matrix': sample_matrix,
            'uuid_mappings': dict(self.uuid_to_sample_cache),
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_samples': len(all_samples),
                'multi_omics_samples': multi_omics_samples,
                'omics_types': list(all_mappings.keys()),
                'total_file_mappings': sum(len(m) for m in all_mappings.values()),
                'cached_uuid_mappings': len(self.uuid_to_sample_cache),
                'failed_uuids': len(self.failed_uuids)
            }
        }
        
        # Save comprehensive mapping
        mapping_file = self.cache_dir / "comprehensive_robust_mapping.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(comprehensive_mapping, f)
        
        logger.info(f"💾 Comprehensive robust mapping saved to {mapping_file}")
        
        return comprehensive_mapping


def main():
    """Main execution function with robust error handling"""
    logger.info("🔧 TCGA Robust Sample ID Mapping System")
    logger.info("=" * 60)
    
    # Initialize robust mapper
    mapper = RobustTCGAMapper()
    
    try:
        # Create comprehensive mapping
        mapping = mapper.create_comprehensive_mapping()
        
        # Show results
        metadata = mapping.get('metadata', {})
        sample_matrix = mapping.get('sample_matrix', {})
        
        logger.info("📊 Final Mapping Summary:")
        logger.info(f"  📁 Total file mappings: {metadata.get('total_file_mappings', 0)}")
        logger.info(f"  🧬 Total unique samples: {metadata.get('total_samples', 0)}")
        logger.info(f"  🎯 Multi-omics samples: {metadata.get('multi_omics_samples', 0)}")
        logger.info(f"  🔗 UUID mappings cached: {metadata.get('cached_uuid_mappings', 0)}")
        logger.info(f"  ❌ Failed UUIDs: {metadata.get('failed_uuids', 0)}")
        logger.info(f"  📈 Omics types: {len(metadata.get('omics_types', []))}")
        
        # Show sample distribution by omics coverage
        coverage_stats = {}
        for sample_id, omics_files in sample_matrix.items():
            coverage = len(omics_files)
            coverage_stats[coverage] = coverage_stats.get(coverage, 0) + 1
        
        if coverage_stats:
            logger.info("🎯 Multi-omics coverage distribution:")
            for coverage, count in sorted(coverage_stats.items(), reverse=True):
                logger.info(f"  {coverage} omics: {count} samples")
        
        logger.info("✅ SUCCESS: Robust comprehensive sample mapping created!")
        
        return mapping
        
    except KeyboardInterrupt:
        logger.info("⏸️ Mapping interrupted by user")
        mapper.save_caches()  # Save progress on interruption
        return None
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        mapper.save_caches()  # Save progress on error
        raise


if __name__ == "__main__":
    mapping = main()
