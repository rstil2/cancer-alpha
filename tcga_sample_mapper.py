#!/usr/bin/env python3
"""
TCGA Sample ID Mapping System
============================

Create comprehensive mapping between UUID filenames and TCGA sample IDs
across all omics data types using GDC API metadata queries.

This solves the critical issue where:
- Expression files use UUID filenames (e.g., 01661d94-fc16-4456-95cf-a5fa4e1e196c.rna_seq...)
- Protein files use TCGA sample IDs (e.g., TCGA-4H-AAAK-01A-21-A43F-20_RPPA_data.tsv)
- Other omics types may use different naming conventions

Key Features:
- GDC API metadata queries to map file UUIDs to TCGA sample IDs
- Comprehensive cross-omics sample matching
- Persistent caching for efficiency
- Batch processing for large datasets
- Error handling and validation

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import requests
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
import warnings
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tcga_sample_mapping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TCGASampleMapper:
    """Map between file UUIDs and TCGA sample IDs across omics types"""
    
    def __init__(self, base_dir: str = "data/production_tcga", cache_dir: str = "cache"):
        self.base_dir = Path(base_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # GDC API endpoints
        self.gdc_api_base = "https://api.gdc.cancer.gov"
        self.files_endpoint = f"{self.gdc_api_base}/files"
        
        # Mapping caches
        self.uuid_to_sample_cache = {}
        self.sample_to_uuid_cache = {}
        self.file_metadata_cache = {}
        
        # Cache files
        self.mapping_cache_file = self.cache_dir / "uuid_sample_mapping.pkl"
        self.metadata_cache_file = self.cache_dir / "file_metadata_cache.pkl"
        
        # Load existing caches
        self.load_caches()
        
        # Rate limiting
        self.request_delay = 0.5
        self.lock = threading.Lock()
        
        logger.info(f"🔗 TCGA Sample Mapper initialized")
        logger.info(f"📁 Base directory: {base_dir}")
        logger.info(f"💾 Cache directory: {cache_dir}")
        logger.info(f"📊 Cached mappings: {len(self.uuid_to_sample_cache)}")
    
    def load_caches(self):
        """Load existing mapping caches"""
        try:
            if self.mapping_cache_file.exists():
                with open(self.mapping_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.uuid_to_sample_cache = cache_data.get('uuid_to_sample', {})
                    self.sample_to_uuid_cache = cache_data.get('sample_to_uuid', {})
                logger.info(f"✅ Loaded {len(self.uuid_to_sample_cache)} cached mappings")
            
            if self.metadata_cache_file.exists():
                with open(self.metadata_cache_file, 'rb') as f:
                    self.file_metadata_cache = pickle.load(f)
                logger.info(f"✅ Loaded {len(self.file_metadata_cache)} cached metadata entries")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load caches: {e}")
    
    def save_caches(self):
        """Save mapping caches"""
        try:
            # Save mappings
            mapping_data = {
                'uuid_to_sample': self.uuid_to_sample_cache,
                'sample_to_uuid': self.sample_to_uuid_cache,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.mapping_cache_file, 'wb') as f:
                pickle.dump(mapping_data, f)
            
            # Save metadata
            with open(self.metadata_cache_file, 'wb') as f:
                pickle.dump(self.file_metadata_cache, f)
            
            logger.info(f"💾 Cached {len(self.uuid_to_sample_cache)} mappings and {len(self.file_metadata_cache)} metadata entries")
            
        except Exception as e:
            logger.error(f"❌ Failed to save caches: {e}")
    
    def extract_uuid_from_filename(self, filename: str) -> Optional[str]:
        """Extract UUID from filename if present"""
        # Standard UUID pattern: 8-4-4-4-12 hexadecimal digits
        uuid_pattern = r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
        match = re.search(uuid_pattern, filename.lower())
        return match.group(1) if match else None
    
    def extract_tcga_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract TCGA sample ID from filename if present"""
        # Standard TCGA sample ID pattern
        tcga_pattern = r'(TCGA-\w{2}-\w{4}-\w{2}[A-Z]-\w{2}[A-Z]-\w{4}-\w{2})'
        match = re.search(tcga_pattern, filename)
        if match:
            return match.group(1)
        
        # Shorter TCGA pattern for project-level matching
        short_pattern = r'(TCGA-\w{2}-\w{4})'
        match = re.search(short_pattern, filename)
        if match:
            return match.group(1)
        
        return None
    
    def query_file_metadata_batch(self, file_uuids: List[str]) -> Dict[str, Dict]:
        """Query GDC API for file metadata in batches"""
        if not file_uuids:
            return {}
        
        # Remove already cached UUIDs
        new_uuids = [uuid for uuid in file_uuids if uuid not in self.file_metadata_cache]
        
        if not new_uuids:
            logger.info(f"📊 All {len(file_uuids)} UUIDs already cached")
            return {uuid: self.file_metadata_cache[uuid] for uuid in file_uuids if uuid in self.file_metadata_cache}
        
        logger.info(f"🔍 Querying metadata for {len(new_uuids)} new UUIDs...")
        
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
            
            time.sleep(self.request_delay)  # Rate limiting
            response = requests.get(self.files_endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('data', [])
            
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
                    'sample_ids': list(set(sample_ids)),  # Remove duplicates
                    'data_category': file_info.get('data_category', ''),
                    'data_type': file_info.get('data_type', ''),
                    'primary_sample_id': sample_ids[0] if sample_ids else None
                }
                
                new_metadata[file_id] = metadata
                self.file_metadata_cache[file_id] = metadata
            
            logger.info(f"✅ Retrieved metadata for {len(new_metadata)} files")
            
            # Combine with existing cache
            all_metadata = {uuid: self.file_metadata_cache[uuid] for uuid in file_uuids if uuid in self.file_metadata_cache}
            
            return all_metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to query file metadata: {e}")
            return {}
    
    def build_sample_mappings(self, omics_type: str) -> Dict[str, str]:
        """Build comprehensive sample ID mappings for an omics type"""
        logger.info(f"🔗 Building sample mappings for {omics_type}...")
        
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
        
        # Separate files by ID type
        uuid_files = []
        tcga_files = []
        
        for file_path, project in all_files:
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
        
        logger.info(f"  📋 UUID files: {len(uuid_files)}")
        logger.info(f"  📋 TCGA ID files: {len(tcga_files)}")
        
        # Query metadata for UUID files
        uuid_mapping = {}
        if uuid_files:
            uuids = [item[0] for item in uuid_files]
            metadata_results = self.query_file_metadata_batch(uuids)
            
            for uuid, filename, project in uuid_files:
                if uuid in metadata_results:
                    metadata = metadata_results[uuid]
                    primary_sample = metadata['primary_sample_id']
                    if primary_sample:
                        uuid_mapping[filename] = primary_sample
                        # Also map by UUID
                        uuid_mapping[uuid] = primary_sample
                        
                        # Update caches
                        self.uuid_to_sample_cache[uuid] = primary_sample
                        if primary_sample not in self.sample_to_uuid_cache:
                            self.sample_to_uuid_cache[primary_sample] = []
                        self.sample_to_uuid_cache[primary_sample].append(uuid)
        
        # Direct mapping for TCGA ID files
        tcga_mapping = {}
        for tcga_id, filename, project in tcga_files:
            tcga_mapping[filename] = tcga_id
        
        # Combine mappings
        all_mapping = {**uuid_mapping, **tcga_mapping}
        
        logger.info(f"✅ Created {len(all_mapping)} sample mappings for {omics_type}")
        
        return all_mapping
    
    def create_comprehensive_mapping(self) -> Dict[str, Dict[str, str]]:
        """Create comprehensive mapping across all omics types"""
        logger.info("🚀 Creating comprehensive cross-omics sample mapping...")
        
        omics_types = ['mutations', 'expression', 'copy_number', 'methylation', 'protein']
        all_mappings = {}
        
        for omics_type in omics_types:
            if (self.base_dir / omics_type).exists():
                mapping = self.build_sample_mappings(omics_type)
                all_mappings[omics_type] = mapping
        
        # Save caches
        self.save_caches()
        
        # Create cross-reference mapping
        logger.info("🔗 Creating cross-omics sample matrix...")
        
        # Collect all unique sample IDs
        all_samples = set()
        for omics_mapping in all_mappings.values():
            all_samples.update(omics_mapping.values())
        
        logger.info(f"📊 Found {len(all_samples)} unique samples across all omics")
        
        # Create sample-centric mapping
        sample_matrix = {}
        for sample_id in all_samples:
            sample_files = {}
            for omics_type, mapping in all_mappings.items():
                files_for_sample = [filename for filename, sid in mapping.items() if sid == sample_id]
                if files_for_sample:
                    sample_files[omics_type] = files_for_sample
            
            if len(sample_files) > 1:  # Only samples with multiple omics
                sample_matrix[sample_id] = sample_files
        
        logger.info(f"🎯 Multi-omics samples: {len(sample_matrix)}")
        
        # Save comprehensive mapping
        comprehensive_mapping = {
            'omics_mappings': all_mappings,
            'sample_matrix': sample_matrix,
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_samples': len(all_samples),
                'multi_omics_samples': len(sample_matrix),
                'omics_types': list(all_mappings.keys())
            }
        }
        
        mapping_file = self.cache_dir / "comprehensive_sample_mapping.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(comprehensive_mapping, f)
        
        logger.info(f"💾 Comprehensive mapping saved to {mapping_file}")
        
        return comprehensive_mapping
    
    def get_sample_mapping(self) -> Dict[str, Dict[str, str]]:
        """Get or create comprehensive sample mapping"""
        mapping_file = self.cache_dir / "comprehensive_sample_mapping.pkl"
        
        if mapping_file.exists():
            logger.info("📥 Loading existing comprehensive mapping...")
            try:
                with open(mapping_file, 'rb') as f:
                    mapping = pickle.load(f)
                
                metadata = mapping.get('metadata', {})
                logger.info(f"✅ Loaded mapping: {metadata.get('total_samples', 0)} total samples, {metadata.get('multi_omics_samples', 0)} multi-omics")
                
                return mapping
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load mapping: {e}")
        
        # Create new mapping
        return self.create_comprehensive_mapping()


def main():
    """Main execution function"""
    logger.info("🔗 TCGA Sample ID Mapping System")
    logger.info("=" * 60)
    
    # Initialize mapper
    mapper = TCGASampleMapper()
    
    try:
        # Create comprehensive mapping
        mapping = mapper.get_sample_mapping()
        
        # Show results
        metadata = mapping.get('metadata', {})
        sample_matrix = mapping.get('sample_matrix', {})
        
        logger.info("📊 Mapping Summary:")
        logger.info(f"  Total unique samples: {metadata.get('total_samples', 0)}")
        logger.info(f"  Multi-omics samples: {metadata.get('multi_omics_samples', 0)}")
        logger.info(f"  Omics types: {len(metadata.get('omics_types', []))}")
        
        # Show sample distribution by omics coverage
        coverage_stats = {}
        for sample_id, omics_files in sample_matrix.items():
            coverage = len(omics_files)
            coverage_stats[coverage] = coverage_stats.get(coverage, 0) + 1
        
        logger.info("🎯 Multi-omics coverage:")
        for coverage, count in sorted(coverage_stats.items(), reverse=True):
            logger.info(f"  {coverage} omics: {count} samples")
        
        logger.info("✅ SUCCESS: Comprehensive sample mapping created!")
        
        return mapping
        
    except KeyboardInterrupt:
        logger.info("⏸️ Mapping interrupted by user")
        return None
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise


if __name__ == "__main__":
    mapping = main()
