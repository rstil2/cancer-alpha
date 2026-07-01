#!/usr/bin/env python3
"""
Oncura GDC API Sample Mapping System
===================================

Production-grade UUID-to-TCGA ID mapping system for scaling to 50,000+ samples.
Integrates with the Genomic Data Commons (GDC) API for accurate sample identification
and comprehensive metadata retrieval.

Features:
- Robust UUID resolution with intelligent batching
- Comprehensive metadata extraction (sample type, cancer type, clinical data)
- Advanced caching and persistence for large-scale operations  
- Rate limiting and error recovery for API reliability
- Multi-threaded processing with progress tracking
- Sample quality assessment and filtering

ONCURA PROJECT - Next-Generation Cancer Genomics Platform
"""

import requests
import json
import time
import logging
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import threading
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Advanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oncura_gdc_mapping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OncuraGDC')

class SampleType(Enum):
    PRIMARY_TUMOR = "Primary Tumor"
    SOLID_TISSUE_NORMAL = "Solid Tissue Normal" 
    BLOOD_DERIVED_NORMAL = "Blood Derived Normal"
    METASTATIC = "Metastatic"
    RECURRENT_TUMOR = "Recurrent Tumor"
    UNKNOWN = "Unknown"

class DataCategory(Enum):
    TRANSCRIPTOME_PROFILING = "Transcriptome Profiling"
    SIMPLE_NUCLEOTIDE_VARIATION = "Simple Nucleotide Variation"  
    COPY_NUMBER_VARIATION = "Copy Number Variation"
    DNA_METHYLATION = "DNA Methylation"
    PROTEOME_PROFILING = "Proteome Profiling"
    CLINICAL = "Clinical"
    BIOSPECIMEN = "Biospecimen"

@dataclass
class SampleMetadata:
    """Comprehensive sample metadata structure"""
    file_uuid: str
    file_name: str
    tcga_barcode: str
    submitter_id: str
    sample_type: SampleType
    cancer_type: str
    project_id: str
    data_category: DataCategory
    data_type: str
    experimental_strategy: str
    platform: str
    file_size: int
    md5sum: str
    created_datetime: str
    updated_datetime: str
    case_uuid: str
    case_submitter_id: str
    sample_uuid: str
    aliquot_uuid: Optional[str] = None
    portion_uuid: Optional[str] = None
    analyte_uuid: Optional[str] = None
    clinical_data: Optional[Dict] = None
    quality_score: float = 0.0

class OncuraGDCMapper:
    """Production-grade GDC API mapper for Oncura platform"""
    
    def __init__(self, 
                 cache_dir: str = "data/oncura_cache",
                 db_path: str = "data/oncura_cache/sample_mapping.db",
                 batch_size: int = 100,
                 max_workers: int = 4,
                 rate_limit_delay: float = 0.5):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        
        # GDC API configuration
        self.gdc_base_url = "https://api.gdc.cancer.gov"
        self.files_endpoint = f"{self.gdc_base_url}/files"
        self.cases_endpoint = f"{self.gdc_base_url}/cases"
        
        # Setup robust session with retries
        self.session = self._create_robust_session()
        
        # Initialize database
        self._init_database()
        
        # Caching and statistics
        self.cache_hits = 0
        self.api_calls = 0
        self.mapping_stats = defaultdict(int)
        
        logger.info("🚀 Oncura GDC Mapper initialized")
        logger.info(f"📊 Configuration: batch_size={batch_size}, workers={max_workers}, rate_limit={rate_limit_delay}s")
    
    def _create_robust_session(self) -> requests.Session:
        """Create a robust requests session with retry logic"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Default headers
        session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Oncura/1.0 (Cancer Genomics Platform)'
        })
        
        return session
    
    def _init_database(self):
        """Initialize SQLite database for persistent caching"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main sample mapping table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sample_mappings (
                    file_uuid TEXT PRIMARY KEY,
                    file_name TEXT,
                    tcga_barcode TEXT,
                    submitter_id TEXT,
                    sample_type TEXT,
                    cancer_type TEXT,
                    project_id TEXT,
                    data_category TEXT,
                    data_type TEXT,
                    experimental_strategy TEXT,
                    platform TEXT,
                    file_size INTEGER,
                    md5sum TEXT,
                    created_datetime TEXT,
                    updated_datetime TEXT,
                    case_uuid TEXT,
                    case_submitter_id TEXT,
                    sample_uuid TEXT,
                    aliquot_uuid TEXT,
                    portion_uuid TEXT,
                    analyte_uuid TEXT,
                    quality_score REAL,
                    cached_datetime TEXT
                )
            ''')
            
            # Create indexes separately
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tcga_barcode ON sample_mappings(tcga_barcode)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cancer_type ON sample_mappings(cancer_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_category ON sample_mappings(data_category)')
            
            # Clinical data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clinical_data (
                    case_uuid TEXT PRIMARY KEY,
                    case_submitter_id TEXT,
                    age_at_diagnosis INTEGER,
                    gender TEXT,
                    race TEXT,
                    ethnicity TEXT,
                    vital_status TEXT,
                    days_to_death INTEGER,
                    days_to_last_follow_up INTEGER,
                    tumor_stage TEXT,
                    tumor_grade TEXT,
                    histological_type TEXT,
                    primary_site TEXT,
                    cached_datetime TEXT
                )
            ''')
            
            # Create indexes for clinical data
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_case_submitter_id ON clinical_data(case_submitter_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_vital_status ON clinical_data(vital_status)')
            
            # API usage statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_stats (
                    date TEXT PRIMARY KEY,
                    total_calls INTEGER,
                    cache_hits INTEGER,
                    successful_mappings INTEGER,
                    failed_mappings INTEGER,
                    avg_response_time REAL
                )
            ''')
            
            conn.commit()
    
    def get_cached_mapping(self, file_uuid: str) -> Optional[SampleMetadata]:
        """Retrieve cached sample mapping from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM sample_mappings WHERE file_uuid = ?
            ''', (file_uuid,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            self.cache_hits += 1
            
            # Convert row to SampleMetadata
            return SampleMetadata(
                file_uuid=row[0],
                file_name=row[1],
                tcga_barcode=row[2],
                submitter_id=row[3],
                sample_type=SampleType(row[4]) if row[4] else SampleType.UNKNOWN,
                cancer_type=row[5],
                project_id=row[6],
                data_category=DataCategory(row[7]) if row[7] else DataCategory.CLINICAL,
                data_type=row[8],
                experimental_strategy=row[9],
                platform=row[10],
                file_size=row[11],
                md5sum=row[12],
                created_datetime=row[13],
                updated_datetime=row[14],
                case_uuid=row[15],
                case_submitter_id=row[16],
                sample_uuid=row[17],
                aliquot_uuid=row[18],
                portion_uuid=row[19],
                analyte_uuid=row[20],
                quality_score=row[21] or 0.0
            )
    
    def cache_mapping(self, metadata: SampleMetadata):
        """Cache sample mapping in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO sample_mappings VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.file_uuid,
                metadata.file_name,
                metadata.tcga_barcode,
                metadata.submitter_id,
                metadata.sample_type.value,
                metadata.cancer_type,
                metadata.project_id,
                metadata.data_category.value,
                metadata.data_type,
                metadata.experimental_strategy,
                metadata.platform,
                metadata.file_size,
                metadata.md5sum,
                metadata.created_datetime,
                metadata.updated_datetime,
                metadata.case_uuid,
                metadata.case_submitter_id,
                metadata.sample_uuid,
                metadata.aliquot_uuid,
                metadata.portion_uuid,
                metadata.analyte_uuid,
                metadata.quality_score,
                datetime.now().isoformat()
            ))
            
            conn.commit()
    
    def query_gdc_files_batch(self, file_uuids: List[str]) -> Dict[str, SampleMetadata]:
        """Query GDC API for batch of file UUIDs"""
        if not file_uuids:
            return {}
        
        # Build comprehensive query for file and sample metadata
        filters = {
            "op": "in",
            "content": {
                "field": "files.id",
                "value": file_uuids
            }
        }
        
        # Comprehensive fields for complete metadata extraction
        fields = [
            "files.id",
            "files.file_name", 
            "files.file_size",
            "files.md5sum",
            "files.data_category",
            "files.data_type",
            "files.experimental_strategy",
            "files.platform",
            "files.created_datetime",
            "files.updated_datetime",
            "files.cases.id",
            "files.cases.submitter_id",
            "files.cases.project.project_id",
            "files.cases.project.primary_site",
            "files.cases.samples.id",
            "files.cases.samples.submitter_id",
            "files.cases.samples.sample_type",
            "files.cases.samples.portions.id",
            "files.cases.samples.portions.analytes.id",
            "files.cases.samples.portions.analytes.aliquots.id",
            "files.cases.samples.portions.analytes.aliquots.submitter_id"
        ]
        
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "json",
            "size": str(len(file_uuids))
        }
        
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(self.files_endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            self.api_calls += 1
            data = response.json()
            
            return self._process_gdc_response(data)
            
        except Exception as e:
            logger.error(f"GDC API batch query failed: {str(e)}")
            return {}
    
    def _process_gdc_response(self, data: Dict) -> Dict[str, SampleMetadata]:
        """Process GDC API response into SampleMetadata objects"""
        mappings = {}
        
        if 'data' not in data or 'hits' not in data['data']:
            return mappings
        
        for file_data in data['data']['hits']:
            try:
                file_uuid = file_data.get('id', '')
                file_name = file_data.get('file_name', '')
                
                # Extract case and sample information
                cases = file_data.get('cases', [])
                if not cases:
                    continue
                
                case = cases[0]  # Take first case
                case_uuid = case.get('id', '')
                case_submitter_id = case.get('submitter_id', '')
                
                # Project information
                project = case.get('project', {})
                project_id = project.get('project_id', '')
                primary_site = project.get('primary_site', '')
                
                # Extract TCGA cancer type from project ID
                cancer_type = project_id.replace('TCGA-', '') if project_id.startswith('TCGA-') else 'Unknown'
                
                # Sample information
                samples = case.get('samples', [])
                sample_info = {}
                if samples:
                    sample = samples[0]  # Take first sample
                    sample_info = {
                        'sample_uuid': sample.get('id', ''),
                        'sample_submitter_id': sample.get('submitter_id', ''),
                        'sample_type': sample.get('sample_type', 'Unknown')
                    }
                    
                    # Extract aliquot information for TCGA barcode
                    portions = sample.get('portions', [])
                    if portions:
                        portion = portions[0]
                        sample_info['portion_uuid'] = portion.get('id', '')
                        
                        analytes = portion.get('analytes', [])
                        if analytes:
                            analyte = analytes[0]
                            sample_info['analyte_uuid'] = analyte.get('id', '')
                            
                            aliquots = analyte.get('aliquots', [])
                            if aliquots:
                                aliquot = aliquots[0]
                                sample_info['aliquot_uuid'] = aliquot.get('id', '')
                                # Use aliquot submitter_id as TCGA barcode
                                sample_info['tcga_barcode'] = aliquot.get('submitter_id', case_submitter_id)
                
                # Determine data category
                data_category_str = file_data.get('data_category', 'Clinical')
                try:
                    data_category = DataCategory(data_category_str)
                except ValueError:
                    data_category = DataCategory.CLINICAL
                
                # Determine sample type
                sample_type_str = sample_info.get('sample_type', 'Unknown')
                try:
                    sample_type = SampleType(sample_type_str)
                except ValueError:
                    sample_type = SampleType.UNKNOWN
                
                # Calculate quality score based on completeness
                quality_score = self._calculate_quality_score(file_data, sample_info)
                
                # Create metadata object
                metadata = SampleMetadata(
                    file_uuid=file_uuid,
                    file_name=file_name,
                    tcga_barcode=sample_info.get('tcga_barcode', case_submitter_id),
                    submitter_id=case_submitter_id,
                    sample_type=sample_type,
                    cancer_type=cancer_type,
                    project_id=project_id,
                    data_category=data_category,
                    data_type=file_data.get('data_type', ''),
                    experimental_strategy=file_data.get('experimental_strategy', ''),
                    platform=file_data.get('platform', ''),
                    file_size=file_data.get('file_size', 0),
                    md5sum=file_data.get('md5sum', ''),
                    created_datetime=file_data.get('created_datetime', ''),
                    updated_datetime=file_data.get('updated_datetime', ''),
                    case_uuid=case_uuid,
                    case_submitter_id=case_submitter_id,
                    sample_uuid=sample_info.get('sample_uuid', ''),
                    aliquot_uuid=sample_info.get('aliquot_uuid'),
                    portion_uuid=sample_info.get('portion_uuid'),
                    analyte_uuid=sample_info.get('analyte_uuid'),
                    quality_score=quality_score
                )
                
                mappings[file_uuid] = metadata
                self.mapping_stats['successful'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to process file data: {str(e)}")
                self.mapping_stats['failed'] += 1
                continue
        
        return mappings
    
    def _calculate_quality_score(self, file_data: Dict, sample_info: Dict) -> float:
        """Calculate sample quality score based on metadata completeness"""
        score = 0.0
        max_score = 10.0
        
        # File completeness (2 points)
        if file_data.get('file_size', 0) > 0:
            score += 1.0
        if file_data.get('md5sum'):
            score += 1.0
        
        # Sample information completeness (4 points)
        if sample_info.get('tcga_barcode'):
            score += 2.0
        if sample_info.get('sample_type', 'Unknown') != 'Unknown':
            score += 1.0
        if sample_info.get('sample_uuid'):
            score += 1.0
        
        # Data type information (2 points)
        if file_data.get('data_category'):
            score += 1.0
        if file_data.get('experimental_strategy'):
            score += 1.0
        
        # Hierarchical sample information (2 points)
        if sample_info.get('aliquot_uuid'):
            score += 1.0
        if sample_info.get('portion_uuid'):
            score += 1.0
        
        return (score / max_score) * 100.0
    
    def map_file_uuids(self, file_uuids: List[str], use_cache: bool = True) -> Dict[str, SampleMetadata]:
        """Map file UUIDs to comprehensive sample metadata"""
        logger.info(f"🔍 Starting mapping for {len(file_uuids):,} file UUIDs")
        
        all_mappings = {}
        uncached_uuids = []
        
        # Check cache first if enabled
        if use_cache:
            for uuid in file_uuids:
                cached = self.get_cached_mapping(uuid)
                if cached:
                    all_mappings[uuid] = cached
                else:
                    uncached_uuids.append(uuid)
            
            logger.info(f"📊 Cache stats: {len(all_mappings):,} hits, {len(uncached_uuids):,} misses")
        else:
            uncached_uuids = file_uuids
        
        if not uncached_uuids:
            return all_mappings
        
        # Process uncached UUIDs in batches
        logger.info(f"🌐 Querying GDC API for {len(uncached_uuids):,} UUIDs in batches of {self.batch_size}")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(0, len(uncached_uuids), self.batch_size):
                batch = uncached_uuids[i:i + self.batch_size]
                future = executor.submit(self.query_gdc_files_batch, batch)
                futures.append(future)
            
            completed = 0
            for future in as_completed(futures):
                try:
                    batch_mappings = future.result()
                    
                    # Cache new mappings
                    for uuid, metadata in batch_mappings.items():
                        self.cache_mapping(metadata)
                        all_mappings[uuid] = metadata
                    
                    completed += 1
                    progress = (completed / len(futures)) * 100
                    logger.info(f"📈 Progress: {completed}/{len(futures)} batches ({progress:.1f}%) - "
                              f"Mapped: {len(batch_mappings):,} samples")
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")
        
        logger.info(f"✅ Mapping complete: {len(all_mappings):,} successful mappings")
        logger.info(f"📊 API calls: {self.api_calls}, Cache hits: {self.cache_hits}")
        
        return all_mappings
    
    def discover_and_map_files(self, data_dirs: List[str]) -> pd.DataFrame:
        """Discover files in data directories and map to TCGA samples"""
        logger.info("🔍 Starting comprehensive file discovery and mapping")
        
        # Extract UUIDs from filenames
        discovered_files = []
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists():
                logger.warning(f"⚠️ Directory not found: {data_dir}")
                continue
            
            logger.info(f"📂 Scanning {data_dir}...")
            
            # Find all data files
            patterns = ['**/*.maf*', '**/*.tsv*', '**/*.txt*', '**/*.gz', '**/*.json']
            for pattern in patterns:
                files = list(data_path.glob(pattern))
                
                for file_path in files:
                    if file_path.is_file() and file_path.stat().st_size > 0:
                        # Try to extract UUID from filename
                        uuid = self._extract_uuid_from_filename(file_path.name)
                        if uuid:
                            discovered_files.append({
                                'file_path': str(file_path),
                                'file_uuid': uuid,
                                'file_size': file_path.stat().st_size,
                                'file_name': file_path.name
                            })
        
        logger.info(f"🎯 Discovered {len(discovered_files):,} files with extractable UUIDs")
        
        if not discovered_files:
            logger.warning("No files with UUIDs found for mapping")
            return pd.DataFrame()
        
        # Extract unique UUIDs for mapping
        unique_uuids = list(set(f['file_uuid'] for f in discovered_files))
        logger.info(f"📊 Unique UUIDs to map: {len(unique_uuids):,}")
        
        # Map UUIDs to sample metadata
        mappings = self.map_file_uuids(unique_uuids)
        
        # Combine discovered files with mappings
        enriched_files = []
        
        for file_info in discovered_files:
            uuid = file_info['file_uuid']
            metadata = mappings.get(uuid)
            
            if metadata:
                enriched_files.append({
                    **file_info,
                    'tcga_barcode': metadata.tcga_barcode,
                    'cancer_type': metadata.cancer_type,
                    'sample_type': metadata.sample_type.value,
                    'data_category': metadata.data_category.value,
                    'data_type': metadata.data_type,
                    'experimental_strategy': metadata.experimental_strategy,
                    'platform': metadata.platform,
                    'project_id': metadata.project_id,
                    'quality_score': metadata.quality_score,
                    'case_uuid': metadata.case_uuid,
                    'sample_uuid': metadata.sample_uuid
                })
        
        df = pd.DataFrame(enriched_files)
        
        if len(df) > 0:
            logger.info("🎉 File discovery and mapping complete!")
            logger.info(f"📊 Successfully mapped files:")
            logger.info(f"   Total files: {len(df):,}")
            logger.info(f"   Cancer types: {df['cancer_type'].nunique()}")
            logger.info(f"   Data categories: {df['data_category'].nunique()}")
            logger.info(f"   Average quality score: {df['quality_score'].mean():.1f}")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.cache_dir / f"oncura_file_mapping_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Results saved to: {output_file}")
        
        return df
    
    def _extract_uuid_from_filename(self, filename: str) -> Optional[str]:
        """Extract UUID from various TCGA filename formats"""
        import re
        
        # Standard UUID pattern (8-4-4-4-12 format)
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        match = re.search(uuid_pattern, filename.lower())
        return match.group(0) if match else None
    
    def get_mapping_statistics(self) -> Dict:
        """Get comprehensive mapping statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM sample_mappings")
            total_mappings = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT cancer_type) FROM sample_mappings")
            cancer_types = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT data_category) FROM sample_mappings")  
            data_categories = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(quality_score) FROM sample_mappings")
            avg_quality = cursor.fetchone()[0] or 0.0
            
            # Cancer type distribution
            cursor.execute("""
                SELECT cancer_type, COUNT(*) as count 
                FROM sample_mappings 
                GROUP BY cancer_type 
                ORDER BY count DESC 
                LIMIT 10
            """)
            top_cancers = dict(cursor.fetchall())
            
            # Data category distribution  
            cursor.execute("""
                SELECT data_category, COUNT(*) as count
                FROM sample_mappings
                GROUP BY data_category
                ORDER BY count DESC
            """)
            data_category_dist = dict(cursor.fetchall())
            
            return {
                'total_mappings': total_mappings,
                'unique_cancer_types': cancer_types,
                'data_categories': data_categories,
                'average_quality_score': avg_quality,
                'api_calls_made': self.api_calls,
                'cache_hits': self.cache_hits,
                'top_cancer_types': top_cancers,
                'data_category_distribution': data_category_dist,
                'current_stats': dict(self.mapping_stats)
            }

def main():
    """Main execution for Oncura GDC mapping system"""
    print("🧬 ONCURA - Next-Generation Cancer Genomics Platform")
    print("🔍 GDC API Sample Mapping System")
    print("=" * 60)
    
    # Data directories to scan
    data_directories = [
        "data/production_tcga",
        "data/tcga_ultra_massive", 
        "data/tcga_real_fixed"
    ]
    
    print(f"📂 Scanning data directories: {len(data_directories)}")
    for dir_path in data_directories:
        print(f"   {dir_path}")
    print()
    
    # Initialize mapper
    mapper = OncuraGDCMapper(
        cache_dir="data/oncura_cache",
        batch_size=50,  # Conservative batch size for API stability
        max_workers=2,  # Conservative threading for API limits
        rate_limit_delay=1.0  # 1 second between API calls
    )
    
    # Discover and map files
    results_df = mapper.discover_and_map_files(data_directories)
    
    if len(results_df) > 0:
        print("\n🎉 ONCURA MAPPING RESULTS")
        print("=" * 40)
        
        # Display key statistics
        stats = mapper.get_mapping_statistics()
        
        print(f"📊 Total mapped files: {stats['total_mappings']:,}")
        print(f"🧬 Cancer types: {stats['unique_cancer_types']}")
        print(f"📈 Data categories: {stats['data_categories']}")
        print(f"⭐ Average quality score: {stats['average_quality_score']:.1f}%")
        print(f"🌐 API calls made: {stats['api_calls_made']:,}")
        print(f"💾 Cache hits: {stats['cache_hits']:,}")
        
        print(f"\n🔝 Top Cancer Types:")
        for cancer, count in list(stats['top_cancer_types'].items())[:5]:
            print(f"   {cancer}: {count:,} files")
        
        print(f"\n📋 Data Categories:")
        for category, count in stats['data_category_distribution'].items():
            print(f"   {category}: {count:,} files")
        
        print(f"\n✅ Ready to scale to 50,000+ samples with comprehensive metadata!")
        
    else:
        print("⚠️ No mappable files found. Check data directory structure and file formats.")
    
    return mapper, results_df

if __name__ == "__main__":
    mapper, results = main()
