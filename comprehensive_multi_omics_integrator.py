#!/usr/bin/env python3
"""
Oncura Comprehensive Multi-Omics Integration Processor (50K+ Samples)
======================================================================

Integrates the ultra-massive TCGA dataset (50,000+ samples) into a unified,
production-grade multi-omics dataset for advanced AI model training.

- Scans multiple data directories for comprehensive data discovery.
- Uses robust filename-based sample ID extraction for all omics types.
- **Handles complex filename formats, including UUIDs and other variations.**
- Integrates data in a memory-efficient, scalable manner.
- Creates a unified dataset ready for advanced model training.

"""

import os
import re
import gzip
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveMultiOmicsIntegrator:
    """Integrates the ultra-massive TCGA dataset for AI model training."""

    def __init__(self, data_dirs, output_dir, max_workers=8):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # Comprehensive mapping of data types to their directory names
        self.data_type_map = {
            'expression': 'Gene Expression Quantification',
            'mutations': 'Masked Somatic Mutation',
            'copy_number': 'Copy Number Segment',
            'methylation': 'Methylation Beta Value',
            'protein': 'Protein Expression Quantification',
            'clinical': 'Clinical Supplement',
            'mirna': 'miRNA Expression Quantification'
        }

    def extract_sample_id(self, filename, file_path):
        """Extracts TCGA sample ID from various filename formats."""
        # Standard TCGA barcode format (e.g., TCGA-A1-A0SP-01A)
        match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', str(filename))
        if match:
            return match.group(1)

        # UUID-based filenames (requires reading file content or metadata)
        if '.maf.gz' in str(filename):
            # Extract from MAF file content
            try:
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        # The TCGA barcode is often in the first few columns
                        parts = line.split('\t')
                        for part in parts:
                            if re.match(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', part):
                                return part.strip()
            except Exception as e:
                logger.warning(f"Could not extract sample ID from MAF file {filename}: {e}")
                return None
        
        # For other file types, we'll rely on the parent directory structure
        # This assumes a structure like: .../TCGA-BRCA/Gene Expression Quantification/file.tsv
        parts = file_path.parts
        for part in reversed(parts):
            if part.startswith('TCGA-'):
                return part
        
        return None

    def discover_files_for_cancer_type(self, cancer_type):
        """Discovers all files for a given cancer type across all data directories."""
        files_by_sample = defaultdict(lambda: defaultdict(list))
        file_count = 0

        for data_dir in self.data_dirs:
            for data_type_key, data_type_name in self.data_type_map.items():
                cancer_data_dir = data_dir / cancer_type / data_type_name
                if cancer_data_dir.exists():
                    for file_path in cancer_data_dir.glob('*'):
                        sample_id = self.extract_sample_id(file_path.name, file_path)
                        if sample_id:
                            files_by_sample[sample_id][data_type_key].append(file_path)
                            file_count += 1
        
        return files_by_sample, file_count

    def process_cancer_type(self, cancer_type):
        """Processes a single cancer type and returns a DataFrame."""
        logger.info(f"Processing cancer type: {cancer_type}...")
        files_by_sample, file_count = self.discover_files_for_cancer_type(cancer_type)
        logger.info(f"Found {file_count} files for {len(files_by_sample)} samples in {cancer_type}.")

        if not files_by_sample:
            return None

        # Create a DataFrame for this cancer type
        records = []
        for sample_id, data_types in files_by_sample.items():
            record = {'sample_id': sample_id, 'cancer_type': cancer_type}
            for data_type, file_paths in data_types.items():
                record[data_type] = ';'.join(str(p) for p in file_paths)
            records.append(record)
        
        return pd.DataFrame(records)

    def run_integration(self):
        """Runs the full multi-omics integration process."""
        logger.info("Starting comprehensive multi-omics integration...")

        all_cancer_types = set()
        for data_dir in self.data_dirs:
            if data_dir.exists():
                for cancer_dir in data_dir.iterdir():
                    if cancer_dir.is_dir() and cancer_dir.name.startswith('TCGA-'):
                        all_cancer_types.add(cancer_dir.name)
        
        logger.info(f"Found {len(all_cancer_types)} unique cancer types to process.")

        all_dfs = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cancer = {executor.submit(self.process_cancer_type, ct): ct for ct in all_cancer_types}
            
            for future in tqdm(as_completed(future_to_cancer), total=len(all_cancer_types), desc="Integrating Cancer Types"):
                try:
                    df = future.result()
                    if df is not None:
                        all_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error processing {future_to_cancer[future]}: {e}")

        if not all_dfs:
            logger.warning("No data was integrated. Please check your data directories.")
            return

        # Concatenate all DataFrames
        integrated_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Successfully integrated data for {len(integrated_df)} samples.")

        # Save the integrated dataset
        output_path = self.output_dir / "oncura_comprehensive_multi_omics_50k.csv"
        integrated_df.to_csv(output_path, index=False)
        logger.info(f"Integrated dataset saved to {output_path}")

        # Save a summary of the dataset
        summary = {
            "total_samples": len(integrated_df),
            "cancer_type_counts": integrated_df['cancer_type'].value_counts().apply(int).to_dict(),
            "data_type_coverage": {
                dt: int(integrated_df[dt].notna().sum()) if dt in integrated_df.columns else 0 for dt in self.data_type_map.keys()
            }
        }

        summary_path = self.output_dir / "oncura_comprehensive_integration_summary_50k.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Integration summary saved to {summary_path}")

if __name__ == '__main__':
    data_directories = [
        '/Users/stillwell/projects/cancer-alpha/data/production_tcga',
        '/Users/stillwell/projects/cancer-alpha/data/tcga_ultra_massive_50k'
    ]
    output_directory = '/Users/stillwell/projects/cancer-alpha/data/processed_50k'

    integrator = ComprehensiveMultiOmicsIntegrator(data_directories, output_directory)
    integrator.run_integration()
