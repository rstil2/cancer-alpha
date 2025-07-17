#!/usr/bin/env python3
"""
Preprocessing Script for Cancer Genomics Project
Processes TCGA, GEO, and ENCODE datasets for methylation and fragmentomics analysis
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

class DataPreprocessor:
    """Class to preprocess cancer genomics data"""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_tcga_methylation(self):
        """Preprocess TCGA methylation data"""
        print("Preprocessing TCGA methylation data...")
        methyl_file = self.raw_data_dir / "tcga_methylation_files.json"
        
        with open(methyl_file, 'r') as f:
            methyl_data = json.load(f)
        
        methyl_records = []
        hits = methyl_data.get('data', {}).get('hits', [])
        for entry in hits:
            # Extract relevant fields
            methyl_records.append({
                "file_id": entry.get("file_id"),
                "file_name": entry.get("file_name"),
                "data_type": entry.get("data_type"),
                "platform": entry.get("platform"),
                "file_size": entry.get("file_size"),
                "experimental_strategy": entry.get("experimental_strategy")
            })
        
        # Convert to DataFrame
        methyl_df = pd.DataFrame(methyl_records)
        # Save processed file
        output_file = self.processed_data_dir / "tcga_methylation_processed.csv"
        methyl_df.to_csv(output_file, index=False)
        
        print(f"TCGA methylation data processed and saved to: {output_file}")
    
    def preprocess_geo_datasets(self):
        """Preprocess GEO cfMeDIP and specific series datasets"""
        print("Preprocessing GEO datasets...")
        
        cfmedip_file = self.raw_data_dir / "geo_cfmedip_results.json"
        specific_series_file = self.raw_data_dir / "geo_specific_series.json"
        
        with open(cfmedip_file, 'r') as f:
            cfmedip_data = json.load(f)
        
        with open(specific_series_file, 'r') as f:
            series_data = json.load(f)
        
        # Extract specific details
        cfmedip_records = []
        for entry in cfmedip_data:
            cfmedip_records.append(entry)
        
        # Convert to DataFrame
        cfmedip_df = pd.DataFrame(cfmedip_records)
        # Save processed file
        cfmedip_output_file = self.processed_data_dir / "geo_cfmedip_processed.csv"
        cfmedip_df.to_csv(cfmedip_output_file, index=False)
        
        # Process specific series
        series_records = []
        for series_id, details in series_data.items():
            if 'result' in details and details['result']:
                # Handle the nested structure properly
                result_data = details['result']
                if isinstance(result_data, dict) and 'uids' in result_data:
                    entry = result_data.get('uids', [{}])[0] if result_data.get('uids') else {}
                else:
                    entry = result_data[0] if isinstance(result_data, list) else result_data
                
                series_records.append({
                    "series_id": series_id,
                    "title": entry.get("title", "Unknown"),
                    "summary": entry.get("summary", "No summary available"),
                    "samples": entry.get("sample_count", "Unknown")
                })
            else:
                # Handle cases where result structure is different
                series_records.append({
                    "series_id": series_id,
                    "title": "Data available", 
                    "summary": "Detailed parsing needed",
                    "samples": "Unknown"
                })
        
        # Convert to DataFrame
        series_df = pd.DataFrame(series_records)
        # Save processed file
        series_output_file = self.processed_data_dir / "geo_series_processed.csv"
        series_df.to_csv(series_output_file, index=False)
        
        print(f"GEO datasets processed and saved to: {cfmedip_output_file} and {series_output_file}")
    
    def preprocess_encode_data(self):
        """Preprocess ENCODE methylation and chromatin data"""
        print("Preprocessing ENCODE data...")
        encode_file = self.raw_data_dir / "encode_experiments_v2.json"
        
        with open(encode_file, 'r') as f:
            encode_data = json.load(f)
        
        encode_records = []
        for query_result in encode_data.values():
            for entry in query_result.get('@graph', []):
                encode_records.append({
                    "accession": entry.get("accession"),
                    "assay_title": entry.get("assay_title"),
                    "biosample_term_name": entry.get("biosample_term_name"),
                    "lab": entry.get("lab", {}).get("title"),
                    "files": len(entry.get("files", []))
                })
        
        # Convert to DataFrame
        encode_df = pd.DataFrame(encode_records)
        # Save processed file
        encode_output_file = self.processed_data_dir / "encode_processed.csv"
        encode_df.to_csv(encode_output_file, index=False)
        
        print(f"ENCODE data processed and saved to: {encode_output_file}")


def main():
    """Main function to run data preprocessing"""
    print("Starting data preprocessing...")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess datasets
    preprocessor.preprocess_tcga_methylation()
    preprocessor.preprocess_geo_datasets()
    preprocessor.preprocess_encode_data()
    
    print("\nData preprocessing complete!")
    print("Check the data/processed directory for processed data.")

if __name__ == "__main__":
    main()
