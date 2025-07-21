#!/usr/bin/env python3
"""
Improved Data Acquisition Script for Cancer Genomics Project
Fixed ENCODE and ICGC API endpoints and error handling
"""

import requests
import json
import pandas as pd
import os
from pathlib import Path
import time
from typing import Dict, List, Any
import urllib.parse

class ImprovedDataAcquisition:
    """Improved class to handle data acquisition from multiple genomics databases"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def search_encode_datasets_v2(self) -> Dict[str, Any]:
        """Improved ENCODE search with corrected API endpoints"""
        print("Searching ENCODE for DNA methylation and chromatin data...")
        
        base_url = "https://www.encodeproject.org"
        
        # Try multiple search strategies
        search_queries = [
            {
                "name": "methylation_experiments",
                "params": {
                    "type": "Experiment",
                    "assay_title": "whole-genome shotgun bisulfite sequencing",
                    "format": "json",
                    "limit": 25
                }
            },
            {
                "name": "chromatin_experiments", 
                "params": {
                    "type": "Experiment",
                    "assay_title": "ATAC-seq",
                    "format": "json",
                    "limit": 25
                }
            },
            {
                "name": "general_search",
                "params": {
                    "searchTerm": "DNA methylation",
                    "type": "Experiment",
                    "format": "json",
                    "limit": 25
                }
            }
        ]
        
        all_encode_results = {}
        
        for query in search_queries:
            search_url = f"{base_url}/search/"
            
            try:
                print(f"  Trying ENCODE query: {query['name']}")
                response = self.session.get(search_url, params=query["params"], timeout=30)
                
                print(f"  Response status: {response.status_code}")
                print(f"  Response URL: {response.url}")
                
                if response.status_code == 200:
                    try:
                        encode_data = response.json()
                        all_encode_results[query["name"]] = encode_data
                        
                        # Count results
                        result_count = len(encode_data.get('@graph', []))
                        print(f"  Found {result_count} results for {query['name']}")
                        
                    except json.JSONDecodeError as e:
                        print(f"  JSON decode error for {query['name']}: {e}")
                        # Save raw response for debugging
                        debug_file = self.data_dir / f"encode_debug_{query['name']}.txt"
                        with open(debug_file, 'w') as f:
                            f.write(response.text[:1000])  # First 1000 chars
                        
                else:
                    print(f"  HTTP {response.status_code} for {query['name']}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  Request error for {query['name']}: {e}")
        
        # Save all ENCODE results
        if all_encode_results:
            encode_file = self.data_dir / "encode_experiments_v2.json"
            with open(encode_file, 'w') as f:
                json.dump(all_encode_results, f, indent=2)
            print(f"ENCODE results saved to: {encode_file}")
        
        return all_encode_results
    
    def search_icgc_datasets_v2(self) -> Dict[str, Any]:
        """Improved ICGC search with better error handling"""
        print("Searching ICGC for lung cancer datasets...")
        
        # Try multiple ICGC endpoints
        icgc_endpoints = [
            {
                "name": "projects",
                "url": "https://dcc.icgc.org/api/v1/projects",
                "params": {"size": 100}
            },
            {
                "name": "donors", 
                "url": "https://dcc.icgc.org/api/v1/donors",
                "params": {
                    "filters": json.dumps({
                        "donor": {
                            "primarySite": {
                                "is": ["Lung"]
                            }
                        }
                    }),
                    "size": 50
                }
            }
        ]
        
        all_icgc_results = {}
        
        for endpoint in icgc_endpoints:
            try:
                print(f"  Trying ICGC endpoint: {endpoint['name']}")
                response = self.session.get(endpoint["url"], params=endpoint["params"], timeout=30)
                
                print(f"  Response status: {response.status_code}")
                print(f"  Response URL: {response.url}")
                
                if response.status_code == 200:
                    try:
                        icgc_data = response.json()
                        all_icgc_results[endpoint["name"]] = icgc_data
                        
                        # Count results
                        if isinstance(icgc_data, dict):
                            result_count = len(icgc_data.get('hits', icgc_data.get('data', [])))
                        else:
                            result_count = len(icgc_data) if isinstance(icgc_data, list) else 1
                        
                        print(f"  Found {result_count} results for {endpoint['name']}")
                        
                    except json.JSONDecodeError as e:
                        print(f"  JSON decode error for {endpoint['name']}: {e}")
                        # Save raw response for debugging
                        debug_file = self.data_dir / f"icgc_debug_{endpoint['name']}.txt"
                        with open(debug_file, 'w') as f:
                            f.write(response.text[:1000])
                else:
                    print(f"  HTTP {response.status_code} for {endpoint['name']}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  Request error for {endpoint['name']}: {e}")
        
        # Save all ICGC results
        if all_icgc_results:
            icgc_file = self.data_dir / "icgc_datasets_v2.json"
            with open(icgc_file, 'w') as f:
                json.dump(all_icgc_results, f, indent=2)
            print(f"ICGC results saved to: {icgc_file}")
        
        return all_icgc_results
    
    def search_additional_geo_datasets(self) -> Dict[str, Any]:
        """Search for additional specific GEO datasets"""
        print("Searching for additional specific GEO datasets...")
        
        # Known relevant GEO series
        specific_series = [
            "GSE149608",  # cfDNA methylation in cancer
            "GSE86831",   # Liquid biopsy methylation
            "GSE120878",  # ctDNA profiling
            "GSE149282"   # cfDNA fragmentomics
        ]
        
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        results = {}
        
        for series_id in specific_series:
            try:
                print(f"  Checking GEO series: {series_id}")
                
                # Get series information
                summary_url = f"{base_url}/esummary.fcgi"
                params = {
                    "db": "gds",
                    "id": series_id,
                    "retmode": "json"
                }
                
                response = self.session.get(summary_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    summary_data = response.json()
                    results[series_id] = summary_data
                    print(f"  Found data for {series_id}")
                else:
                    print(f"  No data found for {series_id}")
                
                time.sleep(0.5)  # Be respectful to NCBI
                
            except Exception as e:
                print(f"  Error accessing {series_id}: {e}")
        
        # Save additional GEO results
        if results:
            geo_additional_file = self.data_dir / "geo_specific_series.json"
            with open(geo_additional_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Additional GEO results saved to: {geo_additional_file}")
        
        return results
    
    def test_api_endpoints(self) -> Dict[str, bool]:
        """Test API endpoint accessibility"""
        print("Testing API endpoint accessibility...")
        
        endpoints = {
            "TCGA": "https://api.gdc.cancer.gov/status",
            "NCBI": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi",
            "ENCODE": "https://www.encodeproject.org/",
            "ICGC": "https://dcc.icgc.org/api/v1/ui"
        }
        
        results = {}
        
        for name, url in endpoints.items():
            try:
                response = self.session.get(url, timeout=10)
                results[name] = response.status_code == 200
                print(f"  {name}: {'✓' if results[name] else '✗'} ({response.status_code})")
            except Exception as e:
                results[name] = False
                print(f"  {name}: ✗ (Error: {e})")
        
        return results
    
    def generate_improved_summary(self) -> None:
        """Generate an improved summary report"""
        print("\nGenerating improved dataset summary report...")
        
        summary = {
            "acquisition_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "v2_improved",
            "databases_attempted": ["TCGA", "GEO", "ENCODE", "ICGC"],
            "files_created": [],
            "api_status": self.test_api_endpoints()
        }
        
        # Check all files created
        data_files = list(self.data_dir.glob("*.json"))
        for file_path in data_files:
            if file_path.name != "acquisition_summary_v2.json":  # Don't include the summary itself
                file_size = file_path.stat().st_size
                summary["files_created"].append({
                    "filename": file_path.name,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024*1024), 2),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", 
                                           time.localtime(file_path.stat().st_mtime))
                })
        
        # Save improved summary
        summary_file = self.data_dir / "acquisition_summary_v2.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Improved summary saved to: {summary_file}")
        print(f"Total files: {len(summary['files_created'])}")

def main():
    """Main function for improved data acquisition"""
    print("Starting Improved Cancer Genomics Data Acquisition...")
    print("=" * 60)
    
    # Initialize improved data acquisition
    acquirer = ImprovedDataAcquisition()
    
    # Test API endpoints first
    api_status = acquirer.test_api_endpoints()
    print()
    
    # Search databases with improved methods
    print("Searching databases...")
    encode_results = acquirer.search_encode_datasets_v2()
    icgc_results = acquirer.search_icgc_datasets_v2()
    additional_geo = acquirer.search_additional_geo_datasets()
    
    # Generate improved summary
    acquirer.generate_improved_summary()
    
    print("\nImproved data acquisition complete!")
    print("Check data/raw/ for all downloaded metadata and debug files.")

if __name__ == "__main__":
    main()
