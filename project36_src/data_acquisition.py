#!/usr/bin/env python3
"""
Data Acquisition Script for Cancer Genomics Project
Accesses TCGA, GEO, ICGC, and ENCODE databases for ctDNA datasets
"""

import requests
import json
import pandas as pd
import os
from pathlib import Path
import time
from typing import Dict, List, Any

class DataAcquisition:
    """Class to handle data acquisition from multiple genomics databases"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def search_tcga_ctdna(self) -> Dict[str, Any]:
        """Search TCGA for NSCLC ctDNA methylation data"""
        print("Searching TCGA for NSCLC ctDNA data...")
        
        # TCGA GDC API endpoint
        base_url = "https://api.gdc.cancer.gov"
        
        # Search for NSCLC projects
        projects_url = f"{base_url}/projects"
        params = {
            "filters": json.dumps({
                "op": "in",
                "content": {
                    "field": "disease_type",
                    "value": ["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"]
                }
            }),
            "format": "json",
            "size": "50"
        }
        
        try:
            response = requests.get(projects_url, params=params)
            response.raise_for_status()
            projects_data = response.json()
            
            # Save TCGA project data
            tcga_file = self.data_dir / "tcga_nsclc_projects.json"
            with open(tcga_file, 'w') as f:
                json.dump(projects_data, f, indent=2)
            
            print(f"Found {len(projects_data.get('data', []))} TCGA NSCLC projects")
            return projects_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error accessing TCGA API: {e}")
            return {}
    
    def search_tcga_methylation_files(self) -> Dict[str, Any]:
        """Search for methylation files in TCGA"""
        print("Searching TCGA for methylation files...")
        
        base_url = "https://api.gdc.cancer.gov"
        files_url = f"{base_url}/files"
        
        params = {
            "filters": json.dumps({
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "data_type",
                            "value": ["Methylation Beta Value"]
                        }
                    },
                    {
                        "op": "in",
                        "content": {
                            "field": "cases.project.disease_type",
                            "value": ["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"]
                        }
                    }
                ]
            }),
            "format": "json",
            "size": "100"
        }
        
        try:
            response = requests.get(files_url, params=params)
            response.raise_for_status()
            files_data = response.json()
            
            # Save methylation files data
            methyl_file = self.data_dir / "tcga_methylation_files.json"
            with open(methyl_file, 'w') as f:
                json.dump(files_data, f, indent=2)
            
            print(f"Found {len(files_data.get('data', []))} TCGA methylation files")
            return files_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error accessing TCGA methylation files: {e}")
            return {}
    
    def search_geo_datasets(self) -> List[Dict[str, Any]]:
        """Search GEO for cfMeDIP-seq NSCLC datasets"""
        print("Searching GEO for cfMeDIP-seq NSCLC datasets...")
        
        # GEO E-utilities API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Search terms for cfMeDIP-seq and NSCLC
        search_terms = [
            "cfMeDIP-seq NSCLC",
            "circulating tumor DNA methylation lung cancer",
            "liquid biopsy methylation NSCLC",
            "cfDNA methylation non-small cell lung cancer"
        ]
        
        all_results = []
        
        for term in search_terms:
            search_url = f"{base_url}/esearch.fcgi"
            params = {
                "db": "gds",
                "term": term,
                "retmax": "20",
                "retmode": "json"
            }
            
            try:
                response = requests.get(search_url, params=params)
                response.raise_for_status()
                search_data = response.json()
                
                if 'esearchresult' in search_data and 'idlist' in search_data['esearchresult']:
                    ids = search_data['esearchresult']['idlist']
                    
                    if ids:
                        # Get detailed information for each dataset
                        summary_url = f"{base_url}/esummary.fcgi"
                        summary_params = {
                            "db": "gds",
                            "id": ",".join(ids),
                            "retmode": "json"
                        }
                        
                        summary_response = requests.get(summary_url, params=summary_params)
                        summary_response.raise_for_status()
                        summary_data = summary_response.json()
                        
                        all_results.append({
                            "search_term": term,
                            "results": summary_data
                        })
                        
                        print(f"Found {len(ids)} datasets for term: {term}")
                
                time.sleep(0.5)  # Be respectful to NCBI servers
                
            except requests.exceptions.RequestException as e:
                print(f"Error searching GEO for term '{term}': {e}")
        
        # Save GEO results
        geo_file = self.data_dir / "geo_cfmedip_results.json"
        with open(geo_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def search_encode_datasets(self) -> Dict[str, Any]:
        """Search ENCODE for relevant datasets"""
        print("Searching ENCODE for fragmentomics and methylation data...")
        
        base_url = "https://www.encodeproject.org"
        search_url = f"{base_url}/search/"
        
        # Search parameters for ENCODE
        params = {
            "type": "Experiment",
            "assay_title": "DNA methylation profiling by array assay",
            "target.investigated_as": "chromatin structure",
            "format": "json",
            "limit": 50
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            encode_data = response.json()
            
            # Save ENCODE results
            encode_file = self.data_dir / "encode_methylation_experiments.json"
            with open(encode_file, 'w') as f:
                json.dump(encode_data, f, indent=2)
            
            print(f"Found {len(encode_data.get('@graph', []))} ENCODE experiments")
            return encode_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error accessing ENCODE API: {e}")
            return {}
    
    def search_icgc_datasets(self) -> Dict[str, Any]:
        """Search ICGC for lung cancer datasets"""
        print("Searching ICGC for lung cancer datasets...")
        
        # ICGC API endpoint
        base_url = "https://dcc.icgc.org/api/v1"
        
        # Search for lung cancer projects
        projects_url = f"{base_url}/projects"
        params = {
            "filters": json.dumps({
                "project": {
                    "primarySite": {
                        "is": ["Lung"]
                    }
                }
            }),
            "size": 25
        }
        
        try:
            response = requests.get(projects_url, params=params)
            response.raise_for_status()
            icgc_data = response.json()
            
            # Save ICGC results
            icgc_file = self.data_dir / "icgc_lung_projects.json"
            with open(icgc_file, 'w') as f:
                json.dump(icgc_data, f, indent=2)
            
            print(f"Found {len(icgc_data.get('hits', []))} ICGC lung cancer projects")
            return icgc_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error accessing ICGC API: {e}")
            return {}
    
    def generate_summary_report(self) -> None:
        """Generate a summary report of all found datasets"""
        print("\nGenerating dataset summary report...")
        
        summary = {
            "acquisition_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "databases_searched": ["TCGA", "GEO", "ENCODE", "ICGC"],
            "files_created": []
        }
        
        # Check which files were created
        data_files = list(self.data_dir.glob("*.json"))
        for file_path in data_files:
            file_size = file_path.stat().st_size
            summary["files_created"].append({
                "filename": file_path.name,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024*1024), 2)
            })
        
        # Save summary
        summary_file = self.data_dir / "acquisition_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved to: {summary_file}")
        print(f"Total files acquired: {len(summary['files_created'])}")

def main():
    """Main function to run data acquisition"""
    print("Starting Cancer Genomics Data Acquisition...")
    print("=" * 50)
    
    # Initialize data acquisition
    acquirer = DataAcquisition()
    
    # Search all databases
    tcga_projects = acquirer.search_tcga_ctdna()
    tcga_methylation = acquirer.search_tcga_methylation_files()
    geo_results = acquirer.search_geo_datasets()
    encode_results = acquirer.search_encode_datasets()
    icgc_results = acquirer.search_icgc_datasets()
    
    # Generate summary
    acquirer.generate_summary_report()
    
    print("\nData acquisition complete!")
    print("Check the data/raw directory for downloaded metadata.")

if __name__ == "__main__":
    main()
