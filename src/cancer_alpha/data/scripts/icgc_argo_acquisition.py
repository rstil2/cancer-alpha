#!/usr/bin/env python3
"""
ICGC ARGO Data Acquisition Script
================================

This script accesses the ICGC ARGO platform to acquire lung cancer genomic data
as the fourth data source for our multi-modal cancer detection analysis.

ICGC ARGO replaces the legacy ICGC Data Portal and provides access to:
- Pan-cancer genomic data
- Somatic mutations
- Copy number alterations
- Structural variations
- Clinical annotations

Author: Cancer Genomics Research Team
Date: July 14, 2025
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time
import os
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ICGCArgoDataAcquisition:
    """
    ICGC ARGO data acquisition class for lung cancer genomic data
    """
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize ICGC ARGO data acquisition
        
        Args:
            output_dir: Directory to save acquired data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ICGC ARGO API endpoints
        self.base_url = "https://api.platform.icgc-argo.org"
        self.graphql_url = f"{self.base_url}/graphql"
        
        # Alternative endpoints to try
        self.alternative_endpoints = [
            "https://api.platform.icgc-argo.org/graphql",
            "https://platform-api.icgc-argo.org/graphql",
            "https://song-api.icgc-argo.org",
            "https://clinical-api.icgc-argo.org"
        ]
        
        # Data collection results
        self.acquired_data = {}
        self.acquisition_log = []
        
        # Lung cancer related terms
        self.lung_cancer_terms = [
            "lung", "LUAD", "LUSC", "NSCLC", "pulmonary", 
            "bronchial", "alveolar", "adenocarcinoma", "squamous"
        ]
        
        logger.info(f"ICGC ARGO Data Acquisition initialized. Output directory: {self.output_dir}")
    
    def test_api_connectivity(self) -> Dict:
        """Test connectivity to ICGC ARGO API endpoints"""
        logger.info("Testing ICGC ARGO API connectivity...")
        
        connectivity_results = {}
        
        for endpoint in self.alternative_endpoints:
            try:
                response = requests.get(endpoint, timeout=10)
                connectivity_results[endpoint] = {
                    "status_code": response.status_code,
                    "accessible": response.status_code < 500,
                    "response_size": len(response.content),
                    "content_type": response.headers.get('content-type', 'unknown')
                }
                logger.info(f"Endpoint {endpoint}: Status {response.status_code}")
                
            except requests.exceptions.RequestException as e:
                connectivity_results[endpoint] = {
                    "status_code": None,
                    "accessible": False,
                    "error": str(e)
                }
                logger.warning(f"Endpoint {endpoint} failed: {e}")
        
        # Save connectivity results
        self.save_json(connectivity_results, "icgc_argo_connectivity_test.json")
        return connectivity_results
    
    def explore_icgc_argo_schema(self) -> Dict:
        """Explore ICGC ARGO GraphQL schema"""
        logger.info("Exploring ICGC ARGO GraphQL schema...")
        
        # GraphQL introspection query
        introspection_query = """
        query IntrospectionQuery {
            __schema {
                queryType { name }
                mutationType { name }
                subscriptionType { name }
                types {
                    ...FullType
                }
            }
        }
        
        fragment FullType on __Type {
            kind
            name
            description
            fields(includeDeprecated: true) {
                name
                description
                type {
                    ...TypeRef
                }
            }
        }
        
        fragment TypeRef on __Type {
            kind
            name
            ofType {
                kind
                name
                ofType {
                    kind
                    name
                }
            }
        }
        """
        
        schema_info = {}
        
        for endpoint in [self.graphql_url] + self.alternative_endpoints:
            try:
                response = requests.post(
                    endpoint,
                    json={"query": introspection_query},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    schema_data = response.json()
                    schema_info[endpoint] = {
                        "status": "success",
                        "data": schema_data,
                        "available_types": [t["name"] for t in schema_data.get("data", {}).get("__schema", {}).get("types", []) if t.get("name")]
                    }
                    logger.info(f"Schema exploration successful for {endpoint}")
                    break
                else:
                    schema_info[endpoint] = {
                        "status": "failed",
                        "status_code": response.status_code,
                        "response": response.text[:500]
                    }
                    
            except Exception as e:
                schema_info[endpoint] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.warning(f"Schema exploration failed for {endpoint}: {e}")
        
        self.save_json(schema_info, "icgc_argo_schema_exploration.json")
        return schema_info
    
    def query_lung_cancer_programs(self) -> Dict:
        """Query for lung cancer programs/studies in ICGC ARGO"""
        logger.info("Querying lung cancer programs in ICGC ARGO...")
        
        # GraphQL query for programs
        programs_query = """
        query {
            programs {
                shortName
                name
                description
                cancerTypes
                countries
                dataCenter
                membershipType
                commitment
                website
                institutions
                cancerTypes
                primarySites
                sampleTypes
                experimentalStrategy
                dataCategories
                genomicFeatures
                description
            }
        }
        """
        
        programs_data = {}
        
        for endpoint in [self.graphql_url] + self.alternative_endpoints:
            try:
                response = requests.post(
                    endpoint,
                    json={"query": programs_query},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    programs_data[endpoint] = data
                    
                    # Filter for lung cancer programs
                    if "data" in data and "programs" in data["data"]:
                        lung_programs = []
                        for program in data["data"]["programs"]:
                            program_text = json.dumps(program).lower()
                            if any(term in program_text for term in self.lung_cancer_terms):
                                lung_programs.append(program)
                        
                        programs_data[endpoint]["lung_cancer_programs"] = lung_programs
                        logger.info(f"Found {len(lung_programs)} lung cancer programs")
                    
                    if lung_programs:
                        break
                        
            except Exception as e:
                programs_data[endpoint] = {"error": str(e)}
                logger.warning(f"Programs query failed for {endpoint}: {e}")
        
        self.save_json(programs_data, "icgc_argo_lung_cancer_programs.json")
        return programs_data
    
    def download_lung_cancer_data(self) -> None:
        """Query molecular data for lung cancer"""
        logger.info("Querying molecular data...")
        
        # GraphQL query for molecular data
        molecular_query = """
        query {
            repository {
                files {
                    objectId
                    name
                    size
                    dataType
                    fileFormat
                    experimentalStrategy
                    donors {
                        donorId
                        primarySite
                        cancerType
                        specimens {
                            specimenId
                            specimenType
                            samples {
                                sampleId
                                sampleType
                                matchedNormalSampleId
                            }
                        }
                    }
                }
            }
        }
        """
        
        molecular_data = {}

        def download_data(file_info, category):
            object_id = file_info.get('objectId')
            file_name = f"{category}_{file_info.get('name')}"
            download_url = f"{self.base_url}/download/{object_id}"

            try:
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    file_path = self.output_dir / file_name
                    with open(file_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    logging.info(f"Downloaded: {file_name}")
                else:
                    logging.error(f"Failed to download {file_name}: {response.status_code}")
            except Exception as e:
                logging.error(f"Error downloading {file_name}: {e}")
        
        for endpoint in [self.graphql_url] + self.alternative_endpoints:
            try:
                response = requests.post(
                    endpoint,
                    json={"query": molecular_query},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    molecular_data[endpoint] = data
                    
                    # Filter for lung cancer files
                    if "data" in data and "repository" in data["data"]:
                        lung_files = []
                        for file_info in data["data"]["repository"].get("files", []):
                            file_text = json.dumps(file_info).lower()
                            if any(term in file_text for term in self.lung_cancer_terms):
                                lung_files.append(file_info)
                        
                        molecular_data[endpoint]["lung_cancer_files"] = lung_files
                        for file in lung_files:
                            if file['dataType'] in ['mutation', 'expression', 'copy number', 'cnv']:
                                download_data(file, file['dataType'])
                        logger.info(f"Found {len(lung_files)} lung cancer molecular files")
                    
                    if lung_files:
                        break
                        
            except Exception as e:
                molecular_data[endpoint] = {"error": str(e)}
                logger.warning(f"Molecular data query failed for {endpoint}: {e}")
        
        self.save_json(molecular_data, "icgc_argo_molecular_data.json")
        return molecular_data
    
    def try_alternative_apis(self) -> Dict:
        """Try alternative approaches to access ICGC ARGO data"""
        logger.info("Trying alternative API approaches...")
        
        alternative_results = {}
        
        # Try REST API endpoints
        rest_endpoints = [
            f"{self.base_url}/programs",
            f"{self.base_url}/donors",
            f"{self.base_url}/files",
            f"{self.base_url}/analysis",
            "https://song-api.icgc-argo.org/studies",
            "https://clinical-api.icgc-argo.org/clinical"
        ]
        
        for endpoint in rest_endpoints:
            try:
                response = requests.get(endpoint, timeout=15)
                alternative_results[endpoint] = {
                    "status_code": response.status_code,
                    "content_type": response.headers.get('content-type', 'unknown'),
                    "response_size": len(response.content),
                    "success": response.status_code == 200
                }
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        alternative_results[endpoint]["data"] = data
                        logger.info(f"Successfully accessed {endpoint}")
                    except:
                        alternative_results[endpoint]["data"] = response.text[:1000]
                        
            except Exception as e:
                alternative_results[endpoint] = {"error": str(e)}
        
        self.save_json(alternative_results, "icgc_argo_alternative_apis.json")
        return alternative_results
    
    def create_synthetic_icgc_data(self) -> pd.DataFrame:
        """Create synthetic ICGC-like data for integration testing"""
        logger.info("Creating synthetic ICGC-like data for integration...")
        
        # Generate synthetic data that mimics ICGC structure
        np.random.seed(42)
        n_samples = 10
        
        # ICGC-like features
        icgc_features = {
            'donor_id': [f'ICGC_DONOR_{i:03d}' for i in range(n_samples)],
            'project_code': ['LUAD-US'] * 5 + ['LUSC-US'] * 5,
            'primary_site': ['lung'] * n_samples,
            'cancer_type': ['adenocarcinoma'] * 5 + ['squamous_cell_carcinoma'] * 5,
            
            # Mutation features
            'total_mutations': np.random.poisson(50, n_samples),
            'missense_mutations': np.random.poisson(30, n_samples),
            'nonsense_mutations': np.random.poisson(5, n_samples),
            'silent_mutations': np.random.poisson(15, n_samples),
            'indel_mutations': np.random.poisson(8, n_samples),
            
            # Copy number features
            'cn_amplifications': np.random.poisson(12, n_samples),
            'cn_deletions': np.random.poisson(15, n_samples),
            'cn_neutral_regions': np.random.poisson(80, n_samples),
            
            # Structural variation features
            'sv_translocations': np.random.poisson(3, n_samples),
            'sv_inversions': np.random.poisson(2, n_samples),
            'sv_insertions': np.random.poisson(4, n_samples),
            'sv_deletions': np.random.poisson(6, n_samples),
            
            # Pathway features
            'tp53_pathway_mutations': np.random.binomial(1, 0.7, n_samples),
            'kras_pathway_mutations': np.random.binomial(1, 0.4, n_samples),
            'pi3k_pathway_mutations': np.random.binomial(1, 0.3, n_samples),
            'rb_pathway_mutations': np.random.binomial(1, 0.5, n_samples),
            
            # Clinical features
            'age_at_diagnosis': np.random.normal(65, 10, n_samples).astype(int),
            'gender': np.random.choice(['male', 'female'], n_samples),
            'smoking_status': np.random.choice(['current', 'former', 'never'], n_samples),
            'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_samples),
            
            # Sample classification
            'sample_type': ['cancer'] * n_samples,
            'label': [1] * n_samples  # All cancer samples
        }
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(icgc_features)
        
        # Add some derived features
        synthetic_df['mutation_burden'] = (
            synthetic_df['total_mutations'] / 
            synthetic_df['total_mutations'].mean()
        )
        
        synthetic_df['cn_instability'] = (
            synthetic_df['cn_amplifications'] + 
            synthetic_df['cn_deletions']
        )
        
        synthetic_df['sv_burden'] = (
            synthetic_df['sv_translocations'] + 
            synthetic_df['sv_inversions'] + 
            synthetic_df['sv_insertions'] + 
            synthetic_df['sv_deletions']
        )
        
        # Save synthetic data
        output_path = self.output_dir / "icgc_argo_synthetic_data.csv"
        synthetic_df.to_csv(output_path, index=False)
        
        logger.info(f"Created synthetic ICGC data with {len(synthetic_df)} samples and {len(synthetic_df.columns)} features")
        logger.info(f"Saved to: {output_path}")
        
        return synthetic_df
    
    def save_json(self, data: Dict, filename: str):
        """Save data as JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {filename} to {output_path}")
    
    def generate_acquisition_report(self):
        """Generate comprehensive acquisition report"""
        logger.info("Generating ICGC ARGO acquisition report...")
        
        report = {
            "acquisition_timestamp": datetime.now().isoformat(),
            "target_database": "ICGC ARGO",
            "purpose": "Fourth data source for multi-modal cancer detection",
            "focus": "Lung cancer genomic data",
            "status": "exploration_phase",
            "findings": {
                "api_status": "under_investigation",
                "data_availability": "to_be_determined",
                "integration_approach": "synthetic_data_created"
            },
            "next_steps": [
                "Complete API endpoint exploration",
                "Identify accessible lung cancer datasets",
                "Develop feature extraction pipeline",
                "Integrate with existing 3-source model",
                "Evaluate performance improvement"
            ],
            "files_generated": [
                "icgc_argo_connectivity_test.json",
                "icgc_argo_schema_exploration.json",
                "icgc_argo_lung_cancer_programs.json",
                "icgc_argo_molecular_data.json",
                "icgc_argo_alternative_apis.json",
                "icgc_argo_synthetic_data.csv"
            ]
        }
        
        self.save_json(report, "icgc_argo_acquisition_report.json")
        
        # Create markdown report
        markdown_report = f"""
# ICGC ARGO Data Acquisition Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Database**: ICGC ARGO (International Cancer Genome Consortium)
**Purpose**: Fourth data source integration for multi-modal cancer detection

## Objective
Integrate ICGC ARGO as the fourth data source to complement existing TCGA, GEO, and ENCODE data sources.

## Current Status
- **API Exploration**: {'✅ Completed' if len(self.acquisition_log) > 0 else '🔄 In Progress'}
- **Data Acquisition**: 🔄 In Progress
- **Synthetic Data**: ✅ Created for integration testing

## Key Findings
- ICGC ARGO replaces the legacy ICGC Data Portal
- New GraphQL-based API architecture
- Focus on lung cancer genomic data (LUAD, LUSC)

## Generated Files
{chr(10).join(f"- {filename}" for filename in report['files_generated'])}

## Next Steps
{chr(10).join(f"1. {step}" for step in report['next_steps'])}

## Integration Plan
1. **Feature Extraction**: Develop ICGC-specific features
2. **Data Harmonization**: Align with existing 3-source format
3. **Model Enhancement**: Extend to 4-source multi-modal model
4. **Performance Evaluation**: Compare 3-source vs 4-source accuracy

---
*Generated by ICGC ARGO Data Acquisition Pipeline*
        """
        
        with open(self.output_dir / "ICGC_ARGO_REPORT.md", 'w') as f:
            f.write(markdown_report)
        
        logger.info("Acquisition report generated successfully")

def main():
    """Main execution function"""
    print("=" * 60)
    print("ICGC ARGO DATA ACQUISITION - FOURTH SOURCE INTEGRATION")
    print("=" * 60)
    
    # Initialize acquisition
    icgc_argo = ICGCArgoDataAcquisition()
    
    # Step 1: Test API connectivity
    print("\n1. Testing API connectivity...")
    connectivity = icgc_argo.test_api_connectivity()
    
    # Step 2: Explore schema
    print("\n2. Exploring GraphQL schema...")
    schema = icgc_argo.explore_icgc_argo_schema()
    
    # Step 3: Query lung cancer programs
    print("\n3. Querying lung cancer programs...")
    programs = icgc_argo.query_lung_cancer_programs()
    
    # Step 4: Download lung cancer data
    print("\n4. Downloading lung cancer data...")
    icgc_argo.download_lung_cancer_data()
    
    # Step 5: Try alternative APIs
    print("\n5. Trying alternative API approaches...")
    alternatives = icgc_argo.try_alternative_apis()
    
    # Step 6: Create synthetic data for testing
    print("\n6. Creating synthetic ICGC data...")
    synthetic_data = icgc_argo.create_synthetic_icgc_data()
    
    # Step 7: Generate report
    print("\n7. Generating acquisition report...")
    icgc_argo.generate_acquisition_report()
    
    print("\n" + "=" * 60)
    print("ICGC ARGO ACQUISITION COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {icgc_argo.output_dir}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print("\nNext step: Integrate with existing 3-source model")

if __name__ == "__main__":
    main()
