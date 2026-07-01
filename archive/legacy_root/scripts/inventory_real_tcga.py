#!/usr/bin/env python3
"""
REAL TCGA DATA INVENTORY SCRIPT
===============================
Phase 3: Inventory all authentic TCGA data that survived the purge

This script:
- Catalogs all remaining authentic TCGA samples
- Validates TCGA barcode formats
- Groups by cancer project/type
- Assesses data quality and completeness
- Provides foundation for Phase 4 download planning

Author: Cancer Alpha Project
Purpose: Inventory authentic data post-cleanup
Rule: ONLY count verified real TCGA samples
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class RealTCGAInventory:
    def __init__(self, data_directory="/Users/stillwell/projects/cancer-alpha/data"):
        self.data_dir = Path(data_directory)
        self.project_root = Path("/Users/stillwell/projects/cancer-alpha")
        self.tcga_barcode_pattern = re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}')
        self.inventory_results = {}
        self.sample_catalog = []
        
        print("📊 REAL TCGA DATA INVENTORY")
        print("=" * 60)
        print("🎯 Cataloging ONLY authentic TCGA samples")
        print("🚫 Post-synthetic data cleanup verification")
        print("=" * 60)
    
    def is_valid_tcga_barcode(self, sample_id):
        """Validate TCGA barcode format strictly"""
        if not isinstance(sample_id, str):
            return False
        return bool(self.tcga_barcode_pattern.match(sample_id))
    
    def extract_tcga_project(self, barcode):
        """Extract TCGA project code from barcode"""
        if not self.is_valid_tcga_barcode(barcode):
            return None
        # TCGA-XX-YYYY format - XX is the project code
        parts = barcode.split('-')
        if len(parts) >= 2:
            return f"TCGA-{parts[1]}"
        return None
    
    def analyze_csv_file(self, csv_path):
        """Analyze a CSV file for authentic TCGA samples"""
        try:
            print(f"📄 Analyzing: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Look for sample ID columns
            id_columns = [col for col in df.columns if 'sample' in col.lower() or 'id' in col.lower()]
            if not id_columns:
                id_columns = [df.columns[0]]  # First column as fallback
            
            authentic_samples = []
            for id_col in id_columns:
                if id_col in df.columns:
                    for sample_id in df[id_col].dropna():
                        if self.is_valid_tcga_barcode(str(sample_id)):
                            project = self.extract_tcga_project(str(sample_id))
                            if project:
                                authentic_samples.append({
                                    'sample_id': str(sample_id),
                                    'project': project,
                                    'source_file': str(csv_path),
                                    'n_features': len(df.columns),
                                    'file_size_mb': csv_path.stat().st_size / (1024*1024)
                                })
            
            if authentic_samples:
                print(f"   ✅ Found {len(authentic_samples)} authentic TCGA samples")
                return authentic_samples
            else:
                print(f"   ⚠️  No authentic TCGA samples found")
                return []
                
        except Exception as e:
            print(f"   ❌ Error analyzing {csv_path}: {e}")
            return []
    
    def scan_remaining_data(self):
        """Scan all remaining data files for authentic samples"""
        print("\n🔍 SCANNING REMAINING DATA FILES")
        print("-" * 60)
        
        csv_files = list(self.data_dir.glob('**/*.csv'))
        print(f"📁 Found {len(csv_files)} CSV files to analyze")
        
        all_samples = []
        for csv_file in csv_files:
            samples = self.analyze_csv_file(csv_file)
            all_samples.extend(samples)
        
        # Deduplicate samples by sample_id
        unique_samples = {}
        for sample in all_samples:
            sample_id = sample['sample_id']
            if sample_id not in unique_samples:
                unique_samples[sample_id] = sample
            else:
                # Keep the one with more features
                if sample['n_features'] > unique_samples[sample_id]['n_features']:
                    unique_samples[sample_id] = sample
        
        self.sample_catalog = list(unique_samples.values())
        print(f"\n✅ Total unique authentic samples: {len(self.sample_catalog)}")
        return self.sample_catalog
    
    def analyze_by_project(self):
        """Analyze samples by TCGA project"""
        print("\n📊 ANALYSIS BY TCGA PROJECT")
        print("-" * 60)
        
        project_stats = defaultdict(lambda: {
            'count': 0,
            'samples': [],
            'source_files': set(),
            'avg_features': 0,
            'total_size_mb': 0
        })
        
        for sample in self.sample_catalog:
            project = sample['project']
            project_stats[project]['count'] += 1
            project_stats[project]['samples'].append(sample['sample_id'])
            project_stats[project]['source_files'].add(sample['source_file'])
            project_stats[project]['avg_features'] += sample['n_features']
            project_stats[project]['total_size_mb'] += sample['file_size_mb']
        
        # Calculate averages
        for project, stats in project_stats.items():
            if stats['count'] > 0:
                stats['avg_features'] = stats['avg_features'] / stats['count']
                stats['source_files'] = list(stats['source_files'])
        
        # Sort by sample count
        sorted_projects = sorted(project_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print(f"📈 PROJECT BREAKDOWN ({len(sorted_projects)} projects):")
        for project, stats in sorted_projects:
            print(f"   {project:12}: {stats['count']:>5} samples ({stats['avg_features']:>5.0f} avg features)")
        
        return dict(project_stats)
    
    def assess_data_quality(self):
        """Assess overall data quality and completeness"""
        print("\n🔬 DATA QUALITY ASSESSMENT")
        print("-" * 60)
        
        if not self.sample_catalog:
            print("❌ No authentic samples found for quality assessment")
            return {}
        
        # Feature distribution analysis
        feature_counts = [sample['n_features'] for sample in self.sample_catalog]
        quality_assessment = {
            'total_samples': len(self.sample_catalog),
            'unique_projects': len(set(s['project'] for s in self.sample_catalog)),
            'feature_stats': {
                'min_features': min(feature_counts),
                'max_features': max(feature_counts),
                'avg_features': np.mean(feature_counts),
                'median_features': np.median(feature_counts)
            },
            'source_files': len(set(s['source_file'] for s in self.sample_catalog)),
            'total_data_size_mb': sum(s['file_size_mb'] for s in self.sample_catalog)
        }
        
        print(f"📊 QUALITY METRICS:")
        print(f"   • Total samples: {quality_assessment['total_samples']:,}")
        print(f"   • Unique projects: {quality_assessment['unique_projects']}")
        print(f"   • Source files: {quality_assessment['source_files']}")
        print(f"   • Features per sample: {quality_assessment['feature_stats']['min_features']}-{quality_assessment['feature_stats']['max_features']} (avg: {quality_assessment['feature_stats']['avg_features']:.0f})")
        print(f"   • Total data size: {quality_assessment['total_data_size_mb']:.1f} MB")
        
        return quality_assessment
    
    def generate_inventory_report(self, project_stats, quality_assessment):
        """Generate comprehensive inventory report"""
        print("\n📋 GENERATING INVENTORY REPORT")
        print("-" * 60)
        
        # Create comprehensive inventory
        inventory = {
            'inventory_timestamp': pd.Timestamp.now().isoformat(),
            'phase': 'Phase 3 - Real Data Inventory',
            'total_samples': len(self.sample_catalog),
            'projects': {},
            'quality_assessment': quality_assessment,
            'gap_analysis': {
                'target_samples': 50000,
                'current_samples': len(self.sample_catalog),
                'gap': 50000 - len(self.sample_catalog),
                'completion_percentage': (len(self.sample_catalog) / 50000) * 100
            },
            'sample_catalog': self.sample_catalog[:100],  # First 100 samples
            'recommendations': []
        }
        
        # Add project details
        for project, stats in project_stats.items():
            inventory['projects'][project] = {
                'sample_count': stats['count'],
                'avg_features': round(stats['avg_features']),
                'source_files': len(stats['source_files']),
                'data_size_mb': round(stats['total_size_mb'], 2)
            }
        
        # Generate recommendations
        gap = inventory['gap_analysis']['gap']
        if gap > 0:
            inventory['recommendations'].extend([
                f"Need to download {gap:,} additional authentic TCGA samples",
                "Focus on high-yield cancer projects (BRCA, LUAD, COAD, etc.)",
                "Use GDC API with strict MD5 verification",
                "Implement sample validation framework before processing"
            ])
        
        # Save inventory
        inventory_path = self.project_root / "real_tcga_inventory.json"
        with open(inventory_path, 'w') as f:
            json.dump(inventory, f, indent=2)
        
        print(f"📄 Inventory saved: {inventory_path}")
        
        # Save sample catalog CSV
        if self.sample_catalog:
            catalog_df = pd.DataFrame(self.sample_catalog)
            catalog_csv_path = self.project_root / "authentic_tcga_catalog.csv"
            catalog_df.to_csv(catalog_csv_path, index=False)
            print(f"📄 Sample catalog saved: {catalog_csv_path}")
        
        return inventory
    
    def run_comprehensive_inventory(self):
        """Execute complete inventory process"""
        print("\n🎯 COMPREHENSIVE REAL TCGA INVENTORY")
        print("=" * 60)
        
        # Scan for samples
        samples = self.scan_remaining_data()
        
        # Analyze by project
        project_stats = self.analyze_by_project()
        
        # Assess quality
        quality_assessment = self.assess_data_quality()
        
        # Generate report
        inventory = self.generate_inventory_report(project_stats, quality_assessment)
        
        # Final summary
        gap = 50000 - len(self.sample_catalog)
        completion_pct = (len(self.sample_catalog) / 50000) * 100
        
        print(f"\n" + "📊" * 60)
        print("✅ REAL TCGA INVENTORY COMPLETED")
        print("📊" * 60)
        print(f"🎯 CURRENT STATUS:")
        print(f"   • Authentic samples found: {len(self.sample_catalog):,}")
        print(f"   • TCGA projects represented: {len(set(s['project'] for s in self.sample_catalog))}")
        print(f"   • Completion toward 50K goal: {completion_pct:.1f}%")
        print(f"   • Samples still needed: {gap:,}")
        print(f"\n🚀 READY FOR PHASE 4: Clean Download Pipeline")
        
        return inventory

def main():
    """Run the real TCGA data inventory"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║              REAL TCGA DATA INVENTORY                    ║
    ║                                                          ║
    ║  📊 Cataloging authentic samples post-cleanup            ║
    ║  🎯 Foundation for 50K real data achievement             ║
    ║  🚫 ZERO synthetic data - real samples only             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    inventory = RealTCGAInventory()
    results = inventory.run_comprehensive_inventory()
    
    print(f"\n✅ Phase 3 complete. Ready for Phase 4: Download additional real TCGA data")

if __name__ == "__main__":
    main()
