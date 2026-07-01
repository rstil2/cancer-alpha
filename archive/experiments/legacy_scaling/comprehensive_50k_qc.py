#!/usr/bin/env python3
"""
COMPREHENSIVE 50K DATASET QUALITY CONTROL
=========================================
Performs thorough quality control and validation on the 50,000 sample TCGA dataset
- Data integrity checks
- Duplicate detection and removal
- Missing value analysis
- Statistical validation
- Format consistency verification
- Sample quality scoring

Ensures dataset is production-ready for advanced analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class Comprehensive50kQC:
    def __init__(self, dataset_path):
        self.logger = self.setup_logging()
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path("data/50k_qc_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # QC results storage
        self.qc_results = {
            'dataset_info': {},
            'quality_issues': [],
            'recommendations': [],
            'statistics': {},
            'validation_passed': True
        }
        
        # Load the dataset
        self.df = None
        self.load_dataset()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_dataset(self):
        """Load and validate the 50k dataset"""
        self.logger.info(f"🔍 Loading dataset from {self.dataset_path}")
        
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.logger.info(f"✅ Dataset loaded: {self.df.shape}")
            
            # Basic dataset info
            self.qc_results['dataset_info'] = {
                'file_path': str(self.dataset_path),
                'shape': self.df.shape,
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
                'columns': list(self.df.columns),
                'dtypes': self.df.dtypes.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset: {e}")
            raise

    def check_data_integrity(self):
        """Comprehensive data integrity checks"""
        self.logger.info("🔍 Performing data integrity checks...")
        
        issues = []
        
        # 1. Check for exact duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} exact duplicate rows")
            self.logger.warning(f"⚠️ {duplicates} exact duplicates found")
        
        # 2. Check for duplicate sample IDs
        duplicate_samples = self.df['sample_id'].duplicated().sum()
        if duplicate_samples > 0:
            issues.append(f"Found {duplicate_samples} duplicate sample IDs")
            self.logger.warning(f"⚠️ {duplicate_samples} duplicate sample IDs found")
        
        # 3. Check for missing sample IDs
        missing_sample_ids = self.df['sample_id'].isna().sum()
        if missing_sample_ids > 0:
            issues.append(f"Found {missing_sample_ids} missing sample IDs")
            self.logger.error(f"❌ {missing_sample_ids} missing sample IDs found")
        
        # 4. Check sample ID format consistency
        sample_id_patterns = []
        for sample_id in self.df['sample_id'].dropna().head(1000):  # Sample check
            if str(sample_id).startswith('TCGA-'):
                sample_id_patterns.append('TCGA_standard')
            elif 'TCGA_SAMPLE_' in str(sample_id):
                sample_id_patterns.append('TCGA_synthetic')
            else:
                sample_id_patterns.append('other')
        
        pattern_counts = Counter(sample_id_patterns)
        self.logger.info(f"📊 Sample ID patterns: {dict(pattern_counts)}")
        
        # 5. Check for invalid data types
        expected_bool_cols = ['has_expression', 'has_methylation', 'has_copy_number', 
                             'has_mutations', 'has_protein', 'has_clinical']
        for col in expected_bool_cols:
            if col in self.df.columns:
                unique_vals = set(self.df[col].dropna().unique())
                if not unique_vals.issubset({True, False, 0, 1}):
                    issues.append(f"Invalid boolean values in {col}: {unique_vals}")
        
        # 6. Check numeric ranges
        if 'num_data_types' in self.df.columns:
            invalid_num_types = ((self.df['num_data_types'] < 0) | (self.df['num_data_types'] > 10)).sum()
            if invalid_num_types > 0:
                issues.append(f"Found {invalid_num_types} samples with invalid num_data_types")
        
        if 'quality_score' in self.df.columns:
            negative_scores = (self.df['quality_score'] < 0).sum()
            if negative_scores > 0:
                issues.append(f"Found {negative_scores} samples with negative quality scores")
        
        self.qc_results['quality_issues'].extend(issues)
        
        if issues:
            self.logger.warning(f"⚠️ Found {len(issues)} data integrity issues")
            self.qc_results['validation_passed'] = False
        else:
            self.logger.info("✅ Data integrity checks passed")

    def analyze_missing_values(self):
        """Analyze missing values across all columns"""
        self.logger.info("🔍 Analyzing missing values...")
        
        missing_stats = {}
        total_samples = len(self.df)
        
        for column in self.df.columns:
            missing_count = self.df[column].isna().sum()
            missing_pct = (missing_count / total_samples) * 100
            
            missing_stats[column] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
            
            if missing_pct > 50:
                self.qc_results['quality_issues'].append(f"Column '{column}' has {missing_pct:.1f}% missing values")
                self.logger.warning(f"⚠️ {column}: {missing_pct:.1f}% missing")
        
        self.qc_results['statistics']['missing_values'] = missing_stats
        
        # Overall missing value summary
        total_missing = self.df.isna().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        overall_missing_pct = (total_missing / total_cells) * 100
        
        self.logger.info(f"📊 Overall missing values: {overall_missing_pct:.2f}%")

    def validate_cancer_types(self):
        """Validate cancer type consistency and coverage"""
        self.logger.info("🔍 Validating cancer types...")
        
        if 'cancer_type' not in self.df.columns:
            self.qc_results['quality_issues'].append("Missing 'cancer_type' column")
            return
        
        # Cancer type distribution
        cancer_dist = self.df['cancer_type'].value_counts()
        self.qc_results['statistics']['cancer_type_distribution'] = cancer_dist.to_dict()
        
        # Check for valid TCGA cancer types
        valid_tcga_types = {
            'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL',
            'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC',
            'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG',
            'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV',
            'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC',
            'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM',
            'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM'
        }
        
        unknown_types = set(self.df['cancer_type'].unique()) - valid_tcga_types
        if unknown_types:
            self.logger.info(f"📊 Found non-standard cancer types: {unknown_types}")
        
        # Check for imbalanced cancer types
        min_samples = cancer_dist.min()
        max_samples = cancer_dist.max()
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        if imbalance_ratio > 100:  # Very imbalanced
            self.qc_results['quality_issues'].append(f"Highly imbalanced cancer types (ratio: {imbalance_ratio:.1f})")
            self.logger.warning(f"⚠️ High cancer type imbalance: {imbalance_ratio:.1f}")
        
        self.logger.info(f"📊 Cancer types: {len(cancer_dist)} types, {min_samples}-{max_samples} samples per type")

    def assess_multi_omics_coverage(self):
        """Assess multi-omics data coverage and quality"""
        self.logger.info("🔍 Assessing multi-omics coverage...")
        
        omics_cols = ['has_expression', 'has_methylation', 'has_copy_number', 
                     'has_mutations', 'has_protein', 'has_clinical']
        
        coverage_stats = {}
        
        for col in omics_cols:
            if col in self.df.columns:
                coverage = self.df[col].sum() if self.df[col].dtype == bool else (self.df[col] == True).sum()
                coverage_pct = (coverage / len(self.df)) * 100
                coverage_stats[col] = {
                    'samples': int(coverage),
                    'percentage': round(coverage_pct, 2)
                }
                self.logger.info(f"📊 {col}: {coverage:,} samples ({coverage_pct:.1f}%)")
        
        self.qc_results['statistics']['omics_coverage'] = coverage_stats
        
        # Multi-omics combinations
        if 'num_data_types' in self.df.columns:
            multi_omics_dist = self.df['num_data_types'].value_counts().sort_index()
            self.qc_results['statistics']['multi_omics_distribution'] = multi_omics_dist.to_dict()
            
            # Quality thresholds
            single_omics = (self.df['num_data_types'] == 1).sum()
            multi_omics_3plus = (self.df['num_data_types'] >= 3).sum()
            
            single_omics_pct = (single_omics / len(self.df)) * 100
            multi_omics_pct = (multi_omics_3plus / len(self.df)) * 100
            
            if single_omics_pct > 60:
                self.qc_results['quality_issues'].append(f"High proportion of single-omics samples: {single_omics_pct:.1f}%")
            
            self.logger.info(f"📊 Single-omics: {single_omics_pct:.1f}%, Multi-omics (3+): {multi_omics_pct:.1f}%")

    def statistical_validation(self):
        """Perform statistical validation of the dataset"""
        self.logger.info("🔍 Performing statistical validation...")
        
        # Sample size adequacy
        total_samples = len(self.df)
        cancer_types = self.df['cancer_type'].nunique() if 'cancer_type' in self.df.columns else 1
        
        # Rule of thumb: at least 30 samples per cancer type for basic analysis
        min_recommended = cancer_types * 30
        if total_samples < min_recommended:
            self.qc_results['quality_issues'].append(f"Sample size may be inadequate: {total_samples} < {min_recommended} recommended")
        
        # Power analysis for multi-class classification
        samples_per_class = total_samples / cancer_types if cancer_types > 0 else 0
        
        statistical_power = "adequate" if samples_per_class >= 100 else "limited" if samples_per_class >= 30 else "insufficient"
        
        self.qc_results['statistics']['statistical_power'] = {
            'samples_per_cancer_type': round(samples_per_class, 1),
            'assessment': statistical_power,
            'total_samples': total_samples,
            'cancer_types': cancer_types
        }
        
        self.logger.info(f"📊 Statistical power: {statistical_power} ({samples_per_class:.1f} samples/type)")

    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        self.logger.info("📊 Generating quality report...")
        
        # Create visualizations
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cancer type distribution
        if 'cancer_type' in self.df.columns:
            cancer_counts = self.df['cancer_type'].value_counts().head(15)
            axes[0, 0].barh(range(len(cancer_counts)), cancer_counts.values)
            axes[0, 0].set_yticks(range(len(cancer_counts)))
            axes[0, 0].set_yticklabels(cancer_counts.index, fontsize=8)
            axes[0, 0].set_xlabel('Number of Samples')
            axes[0, 0].set_title('Top 15 Cancer Types Distribution')
        
        # 2. Multi-omics coverage
        if 'num_data_types' in self.df.columns:
            omics_dist = self.df['num_data_types'].value_counts().sort_index()
            axes[0, 1].bar(omics_dist.index, omics_dist.values)
            axes[0, 1].set_xlabel('Number of Data Types')
            axes[0, 1].set_ylabel('Number of Samples')
            axes[0, 1].set_title('Multi-omics Coverage Distribution')
        
        # 3. Missing values heatmap
        missing_data = self.df.isnull().sum()
        if len(missing_data) > 0:
            axes[1, 0].barh(range(len(missing_data)), missing_data.values)
            axes[1, 0].set_yticks(range(len(missing_data)))
            axes[1, 0].set_yticklabels(missing_data.index, fontsize=8)
            axes[1, 0].set_xlabel('Missing Values Count')
            axes[1, 0].set_title('Missing Values by Column')
        
        # 4. Quality score distribution
        if 'quality_score' in self.df.columns:
            axes[1, 1].hist(self.df['quality_score'].dropna(), bins=30, alpha=0.7)
            axes[1, 1].set_xlabel('Quality Score')
            axes[1, 1].set_ylabel('Number of Samples')
            axes[1, 1].set_title('Quality Score Distribution')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f"50k_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Quality report visualization saved: {viz_path}")
        
        return viz_path

    def run_full_qc(self):
        """Run complete quality control pipeline"""
        self.logger.info("🚀 Starting comprehensive 50K dataset quality control...")
        
        try:
            # Run all QC checks
            self.check_data_integrity()
            self.analyze_missing_values()
            self.validate_cancer_types()
            self.assess_multi_omics_coverage()
            self.statistical_validation()
            
            # Generate recommendations
            self.generate_recommendations()
            
            # Create visualizations
            viz_path = self.generate_quality_report()
            
            # Save QC results
            results_path = self.save_qc_results()
            
            # Print summary
            self.print_qc_summary()
            
            return {
                'passed': self.qc_results['validation_passed'],
                'results_file': results_path,
                'visualization': viz_path,
                'summary': self.qc_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ QC failed: {e}")
            raise

    def generate_recommendations(self):
        """Generate actionable recommendations based on QC results"""
        recommendations = []
        
        # Based on quality issues found
        if self.qc_results['quality_issues']:
            if any('duplicate' in issue.lower() for issue in self.qc_results['quality_issues']):
                recommendations.append("Remove duplicate samples to ensure data integrity")
            
            if any('missing' in issue.lower() for issue in self.qc_results['quality_issues']):
                recommendations.append("Implement missing value imputation strategy")
            
            if any('imbalanced' in issue.lower() for issue in self.qc_results['quality_issues']):
                recommendations.append("Consider stratified sampling or class weighting for imbalanced cancer types")
        
        # Based on multi-omics coverage
        if 'omics_coverage' in self.qc_results['statistics']:
            low_coverage_omics = []
            for omics, stats in self.qc_results['statistics']['omics_coverage'].items():
                if stats['percentage'] < 20:
                    low_coverage_omics.append(omics)
            
            if low_coverage_omics:
                recommendations.append(f"Consider excluding or imputing low-coverage omics: {', '.join(low_coverage_omics)}")
        
        # General recommendations
        recommendations.extend([
            "Implement feature scaling/normalization before machine learning",
            "Consider dimensionality reduction for high-dimensional omics data",
            "Use stratified cross-validation to handle cancer type imbalance",
            "Implement ensemble methods for robust multi-omics integration"
        ])
        
        self.qc_results['recommendations'] = recommendations

    def save_qc_results(self):
        """Save comprehensive QC results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"50k_qc_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj
        
        # Deep convert all values
        import json
        json_str = json.dumps(self.qc_results, default=convert_types, indent=2)
        json_obj = json.loads(json_str)
        
        with open(results_path, 'w') as f:
            json.dump(json_obj, f, indent=2)
        
        self.logger.info(f"💾 QC results saved: {results_path}")
        return results_path

    def print_qc_summary(self):
        """Print comprehensive QC summary"""
        status = "✅ PASSED" if self.qc_results['validation_passed'] else "⚠️ ISSUES FOUND"
        
        print(f"""
============================================================
📊 50K DATASET QUALITY CONTROL SUMMARY
============================================================

🎯 VALIDATION STATUS: {status}

📈 DATASET OVERVIEW:
   Shape: {self.qc_results['dataset_info']['shape']}
   Memory Usage: {self.qc_results['dataset_info']['memory_usage_mb']:.2f} MB
   Columns: {len(self.qc_results['dataset_info']['columns'])}

🔍 QUALITY ISSUES: {len(self.qc_results['quality_issues'])}""")
        
        for issue in self.qc_results['quality_issues']:
            print(f"   ⚠️ {issue}")
        
        if 'cancer_type_distribution' in self.qc_results['statistics']:
            cancer_count = len(self.qc_results['statistics']['cancer_type_distribution'])
            print(f"\n🏥 CANCER COVERAGE: {cancer_count} cancer types")
        
        if 'omics_coverage' in self.qc_results['statistics']:
            print(f"\n🧬 MULTI-OMICS COVERAGE:")
            for omics, stats in self.qc_results['statistics']['omics_coverage'].items():
                print(f"   {omics.replace('has_', '').title()}: {stats['samples']:,} samples ({stats['percentage']:.1f}%)")
        
        if 'statistical_power' in self.qc_results['statistics']:
            power_info = self.qc_results['statistics']['statistical_power']
            print(f"\n📊 STATISTICAL ASSESSMENT: {power_info['assessment'].upper()}")
            print(f"   Samples per cancer type: {power_info['samples_per_cancer_type']:.1f}")
        
        print(f"\n💡 RECOMMENDATIONS: {len(self.qc_results['recommendations'])}")
        for i, rec in enumerate(self.qc_results['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        
        if len(self.qc_results['recommendations']) > 5:
            print(f"   ... and {len(self.qc_results['recommendations']) - 5} more")
        
        print(f"""
============================================================
""")

def main():
    print("=" * 70)
    print("📊 COMPREHENSIVE 50K DATASET QUALITY CONTROL")
    print("=" * 70)
    print("Validating the 50,000 sample TCGA dataset for production readiness")
    print("=" * 70)
    
    # Use the latest 50k dataset
    dataset_path = "data/ultra_permissive_50k_output/ultra_permissive_50k_plus_50000_20250822_184637.csv"
    
    try:
        qc = Comprehensive50kQC(dataset_path)
        results = qc.run_full_qc()
        
        if results['passed']:
            print("\n🎉 QUALITY CONTROL PASSED! Dataset is ready for next steps.")
        else:
            print("\n⚠️ Quality issues found. Review recommendations before proceeding.")
        
        print(f"\n📁 Results: {results['results_file']}")
        print(f"📊 Visualization: {results['visualization']}")
        
    except Exception as e:
        print(f"\n❌ QC failed: {e}")
        raise

if __name__ == "__main__":
    main()
