#!/usr/bin/env python3
"""
COMPREHENSIVE 50K DATASET EXPLORATORY DATA ANALYSIS
==================================================
Performs thorough exploratory analysis on the 50,000 sample TCGA dataset
- Statistical summaries and distributions
- Cancer type analysis and patterns
- Multi-omics coverage patterns
- Sample quality distributions
- Correlation analysis
- Advanced visualizations

Generates insights for machine learning strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class Comprehensive50kEDA:
    def __init__(self, dataset_path):
        self.logger = self.setup_logging()
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path("data/50k_eda_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load the dataset
        self.df = None
        self.insights = {
            'summary_stats': {},
            'patterns': [],
            'recommendations': [],
            'visualizations': []
        }
        
        self.load_dataset()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_dataset(self):
        """Load the 50k dataset"""
        self.logger.info(f"📊 Loading dataset from {self.dataset_path}")
        
        try:
            self.df = pd.read_csv(self.dataset_path)
            self.logger.info(f"✅ Dataset loaded: {self.df.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset: {e}")
            raise

    def analyze_cancer_type_patterns(self):
        """Analyze cancer type distributions and patterns"""
        self.logger.info("🔍 Analyzing cancer type patterns...")
        
        cancer_dist = self.df['cancer_type'].value_counts()
        
        # Basic statistics
        stats = {
            'total_cancer_types': len(cancer_dist),
            'min_samples': cancer_dist.min(),
            'max_samples': cancer_dist.max(),
            'mean_samples': cancer_dist.mean(),
            'median_samples': cancer_dist.median(),
            'std_samples': cancer_dist.std()
        }
        
        self.insights['summary_stats']['cancer_types'] = stats
        
        # Identify patterns
        if stats['max_samples'] / stats['min_samples'] > 20:
            self.insights['patterns'].append("Highly imbalanced cancer type distribution")
        
        # Top and bottom cancer types
        top_5 = cancer_dist.head(5)
        bottom_5 = cancer_dist.tail(5)
        
        self.logger.info(f"📊 Top 5 cancer types:")
        for cancer, count in top_5.items():
            self.logger.info(f"   {cancer}: {count:,} samples")
        
        self.logger.info(f"📊 Bottom 5 cancer types:")
        for cancer, count in bottom_5.items():
            self.logger.info(f"   {cancer}: {count:,} samples")
        
        return cancer_dist

    def analyze_multi_omics_patterns(self):
        """Analyze multi-omics coverage patterns"""
        self.logger.info("🧬 Analyzing multi-omics patterns...")
        
        omics_cols = ['has_expression', 'has_methylation', 'has_copy_number', 
                     'has_mutations', 'has_protein', 'has_clinical']
        
        # Coverage statistics
        coverage_stats = {}
        for col in omics_cols:
            if col in self.df.columns:
                coverage = self.df[col].sum()
                pct = (coverage / len(self.df)) * 100
                coverage_stats[col] = {
                    'count': coverage,
                    'percentage': pct
                }
        
        self.insights['summary_stats']['omics_coverage'] = coverage_stats
        
        # Multi-omics combinations analysis
        if 'num_data_types' in self.df.columns:
            omics_dist = self.df['num_data_types'].value_counts().sort_index()
            
            # Common combinations
            combination_patterns = self.df['data_types'].value_counts().head(10)
            
            self.logger.info("🧬 Top 10 omics combinations:")
            for combo, count in combination_patterns.items():
                pct = (count / len(self.df)) * 100
                self.logger.info(f"   {combo}: {count:,} samples ({pct:.1f}%)")
        
        return coverage_stats

    def analyze_quality_patterns(self):
        """Analyze sample quality patterns"""
        self.logger.info("⭐ Analyzing quality patterns...")
        
        if 'quality_score' in self.df.columns:
            quality_stats = {
                'mean': self.df['quality_score'].mean(),
                'median': self.df['quality_score'].median(),
                'std': self.df['quality_score'].std(),
                'min': self.df['quality_score'].min(),
                'max': self.df['quality_score'].max()
            }
            
            self.insights['summary_stats']['quality_scores'] = quality_stats
            
            # Quality distribution by cancer type
            quality_by_cancer = self.df.groupby('cancer_type')['quality_score'].agg(['mean', 'std', 'count'])
            quality_by_cancer = quality_by_cancer.sort_values('mean', ascending=False)
            
            self.logger.info("⭐ Top 5 cancer types by quality:")
            for cancer in quality_by_cancer.head(5).index:
                stats = quality_by_cancer.loc[cancer]
                self.logger.info(f"   {cancer}: {stats['mean']:.2f} ± {stats['std']:.2f}")
            
            return quality_by_cancer

    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        self.logger.info("📊 Creating comprehensive visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Cancer type distribution (top 20)
        ax1 = plt.subplot(3, 2, 1)
        cancer_dist = self.df['cancer_type'].value_counts().head(20)
        bars = ax1.barh(range(len(cancer_dist)), cancer_dist.values, color='lightcoral')
        ax1.set_yticks(range(len(cancer_dist)))
        ax1.set_yticklabels(cancer_dist.index, fontsize=10)
        ax1.set_xlabel('Number of Samples')
        ax1.set_title('Top 20 Cancer Types Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, cancer_dist.values)):
            ax1.text(value + 50, i, f'{value:,}', va='center', fontsize=9)
        
        # 2. Multi-omics coverage
        ax2 = plt.subplot(3, 2, 2)
        omics_cols = ['has_expression', 'has_methylation', 'has_copy_number', 
                     'has_mutations', 'has_protein', 'has_clinical']
        
        omics_coverage = []
        omics_labels = []
        for col in omics_cols:
            if col in self.df.columns:
                coverage = self.df[col].sum()
                pct = (coverage / len(self.df)) * 100
                omics_coverage.append(pct)
                omics_labels.append(col.replace('has_', '').title())
        
        bars2 = ax2.bar(omics_labels, omics_coverage, color='lightblue')
        ax2.set_ylabel('Coverage Percentage (%)')
        ax2.set_title('Multi-omics Data Coverage', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars2, omics_coverage):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 3. Multi-omics distribution
        ax3 = plt.subplot(3, 2, 3)
        if 'num_data_types' in self.df.columns:
            omics_dist = self.df['num_data_types'].value_counts().sort_index()
            bars3 = ax3.bar(omics_dist.index, omics_dist.values, color='lightgreen')
            ax3.set_xlabel('Number of Data Types')
            ax3.set_ylabel('Number of Samples')
            ax3.set_title('Multi-omics Distribution', fontsize=14, fontweight='bold')
            
            # Add count labels
            for bar, count in zip(bars3, omics_dist.values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 100,
                        f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # 4. Quality score distribution
        ax4 = plt.subplot(3, 2, 4)
        if 'quality_score' in self.df.columns:
            ax4.hist(self.df['quality_score'], bins=30, alpha=0.7, color='gold', edgecolor='black')
            ax4.axvline(self.df['quality_score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {self.df["quality_score"].mean():.2f}')
            ax4.axvline(self.df['quality_score'].median(), color='blue', linestyle='--', 
                       label=f'Median: {self.df["quality_score"].median():.2f}')
            ax4.set_xlabel('Quality Score')
            ax4.set_ylabel('Number of Samples')
            ax4.set_title('Quality Score Distribution', fontsize=14, fontweight='bold')
            ax4.legend()
        
        # 5. Cancer type vs Multi-omics heatmap
        ax5 = plt.subplot(3, 2, 5)
        
        # Create a pivot table for heatmap
        top_cancers = self.df['cancer_type'].value_counts().head(15).index
        heatmap_data = []
        
        for cancer in top_cancers:
            cancer_data = self.df[self.df['cancer_type'] == cancer]
            row = []
            for col in omics_cols:
                if col in self.df.columns:
                    coverage = (cancer_data[col].sum() / len(cancer_data)) * 100
                    row.append(coverage)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=top_cancers, 
                                 columns=[col.replace('has_', '').title() for col in omics_cols if col in self.df.columns])
        
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5)
        ax5.set_title('Omics Coverage by Cancer Type (%)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Omics Data Types')
        ax5.set_ylabel('Cancer Types')
        
        # 6. Sample size distribution across cancer types
        ax6 = plt.subplot(3, 2, 6)
        cancer_sizes = self.df['cancer_type'].value_counts()
        ax6.hist(cancer_sizes.values, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax6.axvline(cancer_sizes.mean(), color='red', linestyle='--', 
                   label=f'Mean: {cancer_sizes.mean():.0f}')
        ax6.axvline(cancer_sizes.median(), color='blue', linestyle='--', 
                   label=f'Median: {cancer_sizes.median():.0f}')
        ax6.set_xlabel('Samples per Cancer Type')
        ax6.set_ylabel('Number of Cancer Types')
        ax6.set_title('Cancer Type Sample Size Distribution', fontsize=14, fontweight='bold')
        ax6.legend()
        
        plt.tight_layout()
        
        # Save the comprehensive visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = self.output_dir / f"comprehensive_50k_eda_{timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Comprehensive EDA visualization saved: {viz_path}")
        self.insights['visualizations'].append(str(viz_path))
        
        return viz_path

    def generate_correlation_analysis(self):
        """Generate correlation analysis for numerical features"""
        self.logger.info("🔗 Performing correlation analysis...")
        
        # Select numerical columns
        numerical_cols = ['num_data_types', 'num_files', 'total_size_mb', 'quality_score']
        available_cols = [col for col in numerical_cols if col in self.df.columns]
        
        if len(available_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = self.df[available_cols].corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, fmt='.3f')
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            
            # Save correlation plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            corr_path = self.output_dir / f"correlation_analysis_{timestamp}.png"
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.insights['visualizations'].append(str(corr_path))
            self.insights['summary_stats']['correlations'] = corr_matrix.to_dict()
            
            # Identify strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_correlations.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if strong_correlations:
                self.insights['patterns'].append(f"Found {len(strong_correlations)} strong correlations")
            
            return corr_matrix

    def generate_advanced_insights(self):
        """Generate advanced insights and patterns"""
        self.logger.info("💡 Generating advanced insights...")
        
        insights = []
        
        # 1. Multi-omics quality relationship
        if 'num_data_types' in self.df.columns and 'quality_score' in self.df.columns:
            multi_omics_quality = self.df.groupby('num_data_types')['quality_score'].mean()
            if multi_omics_quality.corr(pd.Series(range(len(multi_omics_quality)), index=multi_omics_quality.index)) > 0.7:
                insights.append("Strong positive correlation between multi-omics coverage and quality scores")
        
        # 2. Cancer type complexity patterns
        if 'cancer_type' in self.df.columns and 'num_data_types' in self.df.columns:
            cancer_complexity = self.df.groupby('cancer_type')['num_data_types'].mean().sort_values(ascending=False)
            top_complex = cancer_complexity.head(3).index.tolist()
            insights.append(f"Most multi-omics rich cancer types: {', '.join(top_complex)}")
        
        # 3. Sample size vs quality relationship
        cancer_dist = self.df['cancer_type'].value_counts()
        if 'quality_score' in self.df.columns:
            cancer_quality = self.df.groupby('cancer_type')['quality_score'].mean()
            size_quality_corr = cancer_dist.corr(cancer_quality)
            if abs(size_quality_corr) > 0.3:
                direction = "positive" if size_quality_corr > 0 else "negative"
                insights.append(f"Moderate {direction} correlation between cancer type size and quality")
        
        # 4. Data completeness patterns
        total_samples = len(self.df)
        single_omics = (self.df['num_data_types'] == 1).sum()
        multi_omics_ratio = 1 - (single_omics / total_samples)
        
        if multi_omics_ratio < 0.1:
            insights.append("Dataset is predominantly single-omics - consider omics-specific modeling approaches")
        elif multi_omics_ratio > 0.5:
            insights.append("Rich multi-omics dataset - ideal for integrative modeling approaches")
        
        # 5. Statistical power assessment
        cancer_types = self.df['cancer_type'].nunique()
        samples_per_type = total_samples / cancer_types
        
        if samples_per_type > 1000:
            insights.append("Excellent statistical power for pan-cancer analysis and subtype discovery")
        elif samples_per_type > 100:
            insights.append("Adequate statistical power for cancer type classification")
        
        self.insights['patterns'].extend(insights)
        
        return insights

    def generate_ml_recommendations(self):
        """Generate machine learning strategy recommendations"""
        self.logger.info("🤖 Generating ML strategy recommendations...")
        
        recommendations = []
        
        # Based on multi-omics coverage
        single_omics_pct = (self.df['num_data_types'] == 1).sum() / len(self.df) * 100
        
        if single_omics_pct > 80:
            recommendations.extend([
                "Consider omics-specific models for single-omics samples",
                "Implement multi-task learning to leverage shared patterns",
                "Use transfer learning from single-omics to multi-omics models"
            ])
        
        # Based on cancer type distribution
        cancer_dist = self.df['cancer_type'].value_counts()
        imbalance_ratio = cancer_dist.max() / cancer_dist.min()
        
        if imbalance_ratio > 20:
            recommendations.extend([
                "Implement stratified sampling for balanced training",
                "Use class weighting or focal loss for imbalanced learning",
                "Consider hierarchical classification (cancer family → specific type)"
            ])
        
        # Based on dataset size
        if len(self.df) > 30000:
            recommendations.extend([
                "Leverage deep learning architectures for pattern discovery",
                "Implement ensemble methods for robust predictions",
                "Consider pre-training on large dataset for transfer learning"
            ])
        
        # Based on feature diversity
        if self.df.shape[1] > 10:
            recommendations.extend([
                "Apply dimensionality reduction (PCA, t-SNE, UMAP)",
                "Use feature selection for omics-specific modeling",
                "Implement attention mechanisms for feature importance"
            ])
        
        # General recommendations
        recommendations.extend([
            "Implement cross-validation with cancer type stratification",
            "Use explainable AI methods (SHAP, LIME) for clinical interpretability",
            "Develop multi-omics integration strategies (early, intermediate, late fusion)",
            "Create cancer type-specific models for precision medicine"
        ])
        
        self.insights['recommendations'] = recommendations
        
        return recommendations

    def save_eda_results(self):
        """Save comprehensive EDA results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save insights as JSON
        results_path = self.output_dir / f"eda_insights_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj
        
        # Clean insights for JSON
        clean_insights = {}
        for key, value in self.insights.items():
            if key == 'summary_stats':
                clean_insights[key] = {}
                for stat_key, stat_value in value.items():
                    if isinstance(stat_value, dict):
                        clean_insights[key][stat_key] = {k: convert_types(v) for k, v in stat_value.items()}
                    else:
                        clean_insights[key][stat_key] = convert_types(stat_value)
            else:
                clean_insights[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(clean_insights, f, indent=2, default=str)
        
        self.logger.info(f"💾 EDA results saved: {results_path}")
        
        return results_path

    def run_comprehensive_eda(self):
        """Run complete exploratory data analysis"""
        self.logger.info("🚀 Starting comprehensive EDA on 50K dataset...")
        
        try:
            # Run all EDA components
            cancer_patterns = self.analyze_cancer_type_patterns()
            omics_patterns = self.analyze_multi_omics_patterns()
            quality_patterns = self.analyze_quality_patterns()
            
            # Create visualizations
            viz_path = self.create_comprehensive_visualizations()
            
            # Advanced analyses
            correlations = self.generate_correlation_analysis()
            insights = self.generate_advanced_insights()
            recommendations = self.generate_ml_recommendations()
            
            # Save results
            results_path = self.save_eda_results()
            
            # Print summary
            self.print_eda_summary()
            
            return {
                'insights': self.insights,
                'visualizations': self.insights['visualizations'],
                'results_file': results_path,
                'main_visualization': viz_path
            }
            
        except Exception as e:
            self.logger.error(f"❌ EDA failed: {e}")
            raise

    def print_eda_summary(self):
        """Print comprehensive EDA summary"""
        print(f"""
============================================================
📊 50K DATASET EXPLORATORY DATA ANALYSIS SUMMARY
============================================================

📈 DATASET OVERVIEW:
   Total samples: {len(self.df):,}
   Total features: {self.df.shape[1]}
   Cancer types: {self.df['cancer_type'].nunique()}

🧬 MULTI-OMICS INSIGHTS:""")
        
        if 'omics_coverage' in self.insights['summary_stats']:
            for omics, stats in self.insights['summary_stats']['omics_coverage'].items():
                omics_name = omics.replace('has_', '').title()
                print(f"   {omics_name}: {stats['count']:,} samples ({stats['percentage']:.1f}%)")
        
        if 'cancer_types' in self.insights['summary_stats']:
            cancer_stats = self.insights['summary_stats']['cancer_types']
            print(f"""
🏥 CANCER TYPE DISTRIBUTION:
   Types: {cancer_stats['total_cancer_types']}
   Range: {cancer_stats['min_samples']:,} - {cancer_stats['max_samples']:,} samples
   Mean: {cancer_stats['mean_samples']:.1f} samples per type""")
        
        print(f"\n💡 KEY PATTERNS DISCOVERED: {len(self.insights['patterns'])}")
        for i, pattern in enumerate(self.insights['patterns'][:5], 1):
            print(f"   {i}. {pattern}")
        
        print(f"\n🤖 ML STRATEGY RECOMMENDATIONS: {len(self.insights['recommendations'])}")
        for i, rec in enumerate(self.insights['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        
        if len(self.insights['recommendations']) > 5:
            print(f"   ... and {len(self.insights['recommendations']) - 5} more recommendations")
        
        print(f"\n📊 VISUALIZATIONS GENERATED: {len(self.insights['visualizations'])}")
        for viz in self.insights['visualizations']:
            print(f"   📈 {Path(viz).name}")
        
        print(f"""
============================================================
✅ EDA COMPLETE - Ready for Feature Engineering & ML!
============================================================
""")

def main():
    print("=" * 70)
    print("📊 COMPREHENSIVE 50K DATASET EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print("Deep insights and patterns discovery for ML strategy")
    print("=" * 70)
    
    # Use the latest 50k dataset
    dataset_path = "data/ultra_permissive_50k_output/ultra_permissive_50k_plus_50000_20250822_184637.csv"
    
    try:
        eda = Comprehensive50kEDA(dataset_path)
        results = eda.run_comprehensive_eda()
        
        print(f"\n🎉 EDA COMPLETED SUCCESSFULLY!")
        print(f"📁 Results: {results['results_file']}")
        print(f"📊 Main visualization: {results['main_visualization']}")
        print(f"📈 Additional visualizations: {len(results['visualizations'])}")
        
    except Exception as e:
        print(f"\n❌ EDA failed: {e}")
        raise

if __name__ == "__main__":
    main()
