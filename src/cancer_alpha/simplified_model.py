#!/usr/bin/env python3
"""
Simplified Cancer Genomics Model with Explainability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

class CancerGenomicsAnalyzer:
    """Simplified cancer genomics analyzer with explainability"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic multi-modal cancer genomics data"""
        print("Generating synthetic cancer genomics data...")
        
        np.random.seed(42)
        
        # Define feature names based on our extraction pipeline
        methylation_features = [
            'methyl_global_methylation_mean', 'methyl_global_methylation_std',
            'methyl_cpg_high_methylation_ratio', 'methyl_promoter_methylation',
            'methyl_gene_body_methylation', 'methyl_intergenic_methylation',
            'methyl_methylation_entropy', 'methyl_methylation_variance',
            'methyl_differential_variability', 'methyl_hypermethylation_events'
        ]
        
        fragmentomics_features = [
            'fragment_mean_fragment_length', 'fragment_fragment_length_std',
            'fragment_short_fragment_ratio', 'fragment_long_fragment_ratio',
            'fragment_mononucleosome_peak', 'fragment_end_motif_diversity',
            'fragment_nucleosome_periodicity', 'fragment_fragment_jaggedness',
            'fragment_quality_score', 'fragment_tissue_signature_1',
            'fragment_lung_specific_signal'
        ]
        
        cna_features = [
            'cna_total_alterations', 'cna_amplification_burden',
            'cna_deletion_burden', 'cna_chromosomal_instability_index',
            'cna_focal_alterations', 'cna_broad_alterations',
            'cna_lung_cancer_signature', 'cna_oncogene_amplification',
            'cna_genomic_complexity_score', 'cna_heterogeneity_index'
        ]
        
        interaction_features = [
            'methyl_fragment_interaction', 'fragment_cna_interaction',
            'methyl_cna_interaction'
        ]
        
        self.feature_names = (methylation_features + fragmentomics_features + 
                             cna_features + interaction_features)
        
        n_features = len(self.feature_names)
        n_cancer = n_samples // 2
        n_control = n_samples - n_cancer
        
        # Generate cancer samples with characteristic patterns
        cancer_data = np.random.normal(0, 1, (n_cancer, n_features))
        
        # Methylation: higher global methylation, more hypermethylation events
        cancer_data[:, 0] += 0.8  # global methylation mean
        cancer_data[:, 9] += 1.2  # hypermethylation events
        
        # Fragmentomics: shorter fragments, more nucleosome disruption
        cancer_data[:, 10] -= 15  # mean fragment length
        cancer_data[:, 12] += 0.3  # short fragment ratio
        cancer_data[:, 16] += 0.5  # nucleosome periodicity disruption
        
        # CNAs: more alterations and instability
        cancer_data[:, 21] += 50   # total alterations
        cancer_data[:, 24] += 0.8  # chromosomal instability
        cancer_data[:, 27] += 20   # lung cancer signature
        cancer_data[:, 29] += 100  # genomic complexity
        
        # Generate control samples (normal patterns)
        control_data = np.random.normal(0, 0.5, (n_control, n_features))
        
        # Combine data
        features = np.vstack([cancer_data, control_data])
        labels = np.array([1] * n_cancer + [0] * n_control)
        
        # Create DataFrame
        df = pd.DataFrame(features, columns=self.feature_names)
        df['label'] = labels
        
        print(f"Generated {n_samples} samples with {n_features} features")
        print(f"Cancer samples: {n_cancer}, Control samples: {n_control}")
        
        return df
    
    def train_models(self, data):
        """Train multiple models for comparison"""
        print("Training cancer genomics models...")
        
        # Prepare data
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        # Train Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(X_train_scaled, y_train)
        self.models['logistic_regression'] = lr_model
        
        # Evaluate models
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {results[name]['accuracy']:.3f}")
            print(f"AUC: {results[name]['auc']:.3f}")
        
        # Save models
        for name, model in self.models.items():
            model_path = self.output_dir / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'results': results
        }
    
    def generate_explainability_analysis(self, X_train, X_test, y_test):
        """Generate SHAP explainability analysis"""
        print("Generating explainability analysis...")
        
        # SHAP analysis for Random Forest
        rf_model = self.models['random_forest']
        
        # Create SHAP explainer
        explainer = shap.Explainer(rf_model, X_train)
        shap_values = explainer(X_test[:100])  # Analyze first 100 test samples
        
        # Feature importance from SHAP
        if hasattr(shap_values, 'values'):
            feature_importance = np.abs(shap_values.values).mean(0)
        else:
            feature_importance = np.abs(shap_values).mean(0)
        
        # Ensure feature_importance is 1D
        if len(feature_importance.shape) > 1:
            feature_importance = feature_importance.flatten()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        importance_path = self.output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to {importance_path}")
        
        # Generate SHAP plots
        plt.figure(figsize=(12, 8))
        
        # Summary plot
        plt.subplot(2, 2, 1)
        shap.summary_plot(shap_values, X_test[:100], feature_names=self.feature_names, 
                         show=False, max_display=10)
        plt.title("SHAP Summary Plot")
        
        # Feature importance plot
        plt.subplot(2, 2, 2)
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 15 Feature Importance')
        plt.gca().invert_yaxis()
        
        # Modality contribution
        plt.subplot(2, 2, 3)
        modality_importance = self._calculate_modality_importance(importance_df)
        plt.pie(modality_importance.values(), labels=modality_importance.keys(), autopct='%1.1f%%')
        plt.title('Contribution by Modality')
        
        # Confusion matrix
        plt.subplot(2, 2, 4)
        y_pred = rf_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plot_path = self.output_dir / "explainability_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Explainability plots saved to {plot_path}")
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'modality_importance': modality_importance
        }
    
    def _calculate_modality_importance(self, importance_df):
        """Calculate importance by modality"""
        modality_scores = {
            'Methylation': 0,
            'Fragmentomics': 0,
            'CNA': 0,
            'Interactions': 0
        }
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if feature.startswith('methyl_'):
                modality_scores['Methylation'] += importance
            elif feature.startswith('fragment_'):
                modality_scores['Fragmentomics'] += importance
            elif feature.startswith('cna_'):
                modality_scores['CNA'] += importance
            else:
                modality_scores['Interactions'] += importance
        
        return modality_scores
    
    def generate_biological_insights(self, feature_importance, modality_importance):
        """Generate biological insights from model analysis"""
        print("Generating biological insights...")
        
        insights = {
            'top_biological_features': [],
            'modality_ranking': sorted(modality_importance.items(), key=lambda x: x[1], reverse=True),
            'cancer_signatures': [],
            'clinical_implications': []
        }
        
        # Analyze top features for biological significance
        top_features = feature_importance.head(10)
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            if 'methylation' in feature and importance > 0.1:
                insights['cancer_signatures'].append({
                    'type': 'Epigenetic',
                    'feature': feature,
                    'biological_meaning': 'Altered DNA methylation patterns in cancer cells',
                    'clinical_relevance': 'Potential biomarker for early detection'
                })
            
            elif 'fragment' in feature and importance > 0.1:
                insights['cancer_signatures'].append({
                    'type': 'Fragmentomics',
                    'feature': feature,
                    'biological_meaning': 'Disrupted nucleosome positioning in tumor cells',
                    'clinical_relevance': 'Non-invasive tumor monitoring via cfDNA analysis'
                })
            
            elif 'cna' in feature and importance > 0.1:
                insights['cancer_signatures'].append({
                    'type': 'Genomic Instability',
                    'feature': feature,
                    'biological_meaning': 'Chromosomal alterations characteristic of cancer',
                    'clinical_relevance': 'Tumor burden and progression monitoring'
                })
        
        # Clinical implications
        if modality_importance['Methylation'] > 0.3:
            insights['clinical_implications'].append(
                "High methylation signature importance suggests epigenetic biomarkers for NSCLC screening"
            )
        
        if modality_importance['Fragmentomics'] > 0.25:
            insights['clinical_implications'].append(
                "Fragmentomics patterns indicate feasibility of non-invasive liquid biopsy"
            )
        
        if modality_importance['CNA'] > 0.3:
            insights['clinical_implications'].append(
                "CNA burden correlates with tumor progression and treatment response"
            )
        
        # Save insights
        insights_path = self.output_dir / "biological_insights.txt"
        with open(insights_path, 'w') as f:
            f.write("BIOLOGICAL INSIGHTS FROM CANCER GENOMICS MODEL\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("MODALITY IMPORTANCE RANKING:\n")
            for modality, score in insights['modality_ranking']:
                f.write(f"- {modality}: {score:.3f}\n")
            
            f.write("\nCANCER SIGNATURES IDENTIFIED:\n")
            for sig in insights['cancer_signatures']:
                f.write(f"\n{sig['type']}:\n")
                f.write(f"  Feature: {sig['feature']}\n")
                f.write(f"  Biology: {sig['biological_meaning']}\n")
                f.write(f"  Clinical: {sig['clinical_relevance']}\n")
            
            f.write("\nCLINICAL IMPLICATIONS:\n")
            for impl in insights['clinical_implications']:
                f.write(f"- {impl}\n")
        
        print(f"Biological insights saved to {insights_path}")
        return insights

def main():
    """Main analysis pipeline"""
    print("Cancer Genomics Explainable AI Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CancerGenomicsAnalyzer()
    
    # Generate synthetic data
    data = analyzer.generate_synthetic_data(n_samples=1000)
    
    # Train models
    training_results = analyzer.train_models(data)
    
    # Generate explainability analysis
    explainability_results = analyzer.generate_explainability_analysis(
        training_results['X_train'],
        training_results['X_test'],
        training_results['y_test']
    )
    
    # Generate biological insights
    biological_insights = analyzer.generate_biological_insights(
        explainability_results['feature_importance'],
        explainability_results['modality_importance']
    )
    
    print("\nAnalysis completed successfully!")
    print("Results saved in 'results' directory.")
    
    return analyzer, training_results, explainability_results, biological_insights

if __name__ == "__main__":
    analyzer, training_results, explainability_results, biological_insights = main()
