#!/usr/bin/env python3
"""
Cancer Detection Package - Easy-to-use inference interface
=========================================================

This module provides a simple interface for cancer detection using multi-modal genomic data.
Supports both classical ML models and transformer-based models.

Usage:
    from cancer_detector import CancerDetector
    
    # Initialize detector
    detector = CancerDetector()
    
    # Load sample data or your own
    data = detector.load_sample_data()
    
    # Make predictions
    predictions = detector.predict(data)
    
    # Get detailed analysis
    analysis = detector.analyze_features(data)

Author: Cancer Genomics Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import pickle
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap

# Import our custom models
try:
    from .multimodal_model import CancerGenomicsTransformer, MultiModalDataset
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    warnings.warn("Transformer model not available. Using classical models only.")

try:
    from .comprehensive_cancer_analysis import CancerGenomicsAnalyzer
    FULL_ANALYSIS_AVAILABLE = True
except ImportError:
    FULL_ANALYSIS_AVAILABLE = False
    warnings.warn("Full analysis pipeline not available.")

class CancerDetector:
    """
    Main cancer detection interface
    
    Provides easy-to-use methods for cancer detection from multi-modal genomic data.
    Supports both classical ML and transformer models.
    """
    
    def __init__(self, model_type: str = "random_forest", model_path: Optional[str] = None):
        """
        Initialize the cancer detector
        
        Args:
            model_type: Type of model to use ('random_forest', 'logistic', 'transformer')
            model_path: Path to pre-trained model (optional)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model"""
        if self.model_path and Path(self.model_path).exists():
            self.load_model(self.model_path)
        else:
            # Initialize with default parameters
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == "logistic":
                self.model = LogisticRegression(random_state=42, max_iter=1000)
            elif self.model_type == "transformer":
                if TRANSFORMER_AVAILABLE:
                    self.model = CancerGenomicsTransformer(
                        input_dim=47,  # Default feature count
                        d_model=256,
                        n_heads=8,
                        n_layers=4
                    )
                else:
                    raise ValueError("Transformer model not available. Install PyTorch and dependencies.")
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.scaler = StandardScaler()
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the saved model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if model_path.suffix == '.pkl':
            # Classical ML model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data.get('scaler', StandardScaler())
                self.feature_names = model_data.get('feature_names', None)
        
        elif model_path.suffix == '.ckpt':
            # Transformer model
            if TRANSFORMER_AVAILABLE:
                self.model = CancerGenomicsTransformer.load_from_checkpoint(model_path)
                self.model.eval()
            else:
                raise ValueError("Transformer model not available. Install PyTorch.")
        
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        
        print(f"Model loaded successfully from {model_path}")
    
    def save_model(self, save_path: str):
        """
        Save the trained model
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type in ["random_forest", "logistic"]:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            with open(save_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model_data, f)
        
        elif self.model_type == "transformer":
            torch.save(self.model.state_dict(), save_path.with_suffix('.pt'))
        
        print(f"Model saved to {save_path}")
    
    def load_sample_data(self) -> pd.DataFrame:
        """
        Load sample data for testing
        
        Returns:
            DataFrame with sample genomic features
        """
        # Generate sample data matching our feature structure
        np.random.seed(42)
        
        n_samples = 10
        feature_names = [
            # Methylation features
            'methylation_global_mean', 'methylation_global_std', 'methylation_quality_score',
            'methylation_hypermethylated_probes', 'methylation_hypomethylated_probes',
            'methylation_cancer_likelihood', 'methylation_probe_count',
            # CNA features  
            'cna_total_alterations', 'cna_amplifications', 'cna_deletions',
            'cna_instability_index', 'cna_complexity_score', 'cna_heterogeneity',
            # Fragmentomics features
            'fragmentomics_length_mean', 'fragmentomics_length_std', 'fragmentomics_short_ratio',
            'fragmentomics_nucleosome_signal', 'fragmentomics_complexity',
            # Chromatin features
            'chromatin_accessibility_score', 'chromatin_regulatory_burden',
            'chromatin_peak_count', 'chromatin_signal_noise',
            # Interaction features (simplified)
            'interaction_methyl_cna', 'interaction_fragment_chromatin'
        ]
        
        # Create sample data
        data = []
        for i in range(n_samples):
            if i < 5:  # Cancer samples
                sample = {
                    'sample_id': f'cancer_{i+1}',
                    'label': 1,
                    **{feat: np.random.normal(0.5, 0.3) for feat in feature_names}
                }
                # Adjust cancer-specific patterns
                sample['methylation_cancer_likelihood'] = np.random.normal(0.8, 0.1)
                sample['cna_total_alterations'] = np.random.normal(0.7, 0.2)
            else:  # Control samples
                sample = {
                    'sample_id': f'control_{i-4}',
                    'label': 0,
                    **{feat: np.random.normal(0.0, 0.2) for feat in feature_names}
                }
                # Adjust control-specific patterns
                sample['methylation_cancer_likelihood'] = np.random.normal(0.2, 0.1)
                sample['cna_total_alterations'] = np.random.normal(0.1, 0.1)
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        self.feature_names = feature_names
        
        print(f"Sample data loaded: {df.shape[0]} samples, {len(feature_names)} features")
        return df
    
    def train(self, data: pd.DataFrame, target_column: str = 'label'):
        """
        Train the model on provided data
        
        Args:
            data: DataFrame with features and target
            target_column: Name of the target column
        """
        # Prepare features and target
        feature_cols = [col for col in data.columns if col not in ['sample_id', target_column]]
        X = data[feature_cols].values
        y = data[target_column].values
        
        self.feature_names = feature_cols
        
        # Scale features for classical models
        if self.model_type in ["random_forest", "logistic"]:
            X = self.scaler.fit_transform(X)
            self.model.fit(X, y)
            
            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.model) if self.model_type == "random_forest" else shap.LinearExplainer(self.model, X)
        
        elif self.model_type == "transformer":
            # For transformer, we'd need a more complex training loop
            # This is a simplified version
            print("Transformer training requires more complex setup. Use multimodal_model.py directly.")
        
        print(f"Model trained on {X.shape[0]} samples with {X.shape[1]} features")
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict:
        """
        Make predictions on new data
        
        Args:
            data: Input data (DataFrame or numpy array)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Prepare input data
        if isinstance(data, pd.DataFrame):
            if self.feature_names:
                X = data[self.feature_names].values
            else:
                feature_cols = [col for col in data.columns if col not in ['sample_id', 'label']]
                X = data[feature_cols].values
        else:
            X = data
        
        # Scale if needed
        if self.model_type in ["random_forest", "logistic"] and self.scaler:
            X = self.scaler.transform(X)
        
        # Make predictions
        if self.model_type in ["random_forest", "logistic"]:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            results = {
                'predictions': predictions,
                'probabilities': probabilities,
                'cancer_probability': probabilities[:, 1],
                'predicted_labels': ['Cancer' if p == 1 else 'Control' for p in predictions]
            }
        
        elif self.model_type == "transformer":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                logits, attention_weights = self.model(X_tensor)
                probabilities = torch.softmax(logits, dim=1).numpy()
                predictions = torch.argmax(logits, dim=1).numpy()
                
                results = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'cancer_probability': probabilities[:, 1],
                    'predicted_labels': ['Cancer' if p == 1 else 'Control' for p in predictions],
                    'attention_weights': attention_weights
                }
        
        return results
    
    def analyze_features(self, data: Union[pd.DataFrame, np.ndarray], sample_idx: int = 0) -> Dict:
        """
        Analyze feature importance for predictions
        
        Args:
            data: Input data
            sample_idx: Index of sample to analyze
            
        Returns:
            Dictionary with feature analysis
        """
        if self.explainer is None and self.model_type in ["random_forest", "logistic"]:
            print("SHAP explainer not available. Training needed first.")
            return {}
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            if self.feature_names:
                X = data[self.feature_names].values
            else:
                feature_cols = [col for col in data.columns if col not in ['sample_id', 'label']]
                X = data[feature_cols].values
        else:
            X = data
        
        if self.scaler and self.model_type in ["random_forest", "logistic"]:
            X = self.scaler.transform(X)
        
        results = {}
        
        if self.model_type == "random_forest":
            # Global feature importance
            feature_importance = self.model.feature_importances_
            results['global_importance'] = {
                'features': self.feature_names,
                'importance': feature_importance
            }
            
            # SHAP values for sample
            if len(X) > sample_idx:
                shap_values = self.explainer.shap_values(X[sample_idx:sample_idx+1])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Cancer class
                
                results['shap_analysis'] = {
                    'features': self.feature_names,
                    'shap_values': shap_values[0],
                    'feature_values': X[sample_idx]
                }
        
        elif self.model_type == "transformer":
            # For transformer, we have attention weights
            prediction_results = self.predict(X)
            if 'attention_weights' in prediction_results:
                results['attention_analysis'] = {
                    'attention_weights': prediction_results['attention_weights']
                }
        
        return results
    
    def generate_report(self, data: pd.DataFrame, output_path: str = "cancer_detection_report.html"):
        """
        Generate a comprehensive HTML report
        
        Args:
            data: Input data for analysis
            output_path: Path to save the report
        """
        # Make predictions
        predictions = self.predict(data)
        
        # Analyze features
        feature_analysis = self.analyze_features(data)
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cancer Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .prediction {{ font-size: 18px; font-weight: bold; }}
                .cancer {{ color: red; }}
                .control {{ color: green; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cancer Detection Analysis Report</h1>
                <p>Model Type: {self.model_type.title()}</p>
                <p>Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Prediction Results</h2>
                <table>
                    <tr><th>Sample</th><th>Prediction</th><th>Cancer Probability</th></tr>
        """
        
        for i, (pred, prob) in enumerate(zip(predictions['predicted_labels'], predictions['cancer_probability'])):
            css_class = "cancer" if pred == "Cancer" else "control"
            html_content += f"""
                    <tr>
                        <td>Sample {i+1}</td>
                        <td class="{css_class}">{pred}</td>
                        <td>{prob:.3f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_path}")

def quick_cancer_detection(data_path: str = None, model_type: str = "random_forest") -> Dict:
    """
    Quick cancer detection function for one-line usage
    
    Args:
        data_path: Path to data file (CSV) or None for sample data
        model_type: Type of model to use
        
    Returns:
        Dictionary with results
    """
    detector = CancerDetector(model_type=model_type)
    
    if data_path:
        data = pd.read_csv(data_path)
    else:
        data = detector.load_sample_data()
    
    # Train on the data (for demo purposes)
    detector.train(data)
    
    # Make predictions
    predictions = detector.predict(data)
    
    return {
        'detector': detector,
        'data': data,
        'predictions': predictions
    }

if __name__ == "__main__":
    # Demo usage
    print("Cancer Detection Package Demo")
    print("=" * 40)
    
    # Quick demo
    results = quick_cancer_detection()
    
    print("\nPredictions:")
    for i, (label, prob) in enumerate(zip(results['predictions']['predicted_labels'], 
                                         results['predictions']['cancer_probability'])):
        print(f"Sample {i+1}: {label} (probability: {prob:.3f})")
