#!/usr/bin/env python3
"""
Enhanced Model Interpretability for Multi-Modal Transformers
============================================================

This module provides advanced interpretability features specifically designed 
for multi-modal transformer models in cancer genomics.

Author: Oncura Research Team  
Date: July 28, 2025
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import sys
import os
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_transformer import MultiModalTransformer, MultiModalConfig

class TransformerExplainer:
    """
    Advanced explainer for multi-modal transformer models
    """
    
    def __init__(self, model: MultiModalTransformer, scalers: Dict, feature_names: List[str]):
        """
        Initialize the transformer explainer
        
        Args:
            model: Trained MultiModalTransformer model
            scalers: Dictionary of scalers for each modality
            feature_names: List of feature names
        """
        self.model = model
        self.scalers = scalers
        self.feature_names = feature_names
        self.cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
        
        # Modality boundaries
        self.modality_boundaries = {
            'methylation': (0, 20),
            'mutation': (20, 45), 
            'cna': (45, 65),
            'fragmentomics': (65, 80),
            'clinical': (80, 90),
            'icgc': (90, 110)
        }
        
    def extract_attention_weights(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from the transformer model
        
        Args:
            input_data: Input data array [batch_size, n_features]
            
        Returns:
            Dictionary containing attention weights for different layers
        """
        self.model.eval()
        
        # Preprocess input data
        processed_data = self._preprocess_for_model(input_data)
        
        # Create data dictionary for transformer
        data_dict = self._create_data_dict(processed_data)
        
        attention_weights = {}
        
        with torch.no_grad():
            # Forward pass through tab transformer
            encoded_modalities = self.model.tab_transformer(data_dict)
            
            # If using cross-modal attention, extract those weights too
            if self.model.config.use_cross_modal_attention:
                latent_repr = self.model.perceiver_io(encoded_modalities)
                
                # Extract cross-modal attention weights (simplified)
                attention_weights['cross_modal'] = self._extract_cross_modal_weights(
                    encoded_modalities, latent_repr
                )
            
            # Extract self-attention weights from tab transformer
            attention_weights['self_attention'] = self._extract_self_attention_weights(
                encoded_modalities
            )
            
        return attention_weights
    
    def _extract_cross_modal_weights(self, encoded_modalities: torch.Tensor, 
                                   latent_repr: torch.Tensor) -> np.ndarray:
        """Extract cross-modal attention weights"""
        # Simplified attention weight extraction
        batch_size, num_modalities, d_model = encoded_modalities.shape
        
        # Calculate similarity between latents and modalities as proxy for attention
        similarities = torch.zeros(batch_size, num_modalities)
        
        for i in range(num_modalities):
            # Cosine similarity between latent representation and each modality
            modality_repr = encoded_modalities[:, i, :]  # [batch_size, d_model]
            latent_mean = latent_repr.mean(dim=1)        # [batch_size, d_model]
            
            # Normalize vectors
            modality_norm = torch.nn.functional.normalize(modality_repr, dim=1)
            latent_norm = torch.nn.functional.normalize(latent_mean, dim=1)
            
            # Compute cosine similarity
            similarities[:, i] = torch.sum(modality_norm * latent_norm, dim=1)
        
        # Convert to attention-like weights using softmax
        attention_weights = torch.nn.functional.softmax(similarities, dim=1)
        
        return attention_weights.numpy()
    
    def _extract_self_attention_weights(self, encoded_modalities: torch.Tensor) -> np.ndarray:
        """Extract self-attention weights between modalities"""
        batch_size, num_modalities, d_model = encoded_modalities.shape
        
        # Compute pairwise similarities between modalities
        attention_matrix = torch.zeros(batch_size, num_modalities, num_modalities)
        
        for i in range(num_modalities):
            for j in range(num_modalities):
                # Compute similarity between modality i and j
                mod_i = encoded_modalities[:, i, :]
                mod_j = encoded_modalities[:, j, :]
                
                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(mod_i, mod_j, dim=1)
                attention_matrix[:, i, j] = similarity
        
        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(attention_matrix, dim=-1)
        
        return attention_weights.numpy()
    
    def compute_gradient_attributions(self, input_data: np.ndarray, 
                                    target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute gradient-based feature attributions
        
        Args:
            input_data: Input data array [1, n_features] 
            target_class: Target class for attribution (if None, uses predicted class)
            
        Returns:
            Dictionary containing gradient attributions by modality
        """
        self.model.eval()
        
        # Preprocess and prepare data
        processed_data = self._preprocess_for_model(input_data)
        data_dict = self._create_data_dict(processed_data, requires_grad=True)
        
        # Forward pass
        outputs = self.model(data_dict)
        predictions = outputs['probabilities']
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = torch.argmax(predictions, dim=1).item()
        
        # Compute gradients
        target_output = predictions[0, target_class]
        target_output.backward()
        
        # Extract gradients for each modality
        attributions = {}
        for modality, (start, end) in self.modality_boundaries.items():
            if modality in data_dict and data_dict[modality].grad is not None:
                grad = data_dict[modality].grad[0].numpy()  # [modality_features]
                # Multiply by input (Gradient * Input)
                input_vals = data_dict[modality][0].detach().numpy()
                attributions[modality] = grad * input_vals
            else:
                attributions[modality] = np.zeros(end - start)
        
        return attributions
    
    def integrated_gradients(self, input_data: np.ndarray, baseline: Optional[np.ndarray] = None,
                           steps: int = 50, target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute Integrated Gradients attributions
        
        Args:
            input_data: Input data array [1, n_features]
            baseline: Baseline for integration (if None, uses zeros)
            steps: Number of integration steps
            target_class: Target class for attribution
            
        Returns:
            Dictionary containing integrated gradients by modality
        """
        if baseline is None:
            baseline = np.zeros_like(input_data)
        
        # Generate interpolated inputs
        alphas = np.linspace(0, 1, steps)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated_inputs.append(interpolated)
        
        interpolated_inputs = np.vstack(interpolated_inputs)
        
        # Compute gradients for all interpolated inputs
        all_gradients = []
        for i in range(steps):
            gradients = self.compute_gradient_attributions(
                interpolated_inputs[i:i+1], target_class
            )
            all_gradients.append(gradients)
        
        # Average gradients and multiply by (input - baseline)
        integrated_grads = {}
        for modality in self.modality_boundaries.keys():
            # Average gradients across interpolation steps
            avg_grads = np.mean([grad[modality] for grad in all_gradients], axis=0)
            
            # Multiply by input difference
            start, end = self.modality_boundaries[modality]
            input_diff = input_data[0, start:end] - baseline[0, start:end]
            
            # Apply preprocessing if needed
            if modality in self.scalers:
                input_diff = self.scalers[modality].transform(input_diff.reshape(1, -1))[0]
            
            integrated_grads[modality] = avg_grads * input_diff
        
        return integrated_grads
    
    def visualize_attention_weights(self, attention_weights: Dict[str, np.ndarray], 
                                  sample_idx: int = 0) -> go.Figure:
        """
        Visualize attention weights as heatmaps
        
        Args:
            attention_weights: Dictionary of attention weights
            sample_idx: Index of sample to visualize
            
        Returns:
            Plotly figure with attention visualizations
        """
        modality_names = list(self.modality_boundaries.keys())
        
        if 'cross_modal' in attention_weights:
            # Cross-modal attention heatmap
            cross_modal = attention_weights['cross_modal'][sample_idx]
            
            fig = go.Figure(data=go.Heatmap(
                z=cross_modal.reshape(1, -1),
                x=modality_names,
                y=['Attention Weight'],
                colorscale='Viridis',
                text=[[f'{val:.3f}' for val in cross_modal]],
                texttemplate='%{text}',
                textfont={'size': 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Cross-Modal Attention Weights',
                xaxis_title='Genomic Modalities',
                yaxis_title='',
                height=200
            )
            
            return fig
        
        elif 'self_attention' in attention_weights:
            # Self-attention matrix
            self_attn = attention_weights['self_attention'][sample_idx]
            
            fig = go.Figure(data=go.Heatmap(
                z=self_attn,
                x=modality_names,
                y=modality_names,
                colorscale='RdBu',
                zmid=0,
                text=[[f'{val:.3f}' for val in row] for row in self_attn],
                texttemplate='%{text}',
                textfont={'size': 10}
            ))
            
            fig.update_layout(
                title='Self-Attention Between Modalities',
                xaxis_title='Target Modality',
                yaxis_title='Source Modality',
                height=500
            )
            
            return fig
        
        return go.Figure()
    
    def visualize_feature_attributions(self, attributions: Dict[str, np.ndarray], 
                                     method_name: str = "Gradient Attribution") -> go.Figure:
        """
        Visualize feature attributions by modality
        
        Args:
            attributions: Dictionary of feature attributions by modality  
            method_name: Name of the attribution method
            
        Returns:
            Plotly figure with attribution visualizations
        """
        # Prepare data for visualization
        modalities = []
        values = []
        feature_indices = []
        
        for modality, attrs in attributions.items():
            for i, attr in enumerate(attrs):
                modalities.append(modality.capitalize())
                values.append(attr)
                feature_indices.append(f"{modality}_{i}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Modality': modalities,
            'Attribution': values,
            'Feature': feature_indices,
            'Abs_Attribution': np.abs(values)
        })
        
        # Sort by absolute attribution value
        df = df.sort_values('Abs_Attribution', ascending=False).head(20)
        
        # Create bar plot
        fig = px.bar(
            df,
            x='Attribution',
            y='Feature',
            color='Modality',
            orientation='h',
            title=f'Top 20 Feature Attributions ({method_name})',
            labels={'Attribution': f'{method_name} Score', 'Feature': 'Genomic Features'}
        )
        
        fig.update_layout(height=600, showlegend=True)
        
        return fig
    
    def modality_importance_analysis(self, attributions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze importance of each genomic modality
        
        Args:
            attributions: Dictionary of feature attributions by modality
            
        Returns:
            Dictionary with modality importance scores
        """
        modality_importance = {}
        
        for modality, attrs in attributions.items():
            # Use sum of absolute attributions as importance score
            importance = np.sum(np.abs(attrs))
            modality_importance[modality] = float(importance)
        
        # Normalize to percentages
        total_importance = sum(modality_importance.values())
        if total_importance > 0:
            modality_importance = {
                k: (v / total_importance) * 100 
                for k, v in modality_importance.items()
            }
        
        return modality_importance
    
    def generate_biological_insights(self, attributions: Dict[str, np.ndarray], 
                                   predicted_class: int, confidence: float) -> List[str]:
        """
        Generate biological insights based on feature attributions
        
        Args:
            attributions: Feature attributions by modality
            predicted_class: Predicted cancer class
            confidence: Prediction confidence
            
        Returns:
            List of biological insight strings
        """
        insights = []
        cancer_type = self.cancer_types[predicted_class]
        
        # Analyze modality importance
        modality_importance = self.modality_importance_analysis(attributions)
        top_modality = max(modality_importance, key=modality_importance.get)
        
        # General prediction insight
        if confidence > 0.8:
            insights.append(f"🎯 **High confidence {cancer_type} prediction** (confidence: {confidence:.1%}) suggests strong genomic signature")
        elif confidence > 0.6:
            insights.append(f"🔍 **Moderate confidence {cancer_type} prediction** (confidence: {confidence:.1%}) indicates possible cancer signature")
        else:
            insights.append(f"⚠️ **Low confidence prediction** (confidence: {confidence:.1%}) - results should be interpreted cautiously")
        
        # Modality-specific insights
        if modality_importance.get('methylation', 0) > 25:
            insights.append(f"🧬 **DNA methylation patterns** show strong contribution ({modality_importance['methylation']:.1f}%) - suggests epigenetic alterations characteristic of {cancer_type}")
        
        if modality_importance.get('mutation', 0) > 25:
            insights.append(f"🔬 **Mutation signature** is highly influential ({modality_importance['mutation']:.1f}%) - indicates genomic instability typical of {cancer_type}")
        
        if modality_importance.get('cna', 0) > 25:
            insights.append(f"📊 **Copy number alterations** drive prediction ({modality_importance['cna']:.1f}%) - suggests chromosomal instability in {cancer_type}")
        
        if modality_importance.get('fragmentomics', 0) > 25:
            insights.append(f"🧪 **Fragmentomics profile** shows strong signal ({modality_importance['fragmentomics']:.1f}%) - potential for liquid biopsy applications")
        
        if modality_importance.get('clinical', 0) > 20:
            insights.append(f"👩‍⚕️ **Clinical features** contribute significantly ({modality_importance['clinical']:.1f}%) - patient characteristics align with {cancer_type}")
        
        # Cancer-type-specific insights
        if cancer_type == 'BRCA' and modality_importance.get('methylation', 0) > 20:
            insights.append("🎀 **BRCA-specific insight**: Elevated methylation patterns consistent with hormone receptor status alterations")
        
        elif cancer_type == 'LUAD' and modality_importance.get('mutation', 0) > 30:
            insights.append("🫁 **LUAD-specific insight**: High mutation burden typical of smoking-related lung adenocarcinoma")
        
        elif cancer_type == 'KIRC' and modality_importance.get('cna', 0) > 25:
            insights.append("🔴 **KIRC-specific insight**: Copy number alterations consistent with VHL pathway disruption in renal cell carcinoma")
        
        # Multi-modal integration insight
        active_modalities = [mod for mod, imp in modality_importance.items() if imp > 15]
        if len(active_modalities) >= 3:
            insights.append(f"🔗 **Multi-modal integration**: {len(active_modalities)} genomic modalities contribute to prediction, demonstrating comprehensive molecular characterization")
        
        return insights
    
    def _preprocess_for_model(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess input data using modality-specific scalers"""
        if len(self.scalers) == 6:  # Advanced scalers
            # Apply separate scalers for each modality
            methylation = self.scalers['methylation'].transform(input_data[:, :20])
            mutation = self.scalers['mutation'].transform(input_data[:, 20:45])
            cna = self.scalers['cna'].transform(input_data[:, 45:65])
            fragmentomics = self.scalers['fragmentomics'].transform(input_data[:, 65:80])
            clinical = self.scalers['clinical'].transform(input_data[:, 80:90])
            icgc = self.scalers['icgc'].transform(input_data[:, 90:110])
            
            return np.concatenate([methylation, mutation, cna, fragmentomics, clinical, icgc], axis=1)
        else:
            # Use main scaler if available
            if 'main' in self.scalers:
                return self.scalers['main'].transform(input_data)
            else:
                return input_data
    
    def _create_data_dict(self, processed_data: np.ndarray, requires_grad: bool = False) -> Dict[str, torch.Tensor]:
        """Create data dictionary for transformer model"""
        data_dict = {
            'methylation': torch.FloatTensor(processed_data[:, :20]),
            'mutation': torch.FloatTensor(processed_data[:, 20:45]),
            'cna': torch.FloatTensor(processed_data[:, 45:65]),
            'fragmentomics': torch.FloatTensor(processed_data[:, 65:80]),
            'clinical': torch.FloatTensor(processed_data[:, 80:90]),
            'icgc': torch.FloatTensor(processed_data[:, 90:110])
        }
        
        if requires_grad:
            for key in data_dict:
                data_dict[key].requires_grad_(True)
        
        return data_dict

def main():
    """Example usage of the transformer explainer"""
    # This would normally load a real model and data
    print("TransformerExplainer module ready for integration")
    print("Features:")
    print("- Attention weight extraction")
    print("- Gradient-based attributions")  
    print("- Integrated gradients")
    print("- Biological insight generation")
    print("- Multi-modal importance analysis")

if __name__ == '__main__':
    main()
