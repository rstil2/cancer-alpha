#!/usr/bin/env python3
"""
Multi-Modal Transformer Architecture for Cancer Genomics
========================================================

This module implements advanced transformer architectures for multi-modal genomic data integration,
including TabTransformer for tabular data and Perceiver IO for cross-modal attention.

Patent Protection: Provisional Application No. 63/847,316
Author: Dr. R. Craig Stillwell
Date: July 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
from dataclasses import dataclass


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal transformer architecture"""
    # Model dimensions
d_model: int = 512
d_ff: int = 2048
n_heads: int = 16
n_layers: int = 12
dropout: float = 0.2
    
    # Data modality dimensions
    methylation_dim: int = 20
    mutation_dim: int = 25
    cna_dim: int = 20
    fragmentomics_dim: int = 15
    clinical_dim: int = 10
    icgc_dim: int = 20
    
    # Output configuration
    num_classes: int = 8
    max_seq_length: int = 512
    
    # Advanced settings
    use_cross_modal_attention: bool = True
    use_modality_embeddings: bool = True
    temperature_scaling: bool = True


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TabTransformerEncoder(nn.Module):
    """TabTransformer encoder for tabular genomic data"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        # Feature embeddings for each modality
        self.methylation_embed = nn.Linear(config.methylation_dim, config.d_model)
        self.mutation_embed = nn.Linear(config.mutation_dim, config.d_model)
        self.cna_embed = nn.Linear(config.cna_dim, config.d_model)
        self.fragmentomics_embed = nn.Linear(config.fragmentomics_dim, config.d_model)
        self.clinical_embed = nn.Linear(config.clinical_dim, config.d_model)
        self.icgc_embed = nn.Linear(config.icgc_dim, config.d_model)
        
        # Modality type embeddings
        if config.use_modality_embeddings:
            self.modality_embeddings = nn.Embedding(6, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, dropout=config.dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through TabTransformer
        
        Args:
            data_dict: Dictionary containing modality data
                - methylation: [batch_size, methylation_dim]
                - mutation: [batch_size, mutation_dim]
                - cna: [batch_size, cna_dim]
                - fragmentomics: [batch_size, fragmentomics_dim]
                - clinical: [batch_size, clinical_dim]
                - icgc: [batch_size, icgc_dim]
        
        Returns:
            Encoded representations [batch_size, 6, d_model]
        """
        batch_size = next(iter(data_dict.values())).size(0)
        
        # Embed each modality
        embeddings = []
        embeddings.append(self.methylation_embed(data_dict['methylation']))
        embeddings.append(self.mutation_embed(data_dict['mutation']))
        embeddings.append(self.cna_embed(data_dict['cna']))
        embeddings.append(self.fragmentomics_embed(data_dict['fragmentomics']))
        embeddings.append(self.clinical_embed(data_dict['clinical']))
        embeddings.append(self.icgc_embed(data_dict['icgc']))
        
        # Stack embeddings [batch_size, 6, d_model]
        x = torch.stack(embeddings, dim=1)
        
        # Add modality type embeddings
        if self.config.use_modality_embeddings:
            modality_ids = torch.arange(6, device=x.device).unsqueeze(0).expand(batch_size, -1)
            modality_emb = self.modality_embeddings(modality_ids)
            x = x + modality_emb
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [6, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, 6, d_model]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class AdvancedRegularization(nn.Module):
    """Applies advanced regularization techniques"""
    def __init__(self, dropout: float = 0.3, label_smoothing: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.label_smoothing = label_smoothing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return x

    def smooth_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        return labels * (1 - self.label_smoothing) + self.label_smoothing / num_classes
    """Cross-modal attention mechanism for modality interaction"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Cross-modal attention computation
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
        
        Returns:
            Attended output [batch_size, seq_len_q, d_model]
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Output projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output


class PerceiverIOEncoder(nn.Module):
    """Perceiver IO encoder for general-purpose multi-modal learning"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        # Learnable latent array
        self.latent_dim = config.d_model
        self.num_latents = 64  # Compressed representation
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(config) for _ in range(config.n_layers // 2)
        ])
        
        # Self-attention layers for latents
        self.self_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(config.n_layers // 2)
        ])
        
self.advanced_regularization = AdvancedRegularization(dropout=config.dropout)
    
    def forward(self, encoded_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Perceiver IO
        
        Args:
            encoded_inputs: Encoded modality inputs [batch_size, num_modalities, d_model]
        
        Returns:
            Compressed latent representation [batch_size, num_latents, d_model]
        """
        batch_size = encoded_inputs.size(0)
        
        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Interleave cross-attention and self-attention
        for cross_attn, self_attn in zip(self.cross_attention_layers, self.self_attention_layers):
            # Cross-attention: latents attend to inputs
            latents = cross_attn(latents, encoded_inputs, encoded_inputs)
            
            # Self-attention: latents attend to themselves
            latents = self_attn(latents)
        
        return self.layer_norm(latents)


class MultiModalTransformer(nn.Module):
    """Complete multi-modal transformer architecture for cancer genomics"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        # Core transformer components
        self.tab_transformer = TabTransformerEncoder(config)
        
        if config.use_cross_modal_attention:
            self.perceiver_io = PerceiverIOEncoder(config)
        
        # Classification head - fix dimensional calculation
        if config.use_cross_modal_attention:
            input_dim = config.d_model  # Pooled from perceiver latents
        else:
            input_dim = config.d_model * 6  # Flattened modality representations
            
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_ff // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff // 2, config.num_classes)
        )
        
        # Temperature scaling for calibration
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete architecture
        
        Args:
            data_dict: Dictionary containing modality data
        
        Returns:
            Dictionary containing:
                - logits: Raw classification scores
                - probabilities: Softmax probabilities
                - attention_weights: Attention visualization data
                - latent_representations: Learned representations
        """
        # TabTransformer encoding
        encoded_modalities = self.tab_transformer(data_dict)
        
        # Perceiver IO processing (if enabled)
        if self.config.use_cross_modal_attention:
            latent_repr = self.perceiver_io(encoded_modalities)
            # Global average pooling over latents
            pooled_repr = latent_repr.mean(dim=1)  # [batch_size, d_model]
        else:
            # Global average pooling over modalities
            pooled_repr = encoded_modalities.view(encoded_modalities.size(0), -1)
        
        # Apply regularization
        pooled_repr = self.advanced_regularization(pooled_repr)
        logits = self.classifier(pooled_repr)
        
        # Temperature scaling
        if self.config.temperature_scaling:
            logits = logits / self.temperature
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'encoded_modalities': encoded_modalities,
            'latent_representations': latent_repr if self.config.use_cross_modal_attention else encoded_modalities
        }
    
    def get_attention_weights(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract attention weights for visualization"""
        with torch.no_grad():
            # Get encoded representations
            encoded_modalities = self.tab_transformer(data_dict)
            
            # Extract attention weights from transformer layers
            attention_weights = {}
            
            # This is a simplified version - in practice, you'd need to modify
            # the transformer layers to return attention weights
            attention_weights['modality_attention'] = torch.ones(
                encoded_modalities.size(0), 6, 6
            )  # Placeholder
            
            return attention_weights


class MultiModalEnsemble(nn.Module):
    """Ensemble of multiple multi-modal transformers for robust predictions"""
    
    def __init__(self, configs: List[MultiModalConfig], ensemble_weights: Optional[List[float]] = None):
        super().__init__()
        
        self.models = nn.ModuleList([
            MultiModalTransformer(config) for config in configs
        ])
        
        if ensemble_weights is None:
            ensemble_weights = [1.0 / len(configs)] * len(configs)
        
        self.register_buffer('ensemble_weights', torch.tensor(ensemble_weights))
        
        # Meta-learner for dynamic weighting
        self.meta_learner = nn.Sequential(
            nn.Linear(len(configs) * configs[0].num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, len(configs)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Ensemble forward pass with dynamic weighting"""
        individual_outputs = []
        all_logits = []
        
        # Get predictions from each model
        for model in self.models:
            output = model(data_dict)
            individual_outputs.append(output)
            all_logits.append(output['logits'])
        
        # Stack all predictions
        stacked_logits = torch.stack(all_logits, dim=1)  # [batch_size, num_models, num_classes]
        
        # Dynamic ensemble weighting
        concatenated_logits = stacked_logits.view(stacked_logits.size(0), -1)
        dynamic_weights = self.meta_learner(concatenated_logits)
        dynamic_weights = dynamic_weights.unsqueeze(-1)  # [batch_size, num_models, 1]
        
        # Weighted ensemble
        ensemble_logits = (stacked_logits * dynamic_weights).sum(dim=1)
        ensemble_probs = F.softmax(ensemble_logits, dim=-1)
        
        # Compute prediction confidence
        confidence = torch.max(ensemble_probs, dim=-1)[0]
        
        # Compute prediction uncertainty (entropy)
        uncertainty = -torch.sum(ensemble_probs * torch.log(ensemble_probs + 1e-8), dim=-1)
        
        return {
            'logits': ensemble_logits,
            'probabilities': ensemble_probs,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'individual_outputs': individual_outputs,
            'ensemble_weights': dynamic_weights.squeeze(-1)
        }


def create_production_model(num_classes: int = 8) -> MultiModalTransformer:
    """Create production-ready multi-modal transformer model"""
    config = MultiModalConfig(
        d_model=512,
        d_ff=2048,
        n_heads=16,
        n_layers=12,
        dropout=0.1,
        num_classes=num_classes,
        use_cross_modal_attention=True,
        use_modality_embeddings=True,
        temperature_scaling=True
    )
    
    return MultiModalTransformer(config)


def create_ensemble_model(num_classes: int = 8, num_models: int = 5) -> MultiModalEnsemble:
    """Create ensemble of multi-modal transformers"""
    configs = []
    
    # Create diverse configurations for ensemble
    base_config = MultiModalConfig(num_classes=num_classes)
    
    for i in range(num_models):
        config = MultiModalConfig(
            d_model=256 + i * 64,
            d_ff=1024 + i * 256,
            n_heads=8 + i * 2,
            n_layers=6 + i,
            dropout=0.1 + i * 0.02,
            num_classes=num_classes,
            use_cross_modal_attention=True,
            use_modality_embeddings=True,
            temperature_scaling=True
        )
        configs.append(config)
    
    return MultiModalEnsemble(configs)


if __name__ == "__main__":
    # Example usage
    config = MultiModalConfig()
    model = MultiModalTransformer(config)
    
    # Create sample data
    batch_size = 4
    sample_data = {
        'methylation': torch.randn(batch_size, config.methylation_dim),
        'mutation': torch.randn(batch_size, config.mutation_dim),
        'cna': torch.randn(batch_size, config.cna_dim),
        'fragmentomics': torch.randn(batch_size, config.fragmentomics_dim),
        'clinical': torch.randn(batch_size, config.clinical_dim),
        'icgc': torch.randn(batch_size, config.icgc_dim)
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_data)
        print(f"Output shape: {output['logits'].shape}")
        print(f"Probabilities shape: {output['probabilities'].shape}")
        print(f"Sample probabilities: {output['probabilities'][0]}")
