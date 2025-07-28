#!/usr/bin/env python3
"""
Enhanced Multi-Modal Transformer Architecture for Cancer Genomics
==================================================================

Advanced version with improved capacity, regularization, and training techniques.

Author: Cancer Alpha Research Team
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
class EnhancedMultiModalConfig:
    """Enhanced configuration for multi-modal transformer architecture"""
    # Increased model dimensions for better capacity
    d_model: int = 768
    d_ff: int = 3072
    n_heads: int = 12
    n_layers: int = 16
    dropout: float = 0.15
    
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
    
    # Enhanced settings
    use_cross_modal_attention: bool = True
    use_modality_embeddings: bool = True
    temperature_scaling: bool = True
    use_label_smoothing: bool = True
    label_smoothing_factor: float = 0.1
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_focal_loss: bool = True
    focal_loss_alpha: float = 1.0
    focal_loss_gamma: float = 2.0


class AdvancedModalityEncoder(nn.Module):
    """Enhanced modality-specific encoder with pre-processing"""
    
    def __init__(self, input_dim: int, output_dim: int, modality_name: str, dropout: float = 0.1):
        super().__init__()
        self.modality_name = modality_name
        
        # Modality-specific normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Multi-layer feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection for larger input dimensions
        if input_dim == output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        # Feature extraction
        features = self.feature_extractor(x_norm)
        
        # Residual connection
        residual = self.residual(x)
        
        return features + residual * 0.1  # Scaled residual


class MultiHeadCrossModalAttention(nn.Module):
    """Enhanced cross-modal attention with multiple attention heads"""
    
    def __init__(self, config: EnhancedMultiModalConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # Separate attention heads for different interaction types
        self.biological_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.statistical_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Biological attention (focuses on known biological interactions)
        bio_attn, _ = self.biological_attention(query, key, value)
        
        # Statistical attention (learns novel correlations)
        stat_attn, _ = self.statistical_attention(query, key, value)
        
        # Fusion
        combined = torch.cat([bio_attn, stat_attn], dim=-1)
        fused = self.fusion(combined)
        
        # Residual connection and normalization
        return self.layer_norm(fused + query)


class EnhancedPerceiverIO(nn.Module):
    """Enhanced Perceiver IO with adaptive latent size"""
    
    def __init__(self, config: EnhancedMultiModalConfig):
        super().__init__()
        self.config = config
        
        # Adaptive latent array size based on complexity
        self.latent_dim = config.d_model
        self.num_latents = min(128, config.d_model // 4)  # Adaptive size
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim) * 0.02)
        
        # Enhanced cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            MultiHeadCrossModalAttention(config) for _ in range(config.n_layers // 3)
        ])
        
        # Self-attention layers with different configurations
        self.self_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LN for better training stability
            ) for _ in range(config.n_layers // 3)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, encoded_inputs: torch.Tensor) -> torch.Tensor:
        batch_size = encoded_inputs.size(0)
        
        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Progressive refinement through attention layers
        for cross_attn, self_attn in zip(self.cross_attention_layers, self.self_attention_layers):
            # Cross-attention: latents attend to inputs
            latents = cross_attn(latents, encoded_inputs, encoded_inputs)
            
            # Self-attention: latents refine themselves
            latents = self_attn(latents)
        
        # Output projection
        return self.output_projection(latents)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, num_classes: int = 8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MixUpAugmentation:
    """MixUp data augmentation for genomic data"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, data_dict: Dict[str, torch.Tensor], targets: torch.Tensor):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = targets.size(0)
        index = torch.randperm(batch_size)
        
        # Mix inputs
        mixed_data = {}
        for key, value in data_dict.items():
            mixed_data[key] = lam * value + (1 - lam) * value[index, :]
        
        # Mix targets
        mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        return mixed_data, mixed_targets, lam


class EnhancedMultiModalTransformer(nn.Module):
    """Enhanced multi-modal transformer with advanced training techniques"""
    
    def __init__(self, config: EnhancedMultiModalConfig):
        super().__init__()
        self.config = config
        
        # Enhanced modality encoders
        self.modality_encoders = nn.ModuleDict({
            'methylation': AdvancedModalityEncoder(
                config.methylation_dim, config.d_model, 'methylation', config.dropout
            ),
            'mutation': AdvancedModalityEncoder(
                config.mutation_dim, config.d_model, 'mutation', config.dropout
            ),
            'cna': AdvancedModalityEncoder(
                config.cna_dim, config.d_model, 'cna', config.dropout
            ),
            'fragmentomics': AdvancedModalityEncoder(
                config.fragmentomics_dim, config.d_model, 'fragmentomics', config.dropout
            ),
            'clinical': AdvancedModalityEncoder(
                config.clinical_dim, config.d_model, 'clinical', config.dropout
            ),
            'icgc': AdvancedModalityEncoder(
                config.icgc_dim, config.d_model, 'icgc', config.dropout
            ),
        })
        
        # Enhanced Perceiver IO
        if config.use_cross_modal_attention:
            self.perceiver_io = EnhancedPerceiverIO(config)
        
        # Advanced classification head with multiple pathways
        classifier_input_dim = config.d_model if config.use_cross_modal_attention else config.d_model * 6
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_ff // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff // 2, config.d_ff // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff // 4, config.num_classes)
        )
        
        # Loss functions
        if config.use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=config.focal_loss_alpha,
                gamma=config.focal_loss_gamma,
                num_classes=config.num_classes
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing_factor)
        
        # MixUp augmentation
        if config.use_mixup:
            self.mixup = MixUpAugmentation(alpha=config.mixup_alpha)
        
        # Temperature scaling
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights with improved strategy
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Improved weight initialization"""
        if isinstance(module, nn.Linear):
            # Use He initialization for GELU activation
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(self, data_dict: Dict[str, torch.Tensor], 
                targets: Optional[torch.Tensor] = None,
                training: bool = False) -> Dict[str, torch.Tensor]:
        
        # Apply MixUp during training
        if training and self.config.use_mixup and targets is not None:
            data_dict, targets, mixup_lambda = self.mixup(data_dict, targets)
        
        # Encode each modality
        encoded_modalities = []
        for modality_name, encoder in self.modality_encoders.items():
            encoded = encoder(data_dict[modality_name])
            encoded_modalities.append(encoded)
        
        # Stack modalities [batch_size, num_modalities, d_model]
        encoded_stack = torch.stack(encoded_modalities, dim=1)
        
        # Cross-modal processing
        if self.config.use_cross_modal_attention:
            latent_repr = self.perceiver_io(encoded_stack)
            # Advanced pooling: combination of mean, max, and attention-weighted
            mean_pool = latent_repr.mean(dim=1)
            max_pool = latent_repr.max(dim=1)[0]
            
            # Attention-weighted pooling
            attention_weights = F.softmax(
                torch.sum(latent_repr * mean_pool.unsqueeze(1), dim=-1), dim=-1
            )
            attention_pool = torch.sum(
                latent_repr * attention_weights.unsqueeze(-1), dim=1
            )
            
            # Combine different pooling strategies
            pooled_repr = (mean_pool + max_pool + attention_pool) / 3
        else:
            pooled_repr = encoded_stack.view(encoded_stack.size(0), -1)
        
        # Classification
        logits = self.classifier(pooled_repr)
        
        # Temperature scaling
        if self.config.temperature_scaling:
            logits = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        
        # Compute probabilities and predictions
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = self.loss_fn(logits, targets.long())
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions,
            'loss': loss,
            'encoded_modalities': encoded_stack,
            'latent_representations': latent_repr if self.config.use_cross_modal_attention else encoded_stack
        }
    
    def get_attention_weights(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract attention weights for interpretability"""
        with torch.no_grad():
            # Get encoded representations
            encoded_modalities = []
            for modality_name, encoder in self.modality_encoders.items():
                encoded = encoder(data_dict[modality_name])
                encoded_modalities.append(encoded)
            
            encoded_stack = torch.stack(encoded_modalities, dim=1)
            
            # Compute modality importance
            modality_norms = torch.norm(encoded_stack, dim=-1)  # [batch_size, num_modalities]
            modality_importance = F.softmax(modality_norms, dim=-1)
            
            return {
                'modality_importance': modality_importance,
                'modality_names': list(self.modality_encoders.keys()),
                'encoded_representations': encoded_stack
            }


def create_enhanced_model(num_classes: int = 8) -> EnhancedMultiModalTransformer:
    """Create enhanced multi-modal transformer model"""
    config = EnhancedMultiModalConfig(
        d_model=768,
        d_ff=3072,
        n_heads=12,
        n_layers=16,
        dropout=0.15,
        num_classes=num_classes,
        use_cross_modal_attention=True,
        use_modality_embeddings=True,
        temperature_scaling=True,
        use_label_smoothing=True,
        label_smoothing_factor=0.1,
        use_mixup=True,
        mixup_alpha=0.2,
        use_focal_loss=True,
        focal_loss_alpha=1.0,
        focal_loss_gamma=2.0
    )
    
    return EnhancedMultiModalTransformer(config)


if __name__ == "__main__":
    # Test the enhanced model
    model = create_enhanced_model()
    
    # Create sample data
    batch_size = 4
    sample_data = {
        'methylation': torch.randn(batch_size, 20),
        'mutation': torch.randn(batch_size, 25),
        'cna': torch.randn(batch_size, 20),
        'fragmentomics': torch.randn(batch_size, 15),
        'clinical': torch.randn(batch_size, 10),
        'icgc': torch.randn(batch_size, 20)
    }
    
    targets = torch.randint(0, 8, (batch_size,))
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_data, targets, training=False)
        print(f"Logits shape: {output['logits'].shape}")
        print(f"Probabilities shape: {output['probabilities'].shape}")
        print(f"Loss: {output['loss']}")
        print(f"Sample probabilities: {output['probabilities'][0]}")
