#!/usr/bin/env python3
"""
Multi-Modal Transformer Model for Cancer Genomics
Implements attention-based architecture for ctDNA analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class MultiModalDataset(Dataset):
    """Dataset class for multi-modal cancer genomics data"""
    
    def __init__(self, features, labels=None, scaler=None, fit_scaler=True):
        self.features = features
        self.labels = labels
        
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            self.scaler = scaler
            
        if fit_scaler:
            self.features = self.scaler.fit_transform(features)
        else:
            self.features = self.scaler.transform(features)
            
        self.features = torch.FloatTensor(self.features)
        
        if labels is not None:
            self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class MultiModalAttention(nn.Module):
    """Multi-head attention for multi-modal feature fusion"""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x)
        
        return output, attention_weights

class CancerGenomicsTransformer(pl.LightningModule):
    """Multi-modal transformer for cancer detection from ctDNA"""
    
    def __init__(self, 
                 input_dim=46,
                 d_model=256,
                 n_heads=8,
                 n_layers=4,
                 n_classes=2,
                 dropout=0.1,
                 learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Modality-specific encoders
        self.methylation_encoder = self._create_modality_encoder(d_model)
        self.fragmentomics_encoder = self._create_modality_encoder(d_model)
        self.cna_encoder = self._create_modality_encoder(d_model)
        
        # Multi-modal attention layers
        self.attention_layers = nn.ModuleList([
            MultiModalAttention(d_model, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # Metrics tracking
        self.train_accuracy = []
        self.val_accuracy = []
        self.attention_weights_history = []
        
    def _create_modality_encoder(self, d_model):
        """Create encoder for specific modality"""
        return nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Split features by modality (approximate)
        methylation_features = x[:, :self.hparams.d_model//3]
        fragmentomics_features = x[:, self.hparams.d_model//3:2*self.hparams.d_model//3]
        cna_features = x[:, 2*self.hparams.d_model//3:]
        
        # Encode each modality
        methylation_encoded = self.methylation_encoder(methylation_features)
        fragmentomics_encoded = self.fragmentomics_encoder(fragmentomics_features)
        cna_encoded = self.cna_encoder(cna_features)
        
        # Stack modalities for attention
        modality_stack = torch.stack([
            methylation_encoded, 
            fragmentomics_encoded, 
            cna_encoded
        ], dim=1)  # [batch_size, 3, d_model]
        
        # Apply attention layers
        attention_weights_all = []
        for attention_layer in self.attention_layers:
            modality_stack, attention_weights = attention_layer(modality_stack)
            attention_weights_all.append(attention_weights)
        
        # Global pooling across modalities
        pooled_features = torch.mean(modality_stack, dim=1)  # [batch_size, d_model]
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits, attention_weights_all
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, attention_weights = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, attention_weights = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Store attention weights for analysis
        if batch_idx == 0:  # Store first batch attention weights
            self.attention_weights_history.append(attention_weights)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc, 'preds': preds, 'targets': y}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, attention_weights = self(x)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        return {
            'preds': preds,
            'probs': probs,
            'targets': y,
            'attention_weights': attention_weights
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class ModelTrainer:
    """Training pipeline for cancer genomics model"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.trainer = None
        
    def prepare_synthetic_data(self, n_samples=1000):
        """Generate synthetic data for demonstration"""
        print("Generating synthetic multi-modal cancer genomics data...")
        
        np.random.seed(42)
        
        # Generate features for cancer and control samples
        n_cancer = n_samples // 2
        n_control = n_samples - n_cancer
        
        # Cancer samples (elevated methylation, altered fragmentomics, more CNAs)
        cancer_features = np.random.multivariate_normal(
            mean=np.array([0.7, 0.3, 0.6] + [0.0] * 43),  # 46 features total
            cov=np.eye(46) * 0.1,
            size=n_cancer
        )
        
        # Control samples (normal patterns)
        control_features = np.random.multivariate_normal(
            mean=np.array([0.5, 0.2, 0.3] + [0.0] * 43),
            cov=np.eye(46) * 0.05,
            size=n_control
        )
        
        # Combine features and labels
        features = np.vstack([cancer_features, control_features])
        labels = np.array([1] * n_cancer + [0] * n_control)
        
        # Add some noise and realistic patterns
        features = np.abs(features)  # Ensure positive values
        features = np.clip(features, 0, 2)  # Reasonable range
        
        print(f"Generated {n_samples} samples with {features.shape[1]} features")
        return features, labels
    
    def train_model(self, features, labels, test_size=0.2, val_size=0.1):
        """Train the multi-modal transformer model"""
        print("Starting model training...")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Create datasets
        scaler = StandardScaler()
        train_dataset = MultiModalDataset(X_train, y_train, scaler, fit_scaler=True)
        val_dataset = MultiModalDataset(X_val, y_val, scaler, fit_scaler=False)
        test_dataset = MultiModalDataset(X_test, y_test, scaler, fit_scaler=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        self.model = CancerGenomicsTransformer(
            input_dim=features.shape[1],
            d_model=256,
            n_heads=8,
            n_layers=4,
            n_classes=2,
            dropout=0.1,
            learning_rate=1e-4
        )
        
        # Initialize trainer
        self.trainer = pl.Trainer(
            max_epochs=50,
            accelerator='auto',
            devices=1,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        # Train model
        self.trainer.fit(self.model, train_loader, val_loader)
        
        # Test model
        test_results = self.trainer.test(self.model, test_loader)
        
        # Save model
        model_path = self.output_dir / "cancer_genomics_model.ckpt"
        self.trainer.save_checkpoint(model_path)
        print(f"Model saved to: {model_path}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'test_results': test_results,
            'scaler': scaler
        }

def main():
    """Main training pipeline"""
    print("Cancer Genomics Multi-Modal Transformer Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    features, labels = trainer.prepare_synthetic_data(n_samples=1000)
    
    # Train model
    results = trainer.train_model(features, labels)
    
    print("\nTraining completed successfully!")
    print("Model and results saved in 'results' directory.")
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()
