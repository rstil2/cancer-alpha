#!/usr/bin/env python3
"""
Oncura: Ultra-Advanced Transformer Training on Perfectly Balanced Real TCGA Data
==================================================================================

Trains a state-of-the-art multi-head transformer model for Oncura on the perfectly
balanced real TCGA dataset (1,200 samples, 150 samples per cancer type, 8 cancer types).

This dataset requires NO SMOTE or synthetic augmentation - true balanced real data only.

Dataset: /Users/stillwell/projects/cancer-alpha/data/real_tcga_large/
- Features: real_tcga_features_cleaned.csv (2000 genomic features)
- Labels: real_tcga_labels.csv (8 cancer types)
- Perfectly balanced: 150 samples per type
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score
)
import json
import logging
from pathlib import Path
import time
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oncura_transformer_real_tcga_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiHeadTransformerModel(nn.Module):
    """Ultra-advanced multi-head transformer for genomic classification"""
    
    def __init__(self, input_dim, num_classes, embed_dim=512, num_heads=16, 
                 num_layers=12, dropout=0.2, ff_dim=2048):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Input embedding layer
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Multi-head transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-scale attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim // 4, embed_dim // 8),
            nn.LayerNorm(embed_dim // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 8, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # Embed input
        x_embedded = self.input_embedding(x)  # (batch_size, embed_dim)
        x_embedded = x_embedded.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Add positional encoding
        x_embedded = x_embedded + self.pos_encoding
        
        # Apply transformer encoder
        x_transformed = self.transformer_encoder(x_embedded)
        
        # Attention-based pooling
        x_pooled, _ = self.attention_pool(
            x_transformed.mean(dim=1, keepdim=True),
            x_transformed,
            x_transformed
        )
        x_pooled = x_pooled.squeeze(1)
        
        # Refine features
        x_refined = self.feature_refine(x_pooled)
        
        # Classification
        logits = self.classifier(x_refined)
        
        return logits


class RealTCGATransformerTrainer:
    """Trainer for transformer on real balanced TCGA data"""
    
    def __init__(self, data_dir, output_dir, device='cpu'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        
        # Data containers
        self.X = None
        self.y = None
        self.cancer_types = None
        self.label_encoder = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_data(self):
        """Load real TCGA data"""
        logger.info("Loading real TCGA data...")
        
        # Load features and labels
        features_path = self.data_dir / 'real_tcga_features_cleaned.csv'
        labels_path = self.data_dir / 'real_tcga_labels.csv'
        
        X_df = pd.read_csv(features_path)
        y_df = pd.read_csv(labels_path)
        
        self.X = X_df.values
        self.y = y_df['cancer_type_encoded'].values
        self.cancer_types = y_df['cancer_type'].unique()
        
        logger.info(f"✅ Loaded {len(self.X)} samples with {self.X.shape[1]} features")
        logger.info(f"✅ {len(self.cancer_types)} cancer types: {sorted(self.cancer_types)}")
        
        # Verify balance
        unique, counts = np.unique(self.y, return_counts=True)
        logger.info(f"✅ Class distribution (perfectly balanced real data):")
        for cancer_type, count in zip(self.cancer_types, counts):
            logger.info(f"   {cancer_type}: {count} samples")
    
    def create_data_loaders(self, batch_size=32, test_size=0.2, val_size=0.1):
        """Create train/val/test data loaders"""
        logger.info("Preparing data loaders...")
        
        # Preprocessing: RobustScaler for outlier handling
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        X_scaled = scaler.fit_transform(self.X)
        
        # Stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, self.y, test_size=test_size, stratify=self.y, random_state=42
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        logger.info(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"✅ Scaler saved for preprocessing")
        
        # Save scaler
        joblib.dump(scaler, self.output_dir / 'transformer_scaler.pkl')
        
        return train_loader, val_loader, test_loader, scaler
    
    def train(self, model, train_loader, val_loader, epochs=200, learning_rate=1e-4):
        """Train the transformer model"""
        logger.info(f"🚀 Starting transformer training on real TCGA data for {epochs} epochs")
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 50
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_balanced_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_balanced_acc': []
        }
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                train_targets.extend(batch_labels.cpu().numpy())
            
            scheduler.step()
            
            train_acc = accuracy_score(train_targets, train_preds)
            train_balanced_acc = balanced_accuracy_score(train_targets, train_preds)
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    val_targets.extend(batch_labels.cpu().numpy())
            
            val_acc = accuracy_score(val_targets, val_preds)
            val_balanced_acc = balanced_accuracy_score(val_targets, val_preds)
            val_loss /= len(val_loader)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_balanced_acc'].append(train_balanced_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_balanced_acc'].append(val_balanced_acc)
            
            # Logging
            if (epoch + 1) % 10 == 0 or val_acc > best_val_acc:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f} ({val_balanced_acc:.4f} balanced) | "
                    f"LR: {current_lr:.2e}"
                )
            
            # Model checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                if val_acc > 0.90:
                    logger.info(f"🎉 Checkpoint at {val_acc:.2%} accuracy!")
                    torch.save({
                        'model_state_dict': best_model_state,
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'val_balanced_acc': val_balanced_acc
                    }, self.output_dir / f'transformer_checkpoint_acc_{val_acc:.4f}.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"⏹️  Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")
        
        # Save training history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return best_model_state, best_val_acc, history
    
    def evaluate(self, model, test_loader):
        """Evaluate on test set"""
        logger.info("🧪 Evaluating on test set...")
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = model(batch_features)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(batch_labels.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        test_acc = accuracy_score(all_targets, all_preds)
        test_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        test_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        logger.info(f"\n📊 TEST RESULTS ON REAL TCGA DATA:")
        logger.info(f"   Accuracy: {test_acc:.4f}")
        logger.info(f"   Balanced Accuracy: {test_balanced_acc:.4f}")
        logger.info(f"   F1-Score (weighted): {test_f1:.4f}")
        
        logger.info(f"\n📋 Classification Report:")
        cancer_names = sorted(self.cancer_types)
        report = classification_report(all_targets, all_preds, target_names=cancer_names)
        logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        results = {
            'test_accuracy': float(test_acc),
            'test_balanced_accuracy': float(test_balanced_acc),
            'test_f1_score': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return results, all_preds, all_targets


def main():
    """Main training pipeline"""
    
    # Configuration
    data_dir = '/Users/stillwell/projects/cancer-alpha/data/real_tcga_large'
    output_dir = Path('/Users/stillwell/projects/cancer-alpha/models/oncura_transformer_real_tcga_balanced')
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    logger.info(f"🎯 Oncura: Training Ultra-Advanced Transformer on Real Balanced TCGA Data")
    logger.info(f"📍 Data: {data_dir}")
    logger.info(f"💾 Output: {output_dir}")
    logger.info(f"⚙️  Device: {device}")
    
    # Initialize trainer
    trainer = RealTCGATransformerTrainer(data_dir, str(output_dir), device=device)
    
    # Load data
    trainer.load_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader, scaler = trainer.create_data_loaders(
        batch_size=32, test_size=0.15, val_size=0.15
    )
    
    # Create model
    input_dim = trainer.X.shape[1]
    num_classes = len(trainer.cancer_types)
    
    model = MultiHeadTransformerModel(
        input_dim=input_dim,
        num_classes=num_classes,
        embed_dim=512,
        num_heads=16,
        num_layers=12,
        dropout=0.2,
        ff_dim=2048
    )
    
    model.to(trainer.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Model: {total_params:,} parameters")
    
    # Train
    start_time = time.time()
    best_model_state, best_val_acc, history = trainer.train(
        model, train_loader, val_loader,
        epochs=200,
        learning_rate=1e-4
    )
    training_time = time.time() - start_time
    
    logger.info(f"⏱️  Training time: {training_time/60:.2f} minutes")
    
    # Evaluate
    model.load_state_dict(best_model_state)
    results, test_preds, test_targets = trainer.evaluate(model, test_loader)
    
    # Save results
    results['best_validation_accuracy'] = float(best_val_acc)
    results['training_time_seconds'] = training_time
    results['model_parameters'] = total_params
    results['data_samples'] = len(trainer.X)
    results['num_features'] = input_dim
    results['num_classes'] = num_classes
    results['cancer_types'] = sorted(trainer.cancer_types)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'model_config': {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'embed_dim': 512,
            'num_heads': 16,
            'num_layers': 12,
            'dropout': 0.2,
            'ff_dim': 2048
        },
        'results': results
    }, output_dir / 'transformer_real_tcga_balanced.pth')
    
    logger.info(f"\n✅ Oncura results saved to {output_dir}")
    logger.info(f"📊 Oncura Final Test Balanced Accuracy: {results['test_balanced_accuracy']:.4f} ({results['test_balanced_accuracy']*100:.2f}%)")
    
    if results['test_balanced_accuracy'] >= 0.96:
        logger.info("🎉🎉🎉 EXCELLENT! Oncura transformer achieved 96%+ accuracy on real TCGA data!")
    elif results['test_balanced_accuracy'] >= 0.90:
        logger.info("🔥 Outstanding! Oncura transformer achieved 90%+ accuracy on real TCGA data!")
    else:
        logger.info(f"📈 Oncura transformer achieved {results['test_balanced_accuracy']*100:.2f}% accuracy on real TCGA data")


if __name__ == '__main__':
    main()
