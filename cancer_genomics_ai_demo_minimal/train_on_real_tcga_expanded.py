#!/usr/bin/env python3
"""
Training on Real TCGA-Derived Expanded Data
==========================================

This script trains the ultra-advanced transformer model on the expanded
dataset derived from real TCGA mutation patterns.

Author: Oncura Research Team
Date: July 28, 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import logging
from pathlib import Path
import time
import joblib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the ultra-advanced transformer from the existing code
import sys
sys.path.append('.')

class UltraAdvancedTransformer(nn.Module):
    """Ultra-advanced transformer for real TCGA data"""
    
    def __init__(self, input_dim=116, num_classes=8, config=None):
        super().__init__()
        
        if config is None:
            config = {
                'embed_dim': 768,
                'n_heads': 24,
                'n_layers': 16,
                'dropout': 0.15,
                'ff_hidden_dim': 2048,
                'learning_rate': 0.0003,
                'weight_decay': 0.0005,
                'use_residual_mlp': True,
                'use_layer_scale': True,
                'use_stochastic_depth': True
            }
        
        self.config = config
        self.embed_dim = config['embed_dim']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.n_heads,
                dim_feedforward=config['ff_hidden_dim'],
                dropout=self.dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(self.n_layers)
        ])
        
        # Multi-head attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.LayerNorm(self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(self.dropout / 2),
            nn.Linear(self.embed_dim // 4, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Attention pooling
        pooled, _ = self.attention_pooling(x, x, x)
        pooled = pooled.squeeze(1)
        
        # Classification
        output = self.classifier(pooled)
        return output

def create_data_loaders(X, y, batch_size=64, test_size=0.2, val_size=0.2):
    """Create data loaders with proper train/val/test splits"""
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )
    
    # Create balanced samplers
    class_weights = 1.0 / np.bincount(y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

def train_model(model, train_loader, val_loader, config, device):
    """Train the model"""
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    patience = 50
    patience_counter = 0
    
    training_history = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': []
    }
    
    logger.info("Starting training on real TCGA-derived data...")
    
    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        # Calculate accuracies
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Store history
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['train_loss'].append(train_loss / len(train_loader))
        training_history['val_loss'].append(val_loss / len(val_loader))
        
        # Logging
        if epoch % 5 == 0 or val_acc > best_val_acc:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch+1}/{config['max_epochs']} - "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"LR: {current_lr:.8f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint for high accuracy
            if val_acc > 0.85:
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': config,
                    'val_accuracy': val_acc,
                    'epoch': epoch,
                    'training_history': training_history
                }, f'models/real_tcga_checkpoint_acc_{val_acc:.4f}.pth')
                logger.info(f"🚀 Checkpoint saved at {val_acc:.2%} accuracy on real TCGA data!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc, best_model_state, training_history

def evaluate_model(model, test_loader, device, cancer_types):
    """Evaluate the model"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_targets, all_preds)
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Classification report
    report = classification_report(
        all_targets, all_preds,
        target_names=cancer_types,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    return test_acc, test_f1, report, cm, np.array(all_probs)

def create_visualizations(training_history, cm, cancer_types, output_dir="models"):
    """Create training and evaluation visualizations"""
    output_dir = Path(output_dir)
    
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(training_history['train_acc']) + 1)
    
    ax1.plot(epochs, training_history['train_acc'], 'b-', label='Training Accuracy')
    ax1.plot(epochs, training_history['val_acc'], 'r-', label='Validation Accuracy')
    ax1.set_title('Model Accuracy on Real TCGA Data')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss')
    ax2.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss')
    ax2.set_title('Model Loss on Real TCGA Data')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_tcga_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cancer_types, yticklabels=cancer_types)
    plt.title('Confusion Matrix - Real TCGA Data Results')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'real_tcga_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training pipeline for real TCGA data"""
    logger.info("🚀 Starting Real TCGA-Derived Training Pipeline...")
    
    device = 'cpu'  # Keep CPU for stability
    
    # Load the expanded real TCGA data
    logger.info("Loading expanded real TCGA data...")
    data = np.load('expanded_real_tcga_data.npz', allow_pickle=True)
    X, y = data['features'], data['labels']
    cancer_types = data['cancer_types']
    quality_metrics = data['quality_metrics'].item()
    
    logger.info(f"📊 Loaded data: {X.shape}")
    logger.info(f"🧬 Real mutations used: {quality_metrics['expanded_from_real_mutations']}")
    logger.info(f"🎯 Cancer types: {cancer_types}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        X, y, batch_size=32
    )
    
    # Model configuration
    config = {
        'embed_dim': 768,
        'n_heads': 24,
        'n_layers': 12,  # Slightly smaller for this dataset size
        'dropout': 0.15,
        'ff_hidden_dim': 2048,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'max_epochs': 200,
        'use_residual_mlp': True,
        'use_layer_scale': True,
        'use_stochastic_depth': True
    }
    
    # Create model
    model = UltraAdvancedTransformer(input_dim=X.shape[1], config=config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🤖 Model parameters: {total_params:,}")
    
    # Train model
    start_time = time.time()
    best_val_acc, best_model_state, training_history = train_model(
        model, train_loader, val_loader, config, device
    )
    training_time = time.time() - start_time
    
    logger.info(f"🎉 Training completed in {training_time/60:.2f} minutes")
    logger.info(f"🏆 Best validation accuracy on real TCGA data: {best_val_acc:.4f} ({best_val_acc:.2%})")
    
    # Evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        test_acc, test_f1, report, cm, probs = evaluate_model(
            model, test_loader, device, cancer_types
        )
        
        logger.info(f"🎯 Final Results on Real TCGA-Derived Data:")
        logger.info(f"   Test Accuracy: {test_acc:.4f} ({test_acc:.2%})")
        logger.info(f"   Test F1-Score: {test_f1:.4f}")
        logger.info(f"   Per-class results:")
        for i, cancer_type in enumerate(cancer_types):
            if str(i) in report:
                precision = report[str(i)]['precision']
                recall = report[str(i)]['recall']
                f1 = report[str(i)]['f1-score']
                logger.info(f"     {cancer_type}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'test_f1_score': test_f1,
            'best_val_accuracy': best_val_acc,
            'training_time_minutes': training_time / 60,
            'config': config,
            'model_parameters': total_params,
            'data_source': 'real_tcga_derived',
            'real_mutations_used': quality_metrics['expanded_from_real_mutations'],
            'real_samples_used': quality_metrics['real_samples_used'],
            'expansion_method': quality_metrics['expansion_method'],
            'per_class_metrics': report
        }
        
        # Save everything
        with open('models/real_tcga_transformer_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        torch.save({
            'model_state_dict': best_model_state,
            'config': config,
            'results': results,
            'training_history': training_history
        }, 'models/real_tcga_transformer.pth')
        
        joblib.dump(scaler, 'models/real_tcga_scaler.pkl')
        
        # Create visualizations
        create_visualizations(training_history, cm, cancer_types)
        
        logger.info("✅ All results saved!")
        
        if best_val_acc >= 0.85:
            logger.info("🎉🎉🎉 EXCELLENT RESULTS ON REAL TCGA DATA! 🎉🎉🎉")
        elif best_val_acc >= 0.80:
            logger.info("🚀 VERY GOOD RESULTS ON REAL TCGA DATA!")
        else:
            logger.info(f"📈 Good progress: {best_val_acc:.2%} - Real data is challenging!")
    
    return best_val_acc

if __name__ == "__main__":
    main()
