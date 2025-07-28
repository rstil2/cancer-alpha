#!/usr/bin/env python3
"""
Optimized Training Pipeline for 90% Validation Accuracy
=======================================================

This script implements advanced training techniques to achieve 90% validation accuracy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import json
import logging
from pathlib import Path
import time
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMultiModalTransformer(nn.Module):
    """Optimized Multi-Modal Transformer for 90% accuracy target"""
    
    def __init__(self, input_dim=110, num_classes=8, config=None):
        super().__init__()
        
        # Default configuration optimized for high accuracy
        if config is None:
            config = {
                'embed_dim': 512,
                'n_heads': 16,
                'n_layers': 8,
                'dropout': 0.1,
                'ff_hidden_dim': 1024
            }
        
        self.config = config
        self.embed_dim = config['embed_dim']
        self.n_heads = config['n_heads'] 
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        
        # Feature group dimensions
        self.feature_groups = {
            'methylation': 20,
            'mutation': 25,
            'cn_alteration': 20,
            'fragmentomics': 15,
            'clinical': 10,
            'icgc_argo': 20
        }
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        
        for modality, dim in self.feature_groups.items():
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(dim, self.embed_dim // 2),
                nn.LayerNorm(self.embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout / 2),
                nn.Linear(self.embed_dim // 2, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout / 2)
            )
        
        # Cross-modal attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_heads,
            dim_feedforward=config['ff_hidden_dim'],
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers
        )
        
        # Global attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.LayerNorm(self.embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 4, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Split input into modalities
        modality_features = []
        start_idx = 0
        
        for modality, dim in self.feature_groups.items():
            end_idx = start_idx + dim
            modality_input = x[:, start_idx:end_idx]
            
            # Encode modality
            encoded = self.modality_encoders[modality](modality_input)
            modality_features.append(encoded.unsqueeze(1))
            
            start_idx = end_idx
        
        # Stack modalities as sequences
        modality_stack = torch.cat(modality_features, dim=1)
        
        # Apply transformer attention across modalities
        attended_features = self.transformer(modality_stack)
        
        # Global attention pooling
        pooled, _ = self.attention_pool(
            query=attended_features.mean(dim=1, keepdim=True),
            key=attended_features,
            value=attended_features
        )
        
        # Final classification
        pooled = pooled.squeeze(1)
        logits = self.classifier(pooled)
        
        return logits

def load_enhanced_data(data_dir='data/enhanced'):
    """Load the enhanced synthetic dataset"""
    
    data_path = Path(data_dir)
    
    # Load data
    X = np.load(data_path / 'enhanced_X.npy')
    y = np.load(data_path / 'enhanced_y.npy')
    
    # Load metadata
    with open(data_path / 'enhanced_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded enhanced dataset: {X.shape}")
    logger.info(f"Class distribution: {metadata['class_distribution']}")
    
    return X, y, metadata

def create_data_loaders(X, y, batch_size=256, test_size=0.2, val_size=0.2):
    """Create optimized data loaders with proper splitting"""
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Second split: train and validation from remaining data
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return train_loader, val_loader, test_loader, scaler

def train_model(model, train_loader, val_loader, config, device='cpu'):
    """Train model with advanced techniques"""
    
    model = model.to(device)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler_t0'],
        T_mult=2,
        eta_min=config['learning_rate'] * 0.01
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = config.get('patience', 15)
    
    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
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
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        if epoch % 5 == 0 or val_acc > best_val_acc:
            logger.info(f"Epoch {epoch+1}/{config['max_epochs']} - "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save checkpoint if accuracy is promising
            if val_acc > 0.85:
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': config,
                    'val_accuracy': val_acc,
                    'epoch': epoch
                }, f'models/checkpoint_acc_{val_acc:.4f}.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc, best_model_state

def evaluate_model(model, test_loader, best_model_state, config, device='cpu'):
    """Evaluate the best model on test set"""
    
    if best_model_state is None:
        logger.error("No trained model found!")
        return None
    
    # Load best model
    model.load_state_dict(best_model_state)
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_targets, all_preds)
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    
    logger.info(f"Test Results - Accuracy: {test_acc:.4f}, F1-Score: {test_f1:.4f}")
    
    return {
        'test_accuracy': test_acc,
        'test_f1_score': test_f1
    }

def main():
    """Main training pipeline for 90% accuracy target"""
    
    logger.info("Starting optimized training for 90% validation accuracy...")
    
    device = 'cpu'  # Use GPU if available
    
    # Load enhanced data
    X, y, metadata = load_enhanced_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader, scaler = create_data_loaders(
        X, y, batch_size=512
    )
    
    # Optimized configuration for high accuracy
    config = {
        'embed_dim': 256,
        'n_heads': 8,
        'n_layers': 6,
        'dropout': 0.1,
        'ff_hidden_dim': 512,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'label_smoothing': 0.05,
        'scheduler_t0': 10,
        'max_epochs': 100,
        'patience': 15
    }
    
    # Create and train model
    model = OptimizedMultiModalTransformer(config=config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    start_time = time.time()
    best_val_acc, best_model_state = train_model(
        model, train_loader, val_loader, config, device
    )
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Evaluate on test set
    results = evaluate_model(model, test_loader, best_model_state, config, device)
    
    # Save results
    final_results = {
        **results,
        'best_val_accuracy': best_val_acc,
        'training_time_minutes': training_time / 60,
        'config': config,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
    
    with open('models/optimized_90_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save final model
    if best_model_state is not None:
        torch.save({
            'model_state_dict': best_model_state,
            'config': config,
            'results': final_results
        }, 'models/optimized_90_transformer.pth')
    
    # Save scaler
    joblib.save(scaler, 'models/optimized_90_scaler.pkl')
    
    logger.info("Training pipeline completed!")
    
    if best_val_acc >= 0.90:
        logger.info("🎉 SUCCESS: Achieved 90%+ validation accuracy!")
    else:
        logger.info(f"Target not reached. Best: {best_val_acc:.1%}. Need Phase 2 improvements.")
    
    return final_results

if __name__ == "__main__":
    main()
