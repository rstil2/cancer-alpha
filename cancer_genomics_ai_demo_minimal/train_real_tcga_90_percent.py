#!/usr/bin/env python3
"""
Real TCGA Data Training for 90% Accuracy
========================================

This script trains the optimized transformer on REAL TCGA clinical data
to achieve the target of 90% accuracy on actual genomics data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import logging
from pathlib import Path
import time
import joblib
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataOptimizedTransformer(nn.Module):
    """Transformer optimized specifically for real TCGA data patterns"""
    
    def __init__(self, input_dim=110, num_classes=8, config=None):
        super().__init__()
        
        # Configuration optimized for real data challenges
        if config is None:
            config = {
                'embed_dim': 512,  # Increased capacity for real data complexity
                'n_heads': 16,     # More attention heads for pattern capture
                'n_layers': 10,    # Deeper network for real data patterns
                'dropout': 0.2,    # Higher dropout to prevent overfitting on limited real data
                'ff_hidden_dim': 1024
            }
        
        self.config = config
        self.embed_dim = config['embed_dim']
        self.n_heads = config['n_heads'] 
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        
        # Feature group dimensions for real TCGA data
        self.feature_groups = {
            'methylation': 20,
            'mutation': 25,
            'cn_alteration': 20,
            'fragmentomics': 15,
            'clinical': 10,
            'icgc_argo': 20
        }
        
        # Modality-specific encoders with batch normalization for stability
        self.modality_encoders = nn.ModuleDict()
        
        for modality, dim in self.feature_groups.items():
            self.modality_encoders[modality] = nn.Sequential(
                nn.Linear(dim, self.embed_dim // 4),
                nn.BatchNorm1d(self.embed_dim // 4),
                nn.GELU(),
                nn.Dropout(self.dropout / 3),
                nn.Linear(self.embed_dim // 4, self.embed_dim // 2),
                nn.BatchNorm1d(self.embed_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout / 2),
                nn.Linear(self.embed_dim // 2, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout / 2)
            )
        
        # Advanced transformer with layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_heads,
            dim_feedforward=config['ff_hidden_dim'],
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-layer normalization for better training
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers
        )
        
        # Multi-scale attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Regularized classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.BatchNorm1d(self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.BatchNorm1d(self.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 4, self.embed_dim // 8),
            nn.BatchNorm1d(self.embed_dim // 8),
            nn.GELU(),
            nn.Dropout(self.dropout / 2),
            nn.Linear(self.embed_dim // 8, num_classes)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization with proper scaling for deep networks"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)  # GELU gain ~1.0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
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
        pooled, attention_weights = self.attention_pool(
            query=attended_features.mean(dim=1, keepdim=True),
            key=attended_features,
            value=attended_features
        )
        
        # Final classification
        pooled = pooled.squeeze(1)
        logits = self.classifier(pooled)
        
        return logits

def load_real_tcga_data():
    """Load and analyze real TCGA data"""
    
    logger.info("Loading real TCGA clinical data...")
    data = np.load('tcga_processed_data.npz')
    
    X = data['features']
    y = data['labels']
    cancer_types = data['cancer_types']
    feature_names = data['feature_names']
    
    logger.info(f"Real TCGA data shape: {X.shape}")
    logger.info(f"Number of cancer types: {len(cancer_types)}")
    logger.info(f"Cancer types: {cancer_types}")
    logger.info(f"Class distribution: {Counter(y)}")
    
    return X, y, cancer_types, feature_names

def create_robust_data_loaders(X, y, batch_size=64, test_size=0.2, val_size=0.2):
    """Create data loaders with robust preprocessing for real data"""
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Second split: train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
    )
    
    # Use RobustScaler for real data (handles outliers better)
    scaler = RobustScaler()
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
    
    # Smaller batch sizes for limited real data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Real data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return train_loader, val_loader, test_loader, scaler

def train_real_data_model(model, train_loader, val_loader, config, device='cpu'):
    """Advanced training specifically tuned for real TCGA data"""
    
    model = model.to(device)
    
    # Advanced optimizer configuration for real data
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Advanced learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['max_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Focal loss for handling class imbalance in real data
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.label_smoothing = label_smoothing
            
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()
    
    criterion = FocalLoss(
        gamma=config.get('focal_gamma', 2.0),
        label_smoothing=config.get('label_smoothing', 0.1)
    )
    
    # Training loop with advanced techniques
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = config.get('patience', 25)  # More patience for real data
    
    training_history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
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
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()  # Step per batch for OneCycleLR
            
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
        
        # Store history
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['train_loss'].append(train_loss / len(train_loader))
        training_history['val_loss'].append(val_loss / len(val_loader))
        
        # Log progress more frequently for real data monitoring
        if epoch % 10 == 0 or val_acc > best_val_acc:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch+1}/{config['max_epochs']} - "
                      f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save progressive checkpoints
            if val_acc > 0.80:  # Save when getting close to target
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': config,
                    'val_accuracy': val_acc,
                    'epoch': epoch,
                    'training_history': training_history
                }, f'models/real_tcga_checkpoint_acc_{val_acc:.4f}.pth')
                logger.info(f"🎯 Checkpoint saved at {val_acc:.1%} accuracy!")
        else:
            patience_counter += 1
        
        # Early stopping with more patience for real data
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Target reached check
        if val_acc >= 0.90:
            logger.info(f"🎉 TARGET REACHED! Achieved {val_acc:.1%} validation accuracy!")
            break
    
    return best_val_acc, best_model_state, training_history

def evaluate_real_data_model(model, test_loader, best_model_state, config, device='cpu'):
    """Comprehensive evaluation on real TCGA test data"""
    
    if best_model_state is None:
        logger.error("No trained model found!")
        return None
    
    # Load best model
    model.load_state_dict(best_model_state)
    model = model.to(device)
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
    
    # Calculate comprehensive metrics
    test_acc = accuracy_score(all_targets, all_preds)
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    
    logger.info(f"🏥 Real TCGA Test Results:")
    logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc:.1%})")
    logger.info(f"Test F1-Score: {test_f1:.4f}")
    
    # Detailed classification report
    logger.info("Detailed Classification Report:")
    logger.info(f"\n{classification_report(all_targets, all_preds)}")
    
    return {
        'test_accuracy': test_acc,
        'test_f1_score': test_f1,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

def main():
    """Main training pipeline for 90% accuracy on real TCGA data"""
    
    logger.info("🏥 Starting REAL TCGA Data Training for 90% Accuracy Target...")
    
    device = 'cpu'  # Use CPU for stability with limited real data
    
    # Load real TCGA data
    X, y, cancer_types, feature_names = load_real_tcga_data()
    
    # Create data loaders optimized for real data
    train_loader, val_loader, test_loader, scaler = create_robust_data_loaders(
        X, y, batch_size=32  # Smaller batch size for limited real data
    )
    
    # Configuration optimized for real TCGA data challenges
    config = {
        'embed_dim': 512,
        'n_heads': 16,
        'n_layers': 10,
        'dropout': 0.2,
        'ff_hidden_dim': 1024,
        'learning_rate': 5e-4,  # Lower learning rate for real data stability
        'weight_decay': 1e-3,   # Higher weight decay to prevent overfitting
        'focal_gamma': 2.0,
        'label_smoothing': 0.1,
        'max_epochs': 200,      # More epochs for real data convergence
        'patience': 30          # More patience for real data
    }
    
    # Create and train model
    model = RealDataOptimizedTransformer(config=config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model on real data
    start_time = time.time()
    best_val_acc, best_model_state, training_history = train_real_data_model(
        model, train_loader, val_loader, config, device
    )
    training_time = time.time() - start_time
    
    logger.info(f"Real data training completed in {training_time/60:.2f} minutes")
    logger.info(f"Best validation accuracy on REAL data: {best_val_acc:.4f} ({best_val_acc:.1%})")
    
    # Evaluate on real test set
    results = evaluate_real_data_model(model, test_loader, best_model_state, config, device)
    
    # Save comprehensive results
    final_results = {
        **results,
        'best_val_accuracy': best_val_acc,
        'training_time_minutes': training_time / 60,
        'config': config,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_history': training_history,
        'data_info': {
            'total_samples': len(X),
            'features': X.shape[1],
            'cancer_types': cancer_types.tolist(),
            'data_source': 'Real TCGA Clinical Data'
        }
    }
    
    # Remove numpy arrays for JSON serialization
    if 'predictions' in final_results:
        del final_results['predictions']
    if 'targets' in final_results:
        del final_results['targets']
    if 'probabilities' in final_results:
        del final_results['probabilities']
    
    with open('models/real_tcga_90_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save final model
    if best_model_state is not None:
        torch.save({
            'model_state_dict': best_model_state,
            'config': config,
            'results': final_results
        }, 'models/real_tcga_90_transformer.pth')
    
    # Save scaler
    joblib.dump(scaler, 'models/real_tcga_90_scaler.pkl')
    
    logger.info("🏥 Real TCGA training pipeline completed!")
    
    if best_val_acc >= 0.90:
        logger.info("🎉 SUCCESS: Achieved 90%+ validation accuracy on REAL TCGA data!")
    else:
        logger.info(f"🎯 Progress: {best_val_acc:.1%} accuracy achieved. Target: 90%")
        if best_val_acc >= 0.85:
            logger.info("💪 Very close to target! Consider additional training or hyperparameter tuning.")
        elif best_val_acc >= 0.80:
            logger.info("📈 Good progress! On track toward 90% target.")
        else:
            logger.info("🔄 Need further optimization. Consider data augmentation or architecture changes.")
    
    return final_results

if __name__ == "__main__":
    main()
