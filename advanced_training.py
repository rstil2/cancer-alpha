#!/usr/bin/env python3
"""
Advanced Training Pipeline with Hyperparameter Optimization
===========================================================

This module implements advanced training techniques including:
- Learning rate scheduling
- Early stopping
- Cross-validation
- Hyperparameter optimization
- Advanced data augmentation

Author: Dr. R. Craig Stillwell
Date: July 28, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna
from models.multimodal_transformer import MultiModalTransformer, MultiModalConfig, create_ensemble_model
from dataset_loader import load_real_cancer_data
import warnings
warnings.filterwarnings('ignore')


class AdvancedGenomicDataset(Dataset):
    """Enhanced genomic dataset with data augmentation"""
    
    def __init__(self, data: Dict[str, np.ndarray], augment: bool = False, noise_level: float = 0.1):
        self.data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items() if k != 'labels'}
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        self.augment = augment
        self.noise_level = noise_level
        
    def __len__(self) -> int:
        return self.labels.size(0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx].clone() for k, v in self.data.items()}
        
        # Data augmentation for training
        if self.augment:
            for modality in item.keys():
                # Add gaussian noise
                noise = torch.randn_like(item[modality]) * self.noise_level
                item[modality] = item[modality] + noise
                
                # Random feature dropout (masking)
                if torch.rand(1) < 0.1:
                    mask = torch.rand_like(item[modality]) > 0.05
                    item[modality] = item[modality] * mask
        
        item['labels'] = self.labels[idx]
        return item


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class AdvancedTrainer:
    """Advanced trainer with learning rate scheduling and early stopping"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 config: Dict, save_dir: str = "models"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Advanced optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config['scheduler_t0'],
            T_mult=2,
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config['patience'])
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs['logits'], labels)
            
            # Add regularization
            if self.config.get('l2_reg', 0) > 0:
                l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss += self.config['l2_reg'] * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs['logits'], dim=1)
            correct = (predicted == labels).sum().item()
            
            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)
        
        return total_loss / total_samples, total_correct / total_samples
    
    def validate_epoch(self) -> Tuple[float, float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_probs, all_labels = [], []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs['logits'], labels)
                
                _, predicted = torch.max(outputs['logits'], dim=1)
                correct = (predicted == labels).sum().item()
                
                total_loss += loss.item() * labels.size(0)
                total_correct += correct
                total_samples += labels.size(0)
                
                # For AUC calculation
                probs = torch.softmax(outputs['logits'], dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate AUC
        all_probs = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)
        
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        return total_loss / total_samples, total_correct / total_samples, auc
    
    def train(self, epochs: int) -> Dict:
        """Complete training loop"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        best_val_acc = 0
        best_model_state = None
        
        print(f"🚀 Starting advanced training for {epochs} epochs...")
        print(f"📱 Using device: {device}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_auc = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'best_val_accuracy': best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }


def objective(trial) -> float:
    """Optuna objective function for hyperparameter optimization"""
    
    # Hyperparameter suggestions
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'd_model': trial.suggest_categorical('d_model', [256, 512, 768]),
        'n_heads': trial.suggest_categorical('n_heads', [8, 12, 16]),
        'n_layers': trial.suggest_int('n_layers', 4, 12),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'scheduler_t0': trial.suggest_int('scheduler_t0', 5, 20),
        'patience': 15,
        'grad_clip': 1.0,
        'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    }
    
    # Load data
    train_data, val_data = load_real_cancer_data("tcga")
    if train_data is None:
        return 0.0
    
    # Create datasets with augmentation
    train_dataset = AdvancedGenomicDataset(train_data, augment=True, noise_level=0.05)
    val_dataset = AdvancedGenomicDataset(val_data, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Create model
    model_config = MultiModalConfig(
        d_model=config['d_model'],
        d_ff=config['d_model'] * 4,
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        num_classes=8,
        use_cross_modal_attention=True,
        use_modality_embeddings=True,
        temperature_scaling=True
    )
    
    model = MultiModalTransformer(model_config)
    
    # Train model
    trainer = AdvancedTrainer(model, train_loader, val_loader, config)
    results = trainer.train(epochs=30)  # Reduced epochs for optimization
    
    return results['best_val_accuracy']


def hyperparameter_optimization(n_trials: int = 100) -> Dict:
    """Run hyperparameter optimization using Optuna"""
    
    print(f"🔬 Starting hyperparameter optimization with {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=7200)  # 2 hour timeout
    
    print("🏆 Optimization completed!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_path = Path("models/hyperparameter_optimization.json")
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }, f, indent=2)
    
    return study.best_params


def cross_validation_evaluation(config: Dict, k_folds: int = 5) -> Dict:
    """Perform k-fold cross-validation"""
    
    print(f"📊 Starting {k_folds}-fold cross-validation...")
    
    # Load all data
    train_data, val_data = load_real_cancer_data("tcga")
    if train_data is None:
        return {}
    
    # Combine train and validation data for CV
    all_data = {}
    for key in train_data.keys():
        if key in val_data:
            all_data[key] = np.concatenate([train_data[key], val_data[key]])
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data['methylation'], all_data['labels'])):
        print(f"\n🔄 Training fold {fold + 1}/{k_folds}...")
        
        # Split data for this fold
        fold_train_data = {k: v[train_idx] for k, v in all_data.items()}
        fold_val_data = {k: v[val_idx] for k, v in all_data.items()}
        
        # Create datasets
        train_dataset = AdvancedGenomicDataset(fold_train_data, augment=True)
        val_dataset = AdvancedGenomicDataset(fold_val_data, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Create model
        model_config = MultiModalConfig(
            d_model=config['d_model'],
            d_ff=config['d_model'] * 4,
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
            num_classes=8
        )
        
        model = MultiModalTransformer(model_config)
        
        # Add training configuration
        training_config = config.copy()
        training_config.update({
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'epochs': 50,
            'warmup_epochs': 3,
            'patience': 10,
            'use_early_stopping': True,
            'min_delta': 0.001,
            'l2_reg': config.get('l2_reg', 1e-5),
            'augment_data': True,
            'scheduler_eta_min': 1e-6
        })
        
        # Train
        trainer = AdvancedTrainer(model, train_loader, val_loader, training_config)
        results = trainer.train(epochs=training_config['epochs'])
        
        fold_results.append(results['best_val_accuracy'])
        print(f"Fold {fold + 1} accuracy: {results['best_val_accuracy']:.4f}")
    
    cv_mean = np.mean(fold_results)
    cv_std = np.std(fold_results)
    
    print(f"\n📈 Cross-validation results:")
    print(f"Mean accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Individual folds: {[f'{acc:.4f}' for acc in fold_results]}")
    
    return {
        'cv_scores': fold_results,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    }


def train_final_model(best_params: Dict) -> MultiModalTransformer:
    """Train final model with best hyperparameters"""
    
    print("🎯 Training final model with optimized hyperparameters...")
    
    # Load data
    train_data, val_data = load_real_cancer_data("tcga")
    
    # Create datasets with augmentation
    train_dataset = AdvancedGenomicDataset(train_data, augment=True, noise_level=0.05)
    val_dataset = AdvancedGenomicDataset(val_data, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # Create optimized model
    model_config = MultiModalConfig(
        d_model=best_params['d_model'],
        d_ff=best_params['d_model'] * 4,
        n_heads=best_params['n_heads'],
        n_layers=best_params['n_layers'],
        dropout=best_params['dropout'],
        num_classes=8,
        use_cross_modal_attention=True,
        use_modality_embeddings=True,
        temperature_scaling=True
    )
    
    model = MultiModalTransformer(model_config)
    
    # Extended training configuration
    training_config = best_params.copy()
    training_config.update({
        'epochs': 100,
        'patience': 20
    })
    
    # Train
    trainer = AdvancedTrainer(model, train_loader, val_loader, training_config)
    results = trainer.train(epochs=training_config['epochs'])
    
    # Save final model
    model_path = Path("models/optimized_multimodal_transformer.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'training_results': results,
        'hyperparameters': best_params
    }, model_path)
    
    print(f"💾 Saved optimized model to {model_path}")
    print(f"🎯 Final validation accuracy: {results['best_val_accuracy']:.4f}")
    
    return model


if __name__ == "__main__":
    # Step 1: Hyperparameter optimization
    best_params = hyperparameter_optimization(n_trials=50)
    
    # Step 2: Cross-validation with best parameters
    cv_results = cross_validation_evaluation(best_params)
    
    # Step 3: Train final model
    final_model = train_final_model(best_params)
    
    print("\n✅ Advanced training pipeline completed!")
    print(f"Cross-validation accuracy: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
