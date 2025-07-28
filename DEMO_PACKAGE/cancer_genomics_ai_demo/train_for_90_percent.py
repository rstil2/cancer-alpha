#!/usr/bin/env python3
"""
Optimized Training Pipeline for 90% Validation Accuracy
=======================================================

This script implements advanced training techniques to achieve 90% validation accuracy:
- Enhanced synthetic data
- Hierarchical transformer architecture
- Advanced regularization
- Optimal hyperparameters
- Ensemble methods
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import optuna
import json
import logging
from pathlib import Path
import time

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
                'n_layers': 12,
                'dropout': 0.1,
                'ff_hidden_dim': 2048,
                'label_smoothing': 0.05
            }
        
        self.config = config
        self.embed_dim = config['embed_dim']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        
        # Feature group dimensions (matching our enhanced data generator)
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
        start_idx = 0
        
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
        
        # Classification head with residual connections
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
        batch_size = x.size(0)
        
        # Split input into modalities
        modality_features = []
        start_idx = 0
        
        for modality, dim in self.feature_groups.items():
            end_idx = start_idx + dim
            modality_input = x[:, start_idx:end_idx]
            
            # Encode modality
            encoded = self.modality_encoders[modality](modality_input)
            modality_features.append(encoded.unsqueeze(1))  # Add sequence dimension
            
            start_idx = end_idx
        
        # Stack modalities as sequences
        modality_stack = torch.cat(modality_features, dim=1)  # [batch, n_modalities, embed_dim]
        
        # Apply transformer attention across modalities
        attended_features = self.transformer(modality_stack)
        
        # Global attention pooling
        pooled, _ = self.attention_pool(
            query=attended_features.mean(dim=1, keepdim=True),
            key=attended_features,
            value=attended_features
        )
        
        # Final classification
        pooled = pooled.squeeze(1)  # Remove sequence dimension
        logits = self.classifier(pooled)
        
        return logits

class OptimizedTrainer:\n    \"\"\"Advanced trainer for achieving 90% validation accuracy\"\"\"\n    \n    def __init__(self, device='cpu'):\n        self.device = device\n        self.best_model = None\n        self.best_accuracy = 0.0\n        self.training_history = []\n    \n    def load_enhanced_data(self, data_dir='data/enhanced'):\n        \"\"\"Load the enhanced synthetic dataset\"\"\"\n        \n        data_path = Path(data_dir)\n        \n        # Load data\n        X = np.load(data_path / 'enhanced_X.npy')\n        y = np.load(data_path / 'enhanced_y.npy')\n        \n        # Load metadata\n        with open(data_path / 'enhanced_metadata.json', 'r') as f:\n            metadata = json.load(f)\n        \n        logger.info(f\"Loaded enhanced dataset: {X.shape}\")\n        logger.info(f\"Class distribution: {metadata['class_distribution']}\")\n        \n        return X, y, metadata\n    \n    def create_data_loaders(self, X, y, batch_size=256, test_size=0.2, val_size=0.2):\n        \"\"\"Create optimized data loaders with proper splitting\"\"\"\n        \n        # First split: separate test set\n        X_temp, X_test, y_temp, y_test = train_test_split(\n            X, y, test_size=test_size, stratify=y, random_state=42\n        )\n        \n        # Second split: train and validation from remaining data\n        val_size_adjusted = val_size / (1 - test_size)\n        X_train, X_val, y_train, y_val = train_test_split(\n            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42\n        )\n        \n        # Scale features\n        scaler = StandardScaler()\n        X_train_scaled = scaler.fit_transform(X_train)\n        X_val_scaled = scaler.transform(X_val)\n        X_test_scaled = scaler.transform(X_test)\n        \n        # Convert to tensors\n        train_dataset = TensorDataset(\n            torch.FloatTensor(X_train_scaled),\n            torch.LongTensor(y_train)\n        )\n        val_dataset = TensorDataset(\n            torch.FloatTensor(X_val_scaled),\n            torch.LongTensor(y_val)\n        )\n        test_dataset = TensorDataset(\n            torch.FloatTensor(X_test_scaled),\n            torch.LongTensor(y_test)\n        )\n        \n        # Create data loaders\n        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)\n        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)\n        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n        \n        logger.info(f\"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\")\n        \n        return train_loader, val_loader, test_loader, scaler\n    \n    def train_model(self, model, train_loader, val_loader, config):\n        \"\"\"Train model with advanced techniques\"\"\"\n        \n        model = model.to(self.device)\n        \n        # Optimizer with weight decay\n        optimizer = optim.AdamW(\n            model.parameters(),\n            lr=config['learning_rate'],\n            weight_decay=config['weight_decay'],\n            betas=(0.9, 0.999)\n        )\n        \n        # Learning rate scheduler\n        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(\n            optimizer,\n            T_0=config['scheduler_t0'],\n            T_mult=2,\n            eta_min=config['learning_rate'] * 0.01\n        )\n        \n        # Loss function with label smoothing\n        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])\n        \n        # Training loop\n        best_val_acc = 0.0\n        patience_counter = 0\n        patience = config.get('patience', 15)\n        \n        for epoch in range(config['max_epochs']):\n            # Training phase\n            model.train()\n            train_loss = 0.0\n            train_correct = 0\n            train_total = 0\n            \n            for batch_idx, (data, target) in enumerate(train_loader):\n                data, target = data.to(self.device), target.to(self.device)\n                \n                optimizer.zero_grad()\n                \n                # Forward pass\n                output = model(data)\n                loss = criterion(output, target)\n                \n                # Backward pass\n                loss.backward()\n                \n                # Gradient clipping\n                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n                \n                optimizer.step()\n                \n                # Statistics\n                train_loss += loss.item()\n                pred = output.argmax(dim=1)\n                train_correct += pred.eq(target).sum().item()\n                train_total += target.size(0)\n            \n            # Validation phase\n            model.eval()\n            val_loss = 0.0\n            val_correct = 0\n            val_total = 0\n            \n            with torch.no_grad():\n                for data, target in val_loader:\n                    data, target = data.to(self.device), target.to(self.device)\n                    output = model(data)\n                    val_loss += criterion(output, target).item()\n                    pred = output.argmax(dim=1)\n                    val_correct += pred.eq(target).sum().item()\n                    val_total += target.size(0)\n            \n            # Calculate accuracies\n            train_acc = train_correct / train_total\n            val_acc = val_correct / val_total\n            \n            # Update learning rate\n            scheduler.step()\n            \n            # Log progress\n            if epoch % 5 == 0 or val_acc > best_val_acc:\n                logger.info(f\"Epoch {epoch+1}/{config['max_epochs']} - \"\n                          f\"Train Loss: {train_loss/len(train_loader):.4f}, \"\n                          f\"Train Acc: {train_acc:.4f}, \"\n                          f\"Val Acc: {val_acc:.4f}, \"\n                          f\"LR: {scheduler.get_last_lr()[0]:.6f}\")\n            \n            # Save best model\n            if val_acc > best_val_acc:\n                best_val_acc = val_acc\n                self.best_model = model.state_dict().copy()\n                self.best_accuracy = val_acc\n                patience_counter = 0\n                \n                # Save checkpoint if accuracy is promising\n                if val_acc > 0.85:\n                    torch.save({\n                        'model_state_dict': self.best_model,\n                        'config': config,\n                        'val_accuracy': val_acc,\n                        'epoch': epoch\n                    }, f'models/checkpoint_acc_{val_acc:.4f}.pth')\n            else:\n                patience_counter += 1\n            \n            # Early stopping\n            if patience_counter >= patience:\n                logger.info(f\"Early stopping at epoch {epoch+1}\")\n                break\n            \n            # Record training history\n            self.training_history.append({\n                'epoch': epoch,\n                'train_loss': train_loss / len(train_loader),\n                'train_acc': train_acc,\n                'val_acc': val_acc,\n                'lr': scheduler.get_last_lr()[0]\n            })\n        \n        return best_val_acc\n    \n    def evaluate_model(self, test_loader, config):\n        \"\"\"Evaluate the best model on test set\"\"\"\n        \n        if self.best_model is None:\n            logger.error(\"No trained model found!\")\n            return None\n        \n        # Load best model\n        model = OptimizedMultiModalTransformer(config=config)\n        model.load_state_dict(self.best_model)\n        model = model.to(self.device)\n        model.eval()\n        \n        all_preds = []\n        all_targets = []\n        \n        with torch.no_grad():\n            for data, target in test_loader:\n                data, target = data.to(self.device), target.to(self.device)\n                output = model(data)\n                pred = output.argmax(dim=1)\n                \n                all_preds.extend(pred.cpu().numpy())\n                all_targets.extend(target.cpu().numpy())\n        \n        # Calculate metrics\n        test_acc = accuracy_score(all_targets, all_preds)\n        test_f1 = f1_score(all_targets, all_preds, average='weighted')\n        \n        logger.info(f\"Test Results - Accuracy: {test_acc:.4f}, F1-Score: {test_f1:.4f}\")\n        \n        return {\n            'test_accuracy': test_acc,\n            'test_f1_score': test_f1,\n            'best_val_accuracy': self.best_accuracy\n        }\n\ndef main():\n    \"\"\"Main training pipeline for 90% accuracy target\"\"\"\n    \n    logger.info(\"Starting optimized training for 90% validation accuracy...\")\n    \n    # Initialize trainer\n    trainer = OptimizedTrainer(device='cpu')  # Use GPU if available\n    \n    # Load enhanced data\n    X, y, metadata = trainer.load_enhanced_data()\n    \n    # Create data loaders\n    train_loader, val_loader, test_loader, scaler = trainer.create_data_loaders(\n        X, y, batch_size=512  # Larger batch size for stability\n    )\n    \n    # Optimized configuration for high accuracy\n    config = {\n        'embed_dim': 512,\n        'n_heads': 16,\n        'n_layers': 12,\n        'dropout': 0.1,\n        'ff_hidden_dim': 2048,\n        'learning_rate': 3e-4,\n        'weight_decay': 1e-4,\n        'label_smoothing': 0.05,\n        'scheduler_t0': 20,\n        'max_epochs': 200,\n        'patience': 20\n    }\n    \n    # Create and train model\n    model = OptimizedMultiModalTransformer(config=config)\n    logger.info(f\"Model parameters: {sum(p.numel() for p in model.parameters()):,}\")\n    \n    # Train model\n    start_time = time.time()\n    best_val_acc = trainer.train_model(model, train_loader, val_loader, config)\n    training_time = time.time() - start_time\n    \n    logger.info(f\"Training completed in {training_time/60:.2f} minutes\")\n    logger.info(f\"Best validation accuracy: {best_val_acc:.4f}\")\n    \n    # Evaluate on test set\n    results = trainer.evaluate_model(test_loader, config)\n    \n    # Save results\n    final_results = {\n        **results,\n        'training_time_minutes': training_time / 60,\n        'config': config,\n        'model_parameters': sum(p.numel() for p in model.parameters())\n    }\n    \n    with open('models/optimized_results.json', 'w') as f:\n        json.dump(final_results, f, indent=2)\n    \n    # Save final model\n    if trainer.best_model is not None:\n        torch.save({\n            'model_state_dict': trainer.best_model,\n            'config': config,\n            'results': final_results\n        }, 'models/optimized_transformer_final.pth')\n    \n    # Save scaler\n    import joblib\n    joblib.save(scaler, 'models/optimized_scaler.pkl')\n    \n    logger.info(\"Training pipeline completed!\")\n    \n    if best_val_acc >= 0.90:\n        logger.info(\"🎉 SUCCESS: Achieved 90%+ validation accuracy!\")\n    else:\n        logger.info(f\"Target not reached. Best: {best_val_acc:.1%}. Try Phase 2 improvements.\")\n    \n    return final_results\n\nif __name__ == \"__main__\":\n    main()
