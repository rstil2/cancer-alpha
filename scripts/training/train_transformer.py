#!/usr/bin/env python3
"""
Train Multi-Modal Transformer on Cancer Genomics Data
=====================================================

This script trains the multi-modal transformer architecture on real or synthetic genomic datasets,
including cross-validation and evaluation on multiple cancer types.

Author: Dr. R. Craig Stillwell
Date: July 28, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, List
from models.multimodal_transformer import MultiModalTransformer, MultiModalConfig
from dataset_loader import load_real_cancer_data
import joblib
from pathlib import Path


class GenomicDataset(Dataset):
    """Cancer genomics dataset representation for PyTorch"""
    
    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items() if k != 'labels'}
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        
    def __len__(self) -> int:
        return self.labels.size(0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.data.items()}
        item['labels'] = self.labels[idx]
        return item


class Trainer:
    """Training loop for multi-modal transformer"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 20):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        
        # Configure optimizer, loss
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-4)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        """Run the training loop for the configured number of epochs"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss, total_correct = 0, 0
            
            for batch in self.train_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs['logits'], labels)
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['logits'], dim=1)
                correct = (predicted == labels).sum().item()
                
                # Track statistics
                total_loss += loss.item() * labels.size(0)
                total_correct += correct
            
            epoch_loss = total_loss / len(self.train_loader.dataset)
            epoch_accuracy = total_correct / len(self.train_loader.dataset)
            
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
            
            # Validate after each epoch
            self.validate()
    
    def validate(self):
        """Evaluate the model on the validation set"""
        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        total_loss, total_correct = 0, 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                
                outputs = self.model(inputs)
                
                # Calculate loss and accuracy
                loss = self.criterion(outputs['logits'], labels)
                _, predicted = torch.max(outputs['logits'], dim=1)
                correct = (predicted == labels).sum().item()
                
                total_loss += loss.item() * labels.size(0)
                total_correct += correct
        
        val_loss = total_loss / len(self.val_loader.dataset)
        val_accuracy = total_correct / len(self.val_loader.dataset)
        
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


def train_transformer():
    """Train the multi-modal transformer on real cancer genomics data"""
    print("🚀 Starting Multi-Modal Transformer Training...")
    
    # Load real TCGA-like data
    print("📥 Loading cancer genomics data...")
    train_data, val_data = load_real_cancer_data("tcga")
    
    if train_data is None or val_data is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create PyTorch datasets
    train_dataset = GenomicDataset(train_data)
    val_dataset = GenomicDataset(val_data)
    
    # Create data loaders
    batch_size = 32  # Increased batch size for better training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")
    
    # Initialize production-ready model
    config = MultiModalConfig(
        d_model=512,
        d_ff=1024,
        n_heads=8,
        n_layers=8,
        dropout=0.15,
        num_classes=8,
        use_cross_modal_attention=True,
        use_modality_embeddings=True,
        temperature_scaling=True
    )
    
    model = MultiModalTransformer(config)
    print(f"🧠 Initialized Multi-Modal Transformer with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train the model
    trainer = Trainer(model, train_loader, val_loader, epochs=50)
    trainer.train()
    
    # Save the trained model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "multimodal_transformer.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model': model
    }, model_path)
    
    print(f"💾 Saved trained model to {model_path}")
    
    # Evaluate final performance
    evaluate_model(model, val_loader)
    
    return model


def evaluate_model(model: nn.Module, val_loader: DataLoader):
    """Evaluate the trained model and report detailed metrics"""
    print("\n📊 Final Model Evaluation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            
            # Get predictions and confidence scores
            probabilities = outputs['probabilities']
            _, predicted = torch.max(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_confidence = np.mean(all_confidences)
    
    cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
    
    print(f"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🔥 Average Confidence: {avg_confidence:.4f}")
    print("\n📋 Detailed Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=cancer_types))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\n🔍 Confusion Matrix:")
    print("    ", end="")
    for ct in cancer_types:
        print(f"{ct:>6}", end="")
    print()
    
    for i, ct in enumerate(cancer_types):
        print(f"{ct:>4}", end="")
        for j in range(len(cancer_types)):
            print(f"{cm[i,j]:>6}", end="")
        print()
    
    return accuracy, avg_confidence


if __name__ == '__main__':
    train_transformer()

