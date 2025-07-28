#!/usr/bin/env python3
"""
Training Script for Enhanced Multi-Modal Transformer
===================================================

Author: Cancer Alpha Research Team
Date: July 28, 2025
"""

import torch
import torch.optim as optim
import numpy as np
import json
import pickle
from typing import Dict, Tuple
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import logging

from models.enhanced_multimodal_transformer import create_enhanced_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_enhanced_data(num_samples: int = 5000) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Generate enhanced synthetic genomic data"""
    np.random.seed(42)
    
    cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
    samples_per_type = num_samples // len(cancer_types)
    
    all_data = {'methylation': [], 'mutation': [], 'cna': [], 'fragmentomics': [], 'clinical': [], 'icgc': []}
    all_labels = []
    
    for i, cancer_type in enumerate(cancer_types):
        n = samples_per_type + (1 if i < (num_samples % len(cancer_types)) else 0)
        
        # Enhanced patterns for each cancer type
        methylation = np.random.beta(2, 2, (n, 20))
        if cancer_type == 'BRCA':
            methylation[:, :5] += 0.3
        elif cancer_type == 'COAD':
            methylation[:, :8] += 0.4
        
        mutation = np.random.poisson(3, (n, 25)).astype(float)
        if cancer_type == 'HNSC':
            mutation *= 1.5
        elif cancer_type == 'KIRC':
            mutation *= 0.7
        
        cna = np.random.lognormal(np.log(2), 0.5, (n, 20))
        fragmentomics = np.random.normal(0, 1, (n, 15))
        clinical = np.random.normal(0, 1, (n, 10))
        icgc = np.random.normal(0, 1, (n, 20))
        
        # Add cancer-specific noise
        if cancer_type in ['BRCA', 'LUAD']:
            fragmentomics += np.random.normal(0, 0.2, fragmentomics.shape)
        
        all_data['methylation'].append(methylation)
        all_data['mutation'].append(mutation)
        all_data['cna'].append(cna)
        all_data['fragmentomics'].append(fragmentomics)
        all_data['clinical'].append(clinical)
        all_data['icgc'].append(icgc)
        
        all_labels.extend([i] * n)
    
    # Combine all data
    final_data = {}
    for modality in all_data.keys():
        final_data[modality] = np.vstack(all_data[modality])
    
    final_labels = np.array(all_labels)
    
    # Shuffle
    indices = np.random.permutation(len(final_labels))
    for modality in final_data.keys():
        final_data[modality] = final_data[modality][indices]
    final_labels = final_labels[indices]
    
    logger.info(f"Generated {len(final_labels)} samples")
    logger.info(f"Class distribution: {np.bincount(final_labels)}")
    
    return final_data, final_labels


def train_enhanced_model():
    """Train the enhanced model"""
    logger.info("Starting enhanced model training")
    
    # Generate data
    data_dict, labels = generate_enhanced_data(8000)
    
    # Create model
    model = create_enhanced_model(num_classes=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Training on {device}")
    
    # Prepare data
    scalers = {}
    processed_data = {}
    
    for modality, data in data_dict.items():
        scaler = StandardScaler()
        processed_data[modality] = scaler.fit_transform(data)
        scalers[modality] = scaler
    
    # Split data
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=labels[train_idx], random_state=42)
    
    # Convert to tensors
    def create_tensors(indices):
        data_tensors = {}
        for modality, data in processed_data.items():
            data_tensors[modality] = torch.FloatTensor(data[indices])
        return data_tensors, torch.LongTensor(labels[indices])
    
    train_data, train_labels = create_tensors(train_idx)
    val_data, val_labels = create_tensors(val_idx)
    test_data, test_labels = create_tensors(test_idx)
    
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    # Training loop
    num_epochs = 50
    batch_size = 32
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training
        num_samples = len(train_labels)
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_data = {}
            for modality, data in train_data.items():
                batch_data[modality] = data[batch_indices].to(device)
            batch_labels = train_labels[batch_indices].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data, batch_labels, training=True)
            
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs['predictions'] == batch_labels).sum().item()
            total += len(batch_labels)
        
        train_acc = correct / total
        avg_loss = total_loss / (num_samples // batch_size + 1)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            num_val_samples = len(val_labels)
            for i in range(0, num_val_samples, batch_size):
                end_idx = min(i + batch_size, num_val_samples)
                
                batch_data = {}
                for modality, data in val_data.items():
                    batch_data[modality] = data[i:end_idx].to(device)
                batch_labels = val_labels[i:end_idx].to(device)
                
                outputs = model(batch_data, batch_labels, training=False)
                val_loss += outputs['loss'].item()
                val_correct += (outputs['predictions'] == batch_labels).sum().item()
                val_total += len(batch_labels)
        
        val_acc = val_correct / val_total
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }, 'models/enhanced_multimodal_transformer_best.pth')
            logger.info(f"New best model saved with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping")
                break
    
    # Test evaluation
    checkpoint = torch.load('models/enhanced_multimodal_transformer_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_predictions = []
    test_true = []
    
    with torch.no_grad():
        num_test_samples = len(test_labels)
        for i in range(0, num_test_samples, batch_size):
            end_idx = min(i + batch_size, num_test_samples)
            
            batch_data = {}
            for modality, data in test_data.items():
                batch_data[modality] = data[i:end_idx].to(device)
            
            outputs = model(batch_data, training=False)
            test_predictions.extend(outputs['predictions'].cpu().numpy())
            test_true.extend(test_labels[i:end_idx].numpy())
    
    test_acc = accuracy_score(test_true, test_predictions)
    test_f1 = f1_score(test_true, test_predictions, average='weighted')
    
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test F1-Score: {test_f1:.4f}")
    
    # Save scalers and results
    with open('models/enhanced_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    
    results = {
        'test_accuracy': test_acc,
        'test_f1_score': test_f1,
        'best_val_accuracy': best_val_acc
    }
    
    with open('models/enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed successfully!")
    return results


if __name__ == "__main__":
    results = train_enhanced_model()
    print(f"Final Results: {results}")
