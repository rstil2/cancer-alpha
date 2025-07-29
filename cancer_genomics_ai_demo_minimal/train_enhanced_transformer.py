#!/usr/bin/env python3
"""
Advanced Training Script for Enhanced Multi-Modal Transformer
=============================================================

Trains the enhanced model with improved techniques and validation.

Author: Cancer Alpha Research Team
Date: July 28, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

# Import our enhanced model
from models.enhanced_multimodal_transformer import create_enhanced_model, EnhancedMultiModalConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedGenomicDataGenerator:
    """Enhanced synthetic genomic data generator with improved biological realism"""
    
    def __init__(self, num_samples: int = 5000, random_state: int = 42):
        self.num_samples = num_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Enhanced cancer type mappings with more detailed patterns
        self.cancer_types = {
            'BRCA': 0, 'LUAD': 1, 'COAD': 2, 'PRAD': 3,
            'STAD': 4, 'KIRC': 5, 'HNSC': 6, 'LIHC': 7
        }
        
        # Feature dimensions
        self.feature_dims = {
            'methylation': 20,
            'mutation': 25,
            'cna': 20, 
            'fragmentomics': 15,
            'clinical': 10,
            'icgc': 20
        }
    
    def generate_enhanced_dataset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Generate enhanced synthetic dataset with improved biological patterns"""
        
        samples_per_type = self.num_samples // len(self.cancer_types)
        all_data = {key: [] for key in self.feature_dims.keys()}
        all_labels = []
        
        for cancer_type, label in self.cancer_types.items():
            n_samples = samples_per_type + (1 if label < (self.num_samples % len(self.cancer_types)) else 0)
            
            # Generate cancer-type-specific patterns
            type_data = self._generate_cancer_specific_enhanced_features(cancer_type, n_samples)
            
            for modality, data in type_data.items():
                all_data[modality].append(data)
            
            all_labels.extend([label] * n_samples)
        
        # Concatenate all data
        final_data = {}
        for modality in self.feature_dims.keys():
            final_data[modality] = np.vstack(all_data[modality])
        
        final_labels = np.array(all_labels)
        
        # Shuffle data
        indices = np.random.permutation(len(final_labels))
        for modality in final_data.keys():
            final_data[modality] = final_data[modality][indices]
        final_labels = final_labels[indices]
        
        logger.info(f"Generated {len(final_labels)} samples across {len(self.cancer_types)} cancer types")
        logger.info(f"Class distribution: {np.bincount(final_labels)}")
        
        return final_data, final_labels
    
    def _generate_cancer_specific_enhanced_features(self, cancer_type: str, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate enhanced cancer-type-specific features with realistic biological patterns"""
        
        # Base features with appropriate noise levels
        data = {}
        
        for modality, dim in self.feature_dims.items():
            if modality == 'methylation':
                # Methylation patterns (0-1 range with beta distribution)
                base = np.random.beta(2, 2, (n_samples, dim))
                data[modality] = self._apply_cancer_methylation_patterns(base, cancer_type)
                
            elif modality == 'mutation':
                # Mutation counts (Poisson distribution)
                base = np.random.poisson(3, (n_samples, dim)).astype(float)
                data[modality] = self._apply_cancer_mutation_patterns(base, cancer_type)
                
            elif modality == 'cna':
                # Copy number alterations (centered around 2, log-normal)
                base = np.random.lognormal(np.log(2), 0.5, (n_samples, dim))
                data[modality] = self._apply_cancer_cna_patterns(base, cancer_type)
                
            elif modality == 'fragmentomics':
                # Fragment characteristics (normal distribution)
                base = np.random.normal(0, 1, (n_samples, dim))
                data[modality] = self._apply_cancer_fragmentomics_patterns(base, cancer_type)
                
            elif modality == 'clinical':
                # Clinical features (mixed distributions)
                base = np.random.normal(0, 1, (n_samples, dim))
                data[modality] = self._apply_cancer_clinical_patterns(base, cancer_type)
                
            elif modality == 'icgc':
                # ICGC features (normalized)
                base = np.random.normal(0, 1, (n_samples, dim))
                data[modality] = self._apply_cancer_icgc_patterns(base, cancer_type)
        
        return data
    
    def _apply_cancer_methylation_patterns(self, base: np.ndarray, cancer_type: str) -> np.ndarray:
        """Apply cancer-specific methylation patterns"""
        if cancer_type == 'BRCA':
            # BRCA-specific hypermethylation in tumor suppressors
            base[:, :5] += 0.3  # Promoter hypermethylation
            base[:, 10:15] -= 0.2  # Global hypomethylation
        elif cancer_type == 'COAD':
            # CpG island methylator phenotype (CIMP)
            base[:, :8] += 0.4
            base[:, 15:] -= 0.1
        elif cancer_type == 'LUAD':
            # Smoking-related methylation changes
            base[:, 5:10] += 0.25
            base[:, -5:] += 0.15
        # Add more patterns for other cancer types...
        
        return np.clip(base, 0, 1)  # Keep in valid range
    
    def _apply_cancer_mutation_patterns(self, base: np.ndarray, cancer_type: str) -> np.ndarray:
        """Apply cancer-specific mutation patterns"""
        if cancer_type == 'HNSC':
            # High mutation burden
            base *= 2.0
        elif cancer_type == 'KIRC':
            # Lower mutation burden
            base *= 0.5
        elif cancer_type == 'LUAD':
            # Smoking signature
            base[:, :10] *= 1.5
        
        return base
    
    def _apply_cancer_cna_patterns(self, base: np.ndarray, cancer_type: str) -> np.ndarray:
        """Apply cancer-specific copy number patterns"""
        if cancer_type == 'BRCA':
            # HER2 amplification patterns
            base[:, 0] *= np.random.choice([1, 3, 5], size=base.shape[0], p=[0.7, 0.2, 0.1])
        elif cancer_type == 'COAD':
            # Chromosomal instability
            noise = np.random.normal(0, 0.3, base.shape)
            base += noise
        
        return np.clip(base, 0, 10)  # Reasonable CNA range
    
    def _apply_cancer_fragmentomics_patterns(self, base: np.ndarray, cancer_type: str) -> np.ndarray:
        """Apply cancer-specific fragmentomics patterns"""
        # Different cancer types have different fragmentation patterns
        if cancer_type == 'LIHC':
            base[:, :7] += 0.5  # Liver-specific patterns
        elif cancer_type == 'PRAD':
            base[:, 5:12] -= 0.3  # Prostate-specific patterns
        
        return base
    
    def _apply_cancer_clinical_patterns(self, base: np.ndarray, cancer_type: str) -> np.ndarray:
        """Apply cancer-specific clinical patterns"""
        # Age, stage, grade patterns
        if cancer_type == 'PRAD':
            base[:, 0] += 1.0  # Older age
        elif cancer_type == 'BRCA':
            base[:, 1] += 0.5  # Different staging patterns
        
        return base
    
    def _apply_cancer_icgc_patterns(self, base: np.ndarray, cancer_type: str) -> np.ndarray:
        """Apply cancer-specific ICGC patterns"""
        # International genomic patterns
        return base + np.random.normal(0, 0.1, base.shape)


class EnhancedTrainer:
    """Enhanced trainer with advanced techniques"""
    
    def __init__(self, model, config: EnhancedMultiModalConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Advanced optimizer with scheduling
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, eta_min=1e-6
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def prepare_data(self, data_dict: Dict[str, np.ndarray], labels: np.ndarray, 
                    test_size: float = 0.2, val_size: float = 0.2):
        """Prepare data with advanced preprocessing"""
        
        # Normalize each modality
        self.scalers = {}
        processed_data = {}
        
        for modality, data in data_dict.items():
            scaler = StandardScaler()
            processed_data[modality] = scaler.fit_transform(data)
            self.scalers[modality] = scaler
        
        # Split data
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(indices, test_size=test_size + val_size, 
                                              stratify=labels, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size/(test_size + val_size),
                                           stratify=labels[temp_idx], random_state=42)
        
        # Create datasets
        def create_dataset(indices):
            dataset_dict = {}\n            for modality, data in processed_data.items():\n                dataset_dict[modality] = torch.FloatTensor(data[indices])\n            return dataset_dict, torch.LongTensor(labels[indices])\n        \n        self.train_data, self.train_labels = create_dataset(train_idx)\n        self.val_data, self.val_labels = create_dataset(val_idx)\n        self.test_data, self.test_labels = create_dataset(test_idx)\n        \n        logger.info(f\"Data splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}\")\n        \n        # Save scalers\n        with open('models/enhanced_scalers.pkl', 'wb') as f:\n            pickle.dump(self.scalers, f)\n    \n    def train_epoch(self) -> Tuple[float, float]:\n        \"\"\"Train for one epoch with advanced techniques\"\"\"\n        self.model.train()\n        total_loss = 0\n        correct_predictions = 0\n        total_samples = 0\n        \n        # Create batches manually for flexibility\n        batch_size = 32\n        num_samples = len(self.train_labels)\n        indices = torch.randperm(num_samples)\n        \n        for i in range(0, num_samples, batch_size):\n            batch_indices = indices[i:i+batch_size]\n            \n            # Prepare batch\n            batch_data = {}\n            for modality, data in self.train_data.items():\n                batch_data[modality] = data[batch_indices].to(self.device)\n            batch_labels = self.train_labels[batch_indices].to(self.device)\n            \n            # Forward pass\n            self.optimizer.zero_grad()\n            outputs = self.model(batch_data, batch_labels, training=True)\n            \n            loss = outputs['loss']\n            \n            # Backward pass\n            loss.backward()\n            \n            # Gradient clipping\n            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)\n            \n            self.optimizer.step()\n            \n            # Track metrics\n            total_loss += loss.item()\n            predictions = outputs['predictions']\n            correct_predictions += (predictions == batch_labels).sum().item()\n            total_samples += len(batch_labels)\n        \n        avg_loss = total_loss / (num_samples // batch_size + 1)\n        accuracy = correct_predictions / total_samples\n        \n        return avg_loss, accuracy\n    \n    def validate_epoch(self) -> Tuple[float, float]:\n        \"\"\"Validate the model\"\"\"\n        self.model.eval()\n        total_loss = 0\n        correct_predictions = 0\n        total_samples = 0\n        \n        with torch.no_grad():\n            batch_size = 32\n            num_samples = len(self.val_labels)\n            \n            for i in range(0, num_samples, batch_size):\n                end_idx = min(i + batch_size, num_samples)\n                \n                # Prepare batch\n                batch_data = {}\n                for modality, data in self.val_data.items():\n                    batch_data[modality] = data[i:end_idx].to(self.device)\n                batch_labels = self.val_labels[i:end_idx].to(self.device)\n                \n                # Forward pass\n                outputs = self.model(batch_data, batch_labels, training=False)\n                \n                loss = outputs['loss']\n                total_loss += loss.item()\n                \n                predictions = outputs['predictions']\n                correct_predictions += (predictions == batch_labels).sum().item()\n                total_samples += len(batch_labels)\n        \n        avg_loss = total_loss / (num_samples // batch_size + 1)\n        accuracy = correct_predictions / total_samples\n        \n        return avg_loss, accuracy\n    \n    def train(self, num_epochs: int = 50):\n        \"\"\"Main training loop\"\"\"\n        logger.info(f\"Starting training for {num_epochs} epochs on {self.device}\")\n        \n        for epoch in range(num_epochs):\n            start_time = time.time()\n            \n            # Training\n            train_loss, train_acc = self.train_epoch()\n            \n            # Validation\n            val_loss, val_acc = self.validate_epoch()\n            \n            # Learning rate scheduling\n            self.scheduler.step()\n            \n            # Track metrics\n            self.train_losses.append(train_loss)\n            self.val_losses.append(val_loss)\n            self.train_accuracies.append(train_acc)\n            self.val_accuracies.append(val_acc)\n            \n            epoch_time = time.time() - start_time\n            \n            logger.info(\n                f\"Epoch {epoch+1}/{num_epochs} - \"\n                f\"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, \"\n                f\"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, \"\n                f\"Time: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]['lr']:.2e}\"\n            )\n            \n            # Early stopping\n            if val_loss < self.best_val_loss:\n                self.best_val_loss = val_loss\n                self.patience_counter = 0\n                # Save best model\n                torch.save({\n                    'model_state_dict': self.model.state_dict(),\n                    'optimizer_state_dict': self.optimizer.state_dict(),\n                    'epoch': epoch,\n                    'val_loss': val_loss,\n                    'val_acc': val_acc\n                }, 'models/enhanced_multimodal_transformer_best.pth')\n                logger.info(f\"New best model saved with val_loss: {val_loss:.4f}\")\n            else:\n                self.patience_counter += 1\n                if self.patience_counter >= self.patience:\n                    logger.info(f\"Early stopping at epoch {epoch+1}\")\n                    break\n        \n        # Save training history\n        history = {\n            'train_losses': self.train_losses,\n            'val_losses': self.val_losses,\n            'train_accuracies': self.train_accuracies,\n            'val_accuracies': self.val_accuracies\n        }\n        \n        with open('models/enhanced_training_history.json', 'w') as f:\n            json.dump(history, f, indent=2)\n        \n        logger.info(\"Training completed successfully!\")\n    \n    def evaluate_test_set(self) -> Dict:\n        \"\"\"Evaluate on test set\"\"\"\n        # Load best model\n        checkpoint = torch.load('models/enhanced_multimodal_transformer_best.pth', \n                               map_location=self.device)\n        self.model.load_state_dict(checkpoint['model_state_dict'])\n        \n        self.model.eval()\n        all_predictions = []\n        all_true_labels = []\n        \n        with torch.no_grad():\n            batch_size = 32\n            num_samples = len(self.test_labels)\n            \n            for i in range(0, num_samples, batch_size):\n                end_idx = min(i + batch_size, num_samples)\n                \n                batch_data = {}\n                for modality, data in self.test_data.items():\n                    batch_data[modality] = data[i:end_idx].to(self.device)\n                batch_labels = self.test_labels[i:end_idx]\n                \n                outputs = self.model(batch_data, training=False)\n                predictions = outputs['predictions'].cpu().numpy()\n                \n                all_predictions.extend(predictions)\n                all_true_labels.extend(batch_labels.numpy())\n        \n        # Calculate metrics\n        accuracy = accuracy_score(all_true_labels, all_predictions)\n        \n        # Cancer type names\n        cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']\n        \n        # Classification report\n        class_report = classification_report(\n            all_true_labels, all_predictions,\n            target_names=cancer_types,\n            output_dict=True\n        )\n        \n        results = {\n            'test_accuracy': accuracy,\n            'classification_report': class_report,\n            'confusion_matrix': confusion_matrix(all_true_labels, all_predictions).tolist()\n        }\n        \n        logger.info(f\"Test Accuracy: {accuracy:.4f}\")\n        logger.info(f\"Classification Report:\\n{classification_report(all_true_labels, all_predictions, target_names=cancer_types)}\")\n        \n        # Save results\n        with open('models/enhanced_test_results.json', 'w') as f:\n            json.dump(results, f, indent=2)\n        \n        return results


def main():\n    \"\"\"Main training function\"\"\"\n    logger.info(\"Starting Enhanced Multi-Modal Transformer Training\")\n    \n    # Generate enhanced dataset\n    data_generator = AdvancedGenomicDataGenerator(num_samples=8000)\n    data_dict, labels = data_generator.generate_enhanced_dataset()\n    \n    # Create enhanced model\n    model = create_enhanced_model(num_classes=8)\n    config = EnhancedMultiModalConfig()\n    \n    logger.info(f\"Model created with {sum(p.numel() for p in model.parameters())} parameters\")\n    \n    # Initialize trainer\n    trainer = EnhancedTrainer(model, config)\n    \n    # Prepare data\n    trainer.prepare_data(data_dict, labels)\n    \n    # Train model\n    trainer.train(num_epochs=100)\n    \n    # Evaluate\n    test_results = trainer.evaluate_test_set()\n    \n    logger.info(\"Training and evaluation completed successfully!\")\n    logger.info(f\"Final test accuracy: {test_results['test_accuracy']:.4f}\")\n\n\nif __name__ == \"__main__\":\n    main()
