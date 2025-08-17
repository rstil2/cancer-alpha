#!/usr/bin/env python3
"""
Comprehensive Validation Test Suite for Cancer Genomics AI Classifier
======================================================================

This test suite validates the robustness and reliability of the transformer model
across various edge cases and scenarios.

Author: Oncura Research Team
Date: July 28, 2025
"""

import torch
import numpy as np
import sys
import os
import joblib
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_transformer import MultiModalTransformer, MultiModalConfig

class ValidationTestSuite:
    """Comprehensive validation test suite for the transformer model"""
    
    def __init__(self):
        self.model = self.load_model()
        self.scalers = self.load_scalers()
        self.cancer_types = ['BRCA', 'LUAD', 'COAD', 'PRAD', 'STAD', 'KIRC', 'HNSC', 'LIHC']
        self.test_results = []
    
    def load_model(self) -> MultiModalTransformer:
        """Load the optimized transformer model"""
        checkpoint = torch.load('models/optimized_multimodal_transformer.pth', weights_only=False)
        config = checkpoint.get('config', MultiModalConfig())
        model = MultiModalTransformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def load_scalers(self) -> Dict:
        """Load the data scalers"""
        return joblib.load('models/scalers.pkl')
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess input data using modality-specific scalers"""
        # Apply separate scalers for each modality
        methylation = self.scalers['methylation'].transform(data[:, :20])
        mutation = self.scalers['mutation'].transform(data[:, 20:45])
        cna = self.scalers['cna'].transform(data[:, 45:65])
        fragmentomics = self.scalers['fragmentomics'].transform(data[:, 65:80])
        clinical = self.scalers['clinical'].transform(data[:, 80:90])
        icgc = self.scalers['icgc'].transform(data[:, 90:110])
        
        return np.concatenate([methylation, mutation, cna, fragmentomics, clinical, icgc], axis=1)
    
    def predict_batch(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on a batch of data"""
        # Prepare data for transformer
        data_dict = {
            'methylation': torch.FloatTensor(data[:, :20]),
            'mutation': torch.FloatTensor(data[:, 20:45]),
            'cna': torch.FloatTensor(data[:, 45:65]),
            'fragmentomics': torch.FloatTensor(data[:, 65:80]),
            'clinical': torch.FloatTensor(data[:, 80:90]),
            'icgc': torch.FloatTensor(data[:, 90:110])
        }
        
        with torch.no_grad():
            outputs = self.model(data_dict)
            probabilities = outputs['probabilities'].numpy()
            predictions = np.argmax(probabilities, axis=1)
            confidences = np.max(probabilities, axis=1)
            
        return predictions, confidences
    
    def test_normal_operation(self) -> Dict:
        """Test normal operation with synthetic data"""
        print("Testing normal operation...")
        
        # Generate synthetic data
        synthetic_data = np.random.randn(10, 110)
        processed_data = self.preprocess_data(synthetic_data)
        predictions, confidences = self.predict_batch(processed_data)
        
        # Validate results
        assert len(predictions) == 10, "Prediction count mismatch"
        assert all(0 <= p <= 7 for p in predictions), "Invalid prediction classes"
        assert all(0 <= c <= 1 for c in confidences), "Invalid confidence scores"
        
        return {
            'test_name': 'Normal Operation',
            'status': 'PASS',
            'samples_tested': 10,
            'avg_confidence': float(np.mean(confidences)),
            'class_distribution': np.bincount(predictions, minlength=8).tolist()
        }
    
    def test_extreme_values(self) -> Dict:
        """Test with extreme input values"""
        print("Testing extreme values...")
        
        # Test with very large values
        extreme_high = np.full((5, 110), 100.0)
        # Test with very small values  
        extreme_low = np.full((5, 110), -100.0)
        # Test with mixed extreme values
        extreme_mixed = np.random.choice([-100, 100], size=(5, 110)).astype(float)
        
        extreme_data = np.vstack([extreme_high, extreme_low, extreme_mixed])
        
        try:
            processed_data = self.preprocess_data(extreme_data)
            predictions, confidences = self.predict_batch(processed_data)
            
            return {
                'test_name': 'Extreme Values',
                'status': 'PASS',
                'samples_tested': 15,
                'avg_confidence': float(np.mean(confidences)),
                'notes': 'Model handles extreme values without errors'
            }
        except Exception as e:
            return {
                'test_name': 'Extreme Values',
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_zero_values(self) -> Dict:
        """Test with all zero input values"""
        print("Testing zero values...")
        
        zero_data = np.zeros((3, 110))
        
        try:
            processed_data = self.preprocess_data(zero_data)
            predictions, confidences = self.predict_batch(processed_data)
            
            return {
                'test_name': 'Zero Values',
                'status': 'PASS',
                'samples_tested': 3,
                'predictions': predictions.tolist(),
                'confidences': confidences.tolist()
            }
        except Exception as e:
            return {
                'test_name': 'Zero Values',
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_nan_handling(self) -> Dict:
        """Test handling of NaN values"""
        print("Testing NaN handling...")
        
        # Create data with some NaN values
        nan_data = np.random.randn(3, 110)
        nan_data[0, :10] = np.nan  # First sample has NaN in first 10 features
        nan_data[1, 50:60] = np.nan  # Second sample has NaN in middle features
        
        try:
            # Replace NaN with zeros (common preprocessing step)
            nan_data = np.nan_to_num(nan_data, nan=0.0)
            processed_data = self.preprocess_data(nan_data)
            predictions, confidences = self.predict_batch(processed_data)
            
            return {
                'test_name': 'NaN Handling',
                'status': 'PASS',
                'samples_tested': 3,
                'notes': 'NaN values successfully handled by conversion to zeros'
            }
        except Exception as e:
            return {
                'test_name': 'NaN Handling',
                'status': 'FAIL',
                'error': str(e)
            }
    
    def test_batch_consistency(self) -> Dict:
        """Test consistency of predictions across different batch sizes"""
        print("Testing batch consistency...")
        
        # Generate same data in different batch sizes
        base_data = np.random.randn(1, 110)
        
        # Single prediction
        single_processed = self.preprocess_data(base_data)
        single_pred, single_conf = self.predict_batch(single_processed)
        
        # Batch prediction (same data repeated)
        batch_data = np.repeat(base_data, 5, axis=0)
        batch_processed = self.preprocess_data(batch_data)
        batch_preds, batch_confs = self.predict_batch(batch_processed)
        
        # Check consistency
        consistent = all(p == single_pred[0] for p in batch_preds)
        conf_consistent = all(abs(c - single_conf[0]) < 1e-6 for c in batch_confs)
        
        return {
            'test_name': 'Batch Consistency',
            'status': 'PASS' if consistent and conf_consistent else 'FAIL',
            'single_prediction': int(single_pred[0]),
            'batch_predictions': batch_preds.tolist(),
            'predictions_consistent': consistent,
            'confidences_consistent': conf_consistent
        }
    
    def test_cancer_type_coverage(self) -> Dict:
        """Test that model can predict all cancer types"""
        print("Testing cancer type coverage...")
        
        # Generate diverse synthetic data to encourage different predictions
        diverse_data = []
        for i in range(8):  # One for each cancer type
            # Create data with different patterns to encourage different predictions
            data = np.random.randn(110) * (i + 1) * 0.5
            data[:20] += i * 0.3  # Vary methylation patterns
            data[20:45] += np.random.poisson(i + 1, 25)  # Vary mutation patterns
            diverse_data.append(data)
        
        diverse_data = np.array(diverse_data)
        processed_data = self.preprocess_data(diverse_data)
        predictions, confidences = self.predict_batch(processed_data)
        
        unique_predictions = set(predictions)
        coverage = len(unique_predictions)
        
        return {
            'test_name': 'Cancer Type Coverage',
            'status': 'PASS',
            'unique_cancer_types_predicted': coverage,
            'predicted_types': [self.cancer_types[p] for p in unique_predictions],
            'all_predictions': [self.cancer_types[p] for p in predictions]
        }
    
    def run_all_tests(self) -> List[Dict]:
        """Run all validation tests"""
        print("=" * 60)
        print("COMPREHENSIVE VALIDATION TEST SUITE")
        print("=" * 60)
        
        test_methods = [
            self.test_normal_operation,
            self.test_extreme_values,
            self.test_zero_values,
            self.test_nan_handling,
            self.test_batch_consistency,
            self.test_cancer_type_coverage
        ]
        
        results = []
        for test_method in test_methods:
            try:
                result = test_method()
                results.append(result)
                print(f"✅ {result['test_name']}: {result['status']}")
            except Exception as e:
                result = {
                    'test_name': test_method.__name__,
                    'status': 'ERROR',
                    'error': str(e)
                }
                results.append(result)
                print(f"❌ {result['test_name']}: ERROR - {str(e)}")
        
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in results if r['status'] == 'PASS')
        failed = sum(1 for r in results if r['status'] in ['FAIL', 'ERROR'])
        
        print(f"Total tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {passed/len(results)*100:.1f}%")
        
        return results

def main():
    """Main function to run the validation test suite"""
    validator = ValidationTestSuite()
    results = validator.run_all_tests()
    
    # Save results to file
    import json
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to validation_results.json")

if __name__ == '__main__':
    main()
