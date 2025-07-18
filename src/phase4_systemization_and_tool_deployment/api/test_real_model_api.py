#!/usr/bin/env python3
"""
Test Client for Cancer Alpha Real Model API
==========================================

This script tests the real model API with various requests to ensure
it's working correctly with the Phase 2 trained models.

Author: Cancer Alpha Research Team
Date: July 18, 2025
"""

import requests
import json
import time
import random
from typing import Dict, Any
import numpy as np

class APITestClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def test_connection(self) -> bool:
        """Test if API is reachable"""
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except:
            return False
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/models/info")
        return response.json()
    
    def get_cancer_types(self) -> Dict[str, Any]:
        """Get available cancer types"""
        response = self.session.get(f"{self.base_url}/cancer-types")
        return response.json()
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance"""
        response = self.session.get(f"{self.base_url}/models/feature-importance")
        return response.json()
    
    def generate_sample_features(self, num_features: int = 110) -> Dict[str, float]:
        """Generate sample genomic features for testing"""
        features = {}
        
        # Generate realistic feature values
        for i in range(num_features):
            # Different feature types with different distributions
            if i < 20:  # First 20 features: mutation counts (0-10)
                features[f"feature_{i}"] = random.randint(0, 10)
            elif i < 40:  # Next 20: copy number variations (-2 to 2)
                features[f"feature_{i}"] = random.uniform(-2, 2)
            elif i < 60:  # Next 20: methylation levels (0-1)
                features[f"feature_{i}"] = random.uniform(0, 1)
            elif i < 80:  # Next 20: gene expression (log2 values, -5 to 5)
                features[f"feature_{i}"] = random.uniform(-5, 5)
            else:  # Remaining: other omics data
                features[f"feature_{i}"] = random.uniform(-1, 1)
        
        return features
    
    def make_prediction(self, patient_id: str, model_type: str = "ensemble") -> Dict[str, Any]:
        """Make a prediction request"""
        features = self.generate_sample_features()
        
        request_data = {
            "patient_id": patient_id,
            "age": random.randint(30, 85),
            "gender": random.choice(["M", "F"]),
            "features": features,
            "model_type": model_type
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=request_data
        )
        
        return response.json(), response.status_code
    
    def run_comprehensive_test(self):
        """Run comprehensive tests of the API"""
        print("🧪 Starting Comprehensive API Tests")
        print("=" * 50)
        
        # Test 1: Connection
        print("\n1. Testing API Connection...")
        if self.test_connection():
            print("✅ API is reachable")
        else:
            print("❌ API is not reachable")
            return
        
        # Test 2: Health Check
        print("\n2. Testing Health Check...")
        try:
            health = self.get_health()
            print(f"✅ Health Status: {health['status']}")
            print(f"   Models Loaded: {health['models_loaded']}")
            print(f"   Message: {health['message']}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        
        # Test 3: Model Info
        print("\n3. Testing Model Information...")
        try:
            model_info = self.get_model_info()
            print(f"✅ Loaded Models: {model_info['loaded_models']}")
            print(f"   Scaler Loaded: {model_info['scaler_loaded']}")
            print(f"   Feature Count: {model_info['feature_count']}")
            
            if model_info.get('model_performance'):
                print("   Model Performance:")
                for model, perf in model_info['model_performance'].items():
                    print(f"     • {model}: Test Acc = {perf['test_accuracy']}, CV = {perf['cv_mean']}")
        except Exception as e:
            print(f"❌ Model info failed: {e}")
        
        # Test 4: Cancer Types
        print("\n4. Testing Cancer Types...")
        try:
            cancer_types = self.get_cancer_types()
            print(f"✅ Available Cancer Types: {cancer_types['cancer_types']}")
            print(f"   Total Types: {cancer_types['total_types']}")
        except Exception as e:
            print(f"❌ Cancer types failed: {e}")
        
        # Test 5: Feature Importance
        print("\n5. Testing Feature Importance...")
        try:
            feature_importance = self.get_feature_importance()
            print(f"✅ Total Features: {feature_importance['total_features']}")
            print("   Top 5 Features:")
            for i, (feature, importance) in enumerate(list(feature_importance['top_features'].items())[:5]):
                print(f"     {i+1}. {feature}: {importance:.6f}")
        except Exception as e:
            print(f"❌ Feature importance failed: {e}")
        
        # Test 6: Predictions with Different Models
        print("\n6. Testing Predictions...")
        
        # Get available models first
        try:
            model_info = self.get_model_info()
            available_models = model_info['loaded_models']
            print(f"   Available models: {available_models}")
        except:
            available_models = ["ensemble"]
        
        for model_type in available_models:
            print(f"\n   Testing {model_type} model...")
            try:
                patient_id = f"TEST_PATIENT_{random.randint(1000, 9999)}"
                result, status_code = self.make_prediction(patient_id, model_type)
                
                if status_code == 200:
                    print(f"   ✅ {model_type} prediction successful")
                    print(f"      Patient ID: {result['patient_id']}")
                    print(f"      Predicted: {result['predicted_cancer_type']} ({result['predicted_cancer_name']})")
                    print(f"      Confidence: {result['confidence']:.3f}")
                    print(f"      Processing Time: {result['processing_time_ms']:.2f}ms")
                    
                    # Show top 3 probabilities
                    prob_dist = result['probability_distribution']
                    sorted_probs = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("      Top 3 Probabilities:")
                    for cancer_type, prob in sorted_probs:
                        print(f"        {cancer_type}: {prob:.3f}")
                else:
                    print(f"   ❌ {model_type} prediction failed with status {status_code}")
                    print(f"      Error: {result}")
                    
            except Exception as e:
                print(f"   ❌ {model_type} prediction failed: {e}")
        
        # Test 7: Load Testing
        print("\n7. Running Load Test (10 concurrent predictions)...")
        try:
            start_time = time.time()
            predictions = []
            
            for i in range(10):
                patient_id = f"LOAD_TEST_{i}"
                result, status_code = self.make_prediction(patient_id, "ensemble")
                if status_code == 200:
                    predictions.append(result['processing_time_ms'])
                else:
                    print(f"   ❌ Load test request {i} failed")
            
            end_time = time.time()
            
            if predictions:
                avg_processing_time = sum(predictions) / len(predictions)
                total_time = (end_time - start_time) * 1000
                print(f"   ✅ Load test completed")
                print(f"      Successful predictions: {len(predictions)}/10")
                print(f"      Average processing time: {avg_processing_time:.2f}ms")
                print(f"      Total time: {total_time:.2f}ms")
            else:
                print(f"   ❌ Load test failed - no successful predictions")
                
        except Exception as e:
            print(f"   ❌ Load test failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 Comprehensive API Tests Completed!")

def main():
    """Main function to run tests"""
    print("🚀 Cancer Alpha API - Real Model Test Client")
    print("=" * 50)
    
    # Check if API is running
    client = APITestClient()
    
    print("Checking if API is running...")
    if not client.test_connection():
        print("❌ API is not running. Please start the API first:")
        print("   python real_model_api.py")
        return
    
    print("✅ API is running")
    
    # Run comprehensive tests
    client.run_comprehensive_test()
    
    print("\n🔍 Manual Testing:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("   • Health Check: http://localhost:8000/health")
    print("   • Model Info: http://localhost:8000/models/info")

if __name__ == "__main__":
    main()
