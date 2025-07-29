#!/usr/bin/env python3
"""
API Test Suite for Cancer Genomics AI Classifier
================================================

Comprehensive tests for the production FastAPI backend.

Author: Cancer Alpha Research Team
Date: July 28, 2025
"""

import requests
import json
import time
import numpy as np
from typing import Dict, List
import sys
import os

class APITester:
    """Test suite for the Cancer Genomics AI API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "demo-key-123"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.test_results = []
    
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            "test_name": test_name,
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   {details}")
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "healthy" and data.get("model_loaded"):
                    self.log_test("Health Check", "PASS", 
                                f"API healthy, uptime: {data.get('uptime_seconds', 0):.1f}s")
                    return True
                else:
                    self.log_test("Health Check", "FAIL", 
                                f"API unhealthy: {data}")
                    return False
            else:
                self.log_test("Health Check", "FAIL", 
                            f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Health Check", "FAIL", f"Exception: {str(e)}")
            return False
    
    def test_model_info_endpoint(self) -> bool:
        """Test the model info endpoint"""
        try:
            response = requests.get(
                f"{self.base_url}/model/info",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                expected_fields = ['model_type', 'cancer_types', 'feature_count', 'validation_accuracy']
                if all(field in data for field in expected_fields):
                    self.log_test("Model Info", "PASS", 
                                f"Model: {data.get('model_type')}, Accuracy: {data.get('validation_accuracy')}")
                    return True
                else:
                    self.log_test("Model Info", "FAIL", "Missing expected fields")
                    return False
            else:
                self.log_test("Model Info", "FAIL", f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Model Info", "FAIL", f"Exception: {str(e)}")
            return False
    
    def test_single_prediction(self) -> bool:
        """Test single prediction endpoint"""
        try:
            # Generate synthetic test data
            np.random.seed(42)
            test_features = np.random.randn(110).tolist()
            
            payload = {
                "features": test_features,
                "patient_id": "test_patient_001",
                "include_explanations": True,
                "include_biological_insights": True
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                required_fields = ['prediction_id', 'predicted_cancer_type', 'confidence_score', 
                                 'class_probabilities', 'processing_time_ms']
                
                if all(field in data for field in required_fields):
                    cancer_type = data.get('predicted_cancer_type')
                    confidence = data.get('confidence_score')
                    processing_time = data.get('processing_time_ms')
                    
                    self.log_test("Single Prediction", "PASS", 
                                f"Predicted: {cancer_type}, Confidence: {confidence:.3f}, "
                                f"Time: {processing_time:.1f}ms")
                    return True
                else:
                    self.log_test("Single Prediction", "FAIL", "Missing required fields")
                    return False
            else:
                self.log_test("Single Prediction", "FAIL", 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Single Prediction", "FAIL", f"Exception: {str(e)}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        try:
            # Generate multiple synthetic test samples
            np.random.seed(123)
            batch_size = 5
            test_samples = [np.random.randn(110).tolist() for _ in range(batch_size)]
            patient_ids = [f"batch_patient_{i:03d}" for i in range(batch_size)]
            
            payload = {
                "samples": test_samples,
                "patient_ids": patient_ids,
                "include_explanations": False
            }
            
            response = requests.post(
                f"{self.base_url}/predict/batch",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if (data.get('samples_processed') == batch_size and 
                    len(data.get('results', [])) == batch_size):
                    
                    processing_time = data.get('processing_time_ms')
                    avg_time_per_sample = processing_time / batch_size
                    
                    self.log_test("Batch Prediction", "PASS", 
                                f"Processed {batch_size} samples in {processing_time:.1f}ms "
                                f"({avg_time_per_sample:.1f}ms/sample)")
                    return True
                else:
                    self.log_test("Batch Prediction", "FAIL", 
                                f"Expected {batch_size} results, got {len(data.get('results', []))}")
                    return False
            else:
                self.log_test("Batch Prediction", "FAIL", 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Batch Prediction", "FAIL", f"Exception: {str(e)}")
            return False
    
    def test_authentication(self) -> bool:
        """Test API key authentication"""
        try:
            # Test with invalid API key
            invalid_headers = {
                "Authorization": "Bearer invalid-key",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/model/info",
                headers=invalid_headers,
                timeout=10
            )
            
            if response.status_code == 401:
                self.log_test("Authentication", "PASS", "Invalid API key correctly rejected")
                return True
            else:
                self.log_test("Authentication", "FAIL", 
                            f"Expected 401, got {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Authentication", "FAIL", f"Exception: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid input"""
        try:
            # Test with wrong number of features
            payload = {
                "features": [1.0, 2.0, 3.0],  # Only 3 features instead of 110
                "patient_id": "error_test"
            }
            
            response = requests.post(
                f"{self.base_url}/predict",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 422:  # Validation error
                self.log_test("Error Handling", "PASS", 
                            "Invalid input correctly rejected with validation error")
                return True
            else:
                self.log_test("Error Handling", "FAIL", 
                            f"Expected 422, got {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Error Handling", "FAIL", f"Exception: {str(e)}")
            return False
    
    def test_performance(self) -> bool:
        """Test API performance with multiple requests"""
        try:
            num_requests = 10
            start_time = time.time()
            successful_requests = 0
            
            np.random.seed(456)
            
            for i in range(num_requests):
                test_features = np.random.randn(110).tolist()
                payload = {
                    "features": test_features,
                    "patient_id": f"perf_test_{i}",
                    "include_explanations": False,
                    "include_biological_insights": False
                }
                
                response = requests.post(
                    f"{self.base_url}/predict",
                    headers=self.headers,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    successful_requests += 1
            
            total_time = time.time() - start_time
            avg_time_per_request = (total_time / num_requests) * 1000
            
            if successful_requests == num_requests:
                self.log_test("Performance Test", "PASS", 
                            f"{num_requests} requests in {total_time:.2f}s "
                            f"({avg_time_per_request:.1f}ms/request)")
                return True
            else:
                self.log_test("Performance Test", "FAIL", 
                            f"Only {successful_requests}/{num_requests} requests successful")
                return False
                
        except Exception as e:
            self.log_test("Performance Test", "FAIL", f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all API tests"""
        print("=" * 60)
        print("CANCER GENOMICS AI API TEST SUITE")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")
        print(f"Using API key: {self.api_key}")
        print()
        
        # Run all tests
        tests = [
            self.test_health_endpoint,
            self.test_model_info_endpoint,
            self.test_authentication,
            self.test_error_handling,
            self.test_single_prediction,
            self.test_batch_prediction,
            self.test_performance
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
        
        print()
        print("=" * 60)
        print("API TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {(passed/total)*100:.1f}%")
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': (passed/total)*100,
            'test_results': self.test_results
        }

def main():
    """Main function to run API tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Cancer Genomics AI API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--key", default="demo-key-123", help="API key")
    
    args = parser.parse_args()
    
    tester = APITester(base_url=args.url, api_key=args.key)
    results = tester.run_all_tests()
    
    # Save results
    with open('api_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to api_test_results.json")
    
    # Exit with appropriate code
    sys.exit(0 if results['success_rate'] == 100 else 1)

if __name__ == '__main__':
    main()
