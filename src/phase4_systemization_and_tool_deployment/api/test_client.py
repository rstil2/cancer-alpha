#!/usr/bin/env python3
"""
Simple Test Client for Cancer Alpha API
======================================

This script shows you how to interact with the Cancer Alpha API.
It's like a simple example of how a doctor or researcher would use our system.

To use this:
1. First run the API: python simple_api.py
2. Then run this test client: python test_client.py

Author: Cancer Alpha Research Team
Date: July 17, 2025
"""

import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test if the API is running"""
    print("🧪 Testing API connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy! Status: {data['status']}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running!")
        return False

def get_cancer_types():
    """Get the list of cancer types the API can predict"""
    print("\n📋 Getting cancer types...")
    try:
        response = requests.get(f"{API_BASE_URL}/cancer-types")
        if response.status_code == 200:
            data = response.json()
            print("✅ Available cancer types:")
            for cancer_type, description in data["descriptions"].items():
                print(f"   • {cancer_type}: {description}")
            return data["cancer_types"]
        else:
            print(f"❌ Failed to get cancer types: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error getting cancer types: {e}")
        return []

def make_prediction(patient_id, age, gender, features):
    """Make a cancer prediction for a patient"""
    print(f"\n🔮 Making prediction for patient {patient_id}...")
    
    # Prepare the data to send
    request_data = {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "features": features
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print("✅ Prediction successful!")
            print(f"   Patient: {prediction['patient_id']}")
            print(f"   Predicted cancer type: {prediction['predicted_cancer_type']}")
            print(f"   Confidence: {prediction['confidence']:.2%}")
            print(f"   Timestamp: {prediction['timestamp']}")
            return prediction
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None

def main():
    """Main function to run all tests"""
    print("🚀 Cancer Alpha API Test Client")
    print("=" * 50)
    
    # Test 1: Check if API is running
    if not test_api_connection():
        print("\n❌ API is not running. Please start it first with: python simple_api.py")
        return
    
    # Test 2: Get available cancer types
    cancer_types = get_cancer_types()
    
    # Test 3: Make some example predictions
    print("\n🧬 Making example predictions...")
    
    # Example 1: High-risk breast cancer patient
    print("\n--- Example 1: High-risk patient ---")
    make_prediction(
        patient_id="BRCA_001",
        age=55,
        gender="female",
        features={
            "BRCA1_expression": 8.5,
            "BRCA2_expression": 7.2,
            "estrogen_receptor": 4.1,
            "tumor_size": 3.5
        }
    )
    
    # Example 2: Lung cancer patient
    print("\n--- Example 2: Lung cancer patient ---")
    make_prediction(
        patient_id="LUAD_001",
        age=62,
        gender="male",
        features={
            "smoking_history": 1.0,
            "EGFR_mutation": 2.1,
            "KRAS_mutation": 3.2,
            "lung_function": 0.6
        }
    )
    
    # Example 3: Low-risk patient
    print("\n--- Example 3: Low-risk patient ---")
    make_prediction(
        patient_id="LOW_001",
        age=35,
        gender="female",
        features={
            "gene_expression": 0.5,
            "methylation": 0.2,
            "mutation_count": 1
        }
    )
    
    # Test 4: Get API statistics
    print("\n📊 Getting API statistics...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ API Statistics:")
            print(f"   Total predictions: {stats['total_predictions_made']}")
            print(f"   Most common prediction: {stats['most_common_prediction']}")
            print(f"   Average confidence: {stats['average_confidence']:.2%}")
            print(f"   Uptime: {stats['api_uptime']}")
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
    
    print("\n🎉 All tests completed!")
    print("\nNext steps:")
    print("1. Try the interactive API docs at: http://localhost:8000/docs")
    print("2. Modify this script to test your own data")
    print("3. Build a web interface to make it easier to use")

if __name__ == "__main__":
    main()
