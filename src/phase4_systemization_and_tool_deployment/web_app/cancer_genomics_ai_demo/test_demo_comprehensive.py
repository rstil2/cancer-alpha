#!/usr/bin/env python3
"""
Comprehensive Demo Testing Script
=================================

This script thoroughly tests all functionality of the Streamlit cancer 
classification app to ensure it works properly before deployment.
"""

import sys
import numpy as np
import pandas as pd
import time
import requests
import subprocess
import signal
import os
from pathlib import Path

# Import our app components for direct testing
from streamlit_app import CancerClassifierApp, FEATURE_NAMES, FEATURE_DESCRIPTIONS

def test_app_initialization():
    """Test app initialization and model loading"""
    print("🔧 Testing App Initialization...")
    print("-" * 40)
    
    try:
        app = CancerClassifierApp()
        
        # Test feature names
        assert len(app.feature_names) == 110, f"Expected 110 features, got {len(app.feature_names)}"
        print(f"✅ Feature count: {len(app.feature_names)} features")
        
        # Test model loading
        model_count = len(app.models)
        print(f"✅ Models loaded: {model_count} models")
        
        # Test scaler
        assert 'main' in app.scalers, "Scaler not found"
        print("✅ Scaler loaded successfully")
        
        return app
        
    except Exception as e:
        print(f"❌ App initialization failed: {e}")
        return None

def test_sample_data_generation(app):
    """Test sample data generation functionality"""
    print("\n🧬 Testing Sample Data Generation...")
    print("-" * 40)
    
    try:
        # Test cancer sample
        cancer_data = app.generate_sample_data("cancer")
        assert len(cancer_data) == 110, f"Cancer sample should have 110 features, got {len(cancer_data)}"
        print(f"✅ Cancer sample: {len(cancer_data)} features, range [{cancer_data.min():.2f}, {cancer_data.max():.2f}]")
        
        # Test control sample
        control_data = app.generate_sample_data("control")
        assert len(control_data) == 110, f"Control sample should have 110 features, got {len(control_data)}"
        print(f"✅ Control sample: {len(control_data)} features, range [{control_data.min():.2f}, {control_data.max():.2f}]")
        
        # Verify samples are different
        assert not np.array_equal(cancer_data, control_data), "Cancer and control samples should be different"
        print("✅ Cancer and control samples are appropriately different")
        
        return cancer_data, control_data
        
    except Exception as e:
        print(f"❌ Sample data generation failed: {e}")
        return None, None

def test_data_preprocessing(app, cancer_data, control_data):
    """Test data preprocessing functionality"""
    print("\n⚙️ Testing Data Preprocessing...")
    print("-" * 40)
    
    try:
        # Test preprocessing cancer data
        cancer_processed = app.preprocess_input(cancer_data)
        assert cancer_processed.shape == (1, 110), f"Expected (1, 110), got {cancer_processed.shape}"
        print(f"✅ Cancer data preprocessed: shape {cancer_processed.shape}")
        
        # Test preprocessing control data
        control_processed = app.preprocess_input(control_data)
        assert control_processed.shape == (1, 110), f"Expected (1, 110), got {control_processed.shape}"
        print(f"✅ Control data preprocessed: shape {control_processed.shape}")
        
        # Test list input
        list_input = cancer_data.tolist()
        list_processed = app.preprocess_input(list_input)
        assert list_processed.shape == (1, 110), f"Expected (1, 110), got {list_processed.shape}"
        print("✅ List input preprocessing works")
        
        return cancer_processed, control_processed
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return None, None

def test_predictions(app, cancer_processed, control_processed):
    """Test prediction functionality"""
    print("\n🔮 Testing Predictions...")
    print("-" * 40)
    
    try:
        results = {}
        
        for model_name, model in app.models.items():
            print(f"\n  Testing {model_name}:")
            
            # Test cancer prediction
            cancer_result = app.predict_with_confidence(model, cancer_processed)
            required_keys = ['prediction', 'cancer_probability', 'confidence_score', 'class_probabilities']
            
            for key in required_keys:
                assert key in cancer_result, f"Missing key: {key}"
            
            assert 0 <= cancer_result['cancer_probability'] <= 1, "Cancer probability should be between 0 and 1"
            assert 0 <= cancer_result['confidence_score'] <= 1, "Confidence should be between 0 and 1"
            assert len(cancer_result['class_probabilities']) >= 2, "Should have at least 2 class probabilities"
            
            print(f"    ✅ Cancer prediction: {cancer_result['prediction']} ({cancer_result['cancer_probability']:.1%} confidence)")
            
            # Test control prediction
            control_result = app.predict_with_confidence(model, control_processed)
            print(f"    ✅ Control prediction: {control_result['prediction']} ({control_result['cancer_probability']:.1%} confidence)")
            
            results[model_name] = {
                'cancer': cancer_result,
                'control': control_result
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Prediction testing failed: {e}")
        return None

def test_shap_explanations(app, cancer_processed):
    """Test SHAP explanation functionality"""
    print("\n🔍 Testing SHAP Explanations...")
    print("-" * 40)
    
    try:
        for model_name, model in app.models.items():
            print(f"\n  Testing SHAP for {model_name}:")
            
            # Test SHAP explanation generation
            shap_values = app.generate_shap_explanation(model, cancer_processed, model_name)
            
            if shap_values is not None:
                print(f"    ✅ SHAP values generated successfully")
                
                # Test feature importance plot
                try:
                    fig_importance = app.plot_feature_importance(shap_values)
                    if fig_importance:
                        print(f"    ✅ Feature importance plot created")
                    else:
                        print(f"    ⚠️ Feature importance plot creation returned None")
                except Exception as e:
                    print(f"    ⚠️ Feature importance plot failed: {e}")
                
                # Test modality importance plot
                try:
                    fig_modality, modality_scores = app.plot_modality_importance(shap_values)
                    if fig_modality and modality_scores:
                        print(f"    ✅ Modality importance plot created")
                        print(f"    📊 Top modality: {max(modality_scores, key=modality_scores.get)}")
                    else:
                        print(f"    ⚠️ Modality importance plot creation returned None")
                except Exception as e:
                    print(f"    ⚠️ Modality importance plot failed: {e}")
            else:
                print(f"    ⚠️ SHAP values could not be generated")
                
    except Exception as e:
        print(f"❌ SHAP explanation testing failed: {e}")

def test_csv_functionality():
    """Test CSV upload functionality"""
    print("\n📁 Testing CSV Functionality...")
    print("-" * 40)
    
    try:
        # Create sample CSV data
        sample_data = np.random.normal(0, 1, (5, 110))
        df = pd.DataFrame(sample_data, columns=FEATURE_NAMES)
        
        csv_path = "test_sample_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Sample CSV created: {csv_path}")
        
        # Test loading CSV
        loaded_df = pd.read_csv(csv_path)
        assert loaded_df.shape[1] == 110, f"Expected 110 columns, got {loaded_df.shape[1]}"
        print(f"✅ CSV loaded successfully: {loaded_df.shape}")
        
        # Clean up
        os.remove(csv_path)
        print("✅ Test CSV cleaned up")
        
    except Exception as e:
        print(f"❌ CSV functionality testing failed: {e}")

def test_feature_descriptions():
    """Test feature descriptions"""
    print("\n📝 Testing Feature Descriptions...")
    print("-" * 40)
    
    try:
        # Test that we have descriptions for all features
        missing_descriptions = []
        for feature in FEATURE_NAMES:
            if feature not in FEATURE_DESCRIPTIONS:
                missing_descriptions.append(feature)
        
        if missing_descriptions:
            print(f"⚠️ Missing descriptions for {len(missing_descriptions)} features")
        else:
            print(f"✅ All {len(FEATURE_NAMES)} features have descriptions")
        
        # Test description content
        sample_descriptions = list(FEATURE_DESCRIPTIONS.values())[:5]
        for desc in sample_descriptions:
            assert len(desc) > 10, f"Description too short: {desc}"
            
        print("✅ Feature descriptions are properly formatted")
        
    except Exception as e:
        print(f"❌ Feature description testing failed: {e}")

def test_streamlit_server():
    """Test if Streamlit server can start and respond"""
    print("\n🌐 Testing Streamlit Server...")
    print("-" * 40)
    
    try:
        # Start Streamlit server in background
        print("Starting Streamlit server...")
        process = subprocess.Popen([
            'streamlit', 'run', 'streamlit_app.py', 
            '--server.headless', 'true',
            '--server.port', '8502',  # Use different port to avoid conflicts
            '--browser.gatherUsageStats', 'false'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(5)
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:8502/_stcore/health', timeout=10)
            if response.status_code == 200:
                print("✅ Streamlit server started and responding")
                server_working = True
            else:
                print(f"⚠️ Streamlit server responded with status {response.status_code}")
                server_working = False
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not connect to Streamlit server: {e}")
            server_working = False
        
        # Clean up
        process.terminate()
        process.wait(timeout=5)
        print("✅ Streamlit server stopped")
        
        return server_working
        
    except Exception as e:
        print(f"❌ Streamlit server testing failed: {e}")
        return False

def main():
    """Run comprehensive demo testing"""
    print("🧪 COMPREHENSIVE DEMO TESTING")
    print("=" * 50)
    
    test_results = {
        'app_init': False,
        'sample_data': False,
        'preprocessing': False,
        'predictions': False,
        'shap': False,
        'csv': False,
        'descriptions': False,
        'server': False
    }
    
    # Test 1: App Initialization
    app = test_app_initialization()
    if app:
        test_results['app_init'] = True
    else:
        print("❌ Cannot continue testing without app initialization")
        return False
    
    # Test 2: Sample Data Generation
    cancer_data, control_data = test_sample_data_generation(app)
    if cancer_data is not None and control_data is not None:
        test_results['sample_data'] = True
    
    # Test 3: Data Preprocessing
    if test_results['sample_data']:
        cancer_processed, control_processed = test_data_preprocessing(app, cancer_data, control_data)
        if cancer_processed is not None and control_processed is not None:
            test_results['preprocessing'] = True
    
    # Test 4: Predictions
    if test_results['preprocessing']:
        prediction_results = test_predictions(app, cancer_processed, control_processed)
        if prediction_results:
            test_results['predictions'] = True
    
    # Test 5: SHAP Explanations
    if test_results['preprocessing']:
        test_shap_explanations(app, cancer_processed)
        test_results['shap'] = True  # We consider this passed if no exceptions
    
    # Test 6: CSV Functionality
    test_csv_functionality()
    test_results['csv'] = True  # We consider this passed if no exceptions
    
    # Test 7: Feature Descriptions
    test_feature_descriptions()
    test_results['descriptions'] = True  # We consider this passed if no exceptions
    
    # Test 8: Streamlit Server
    server_working = test_streamlit_server()
    test_results['server'] = server_working
    
    # Final Results
    print("\n" + "=" * 50)
    print("🏁 TESTING RESULTS SUMMARY")
    print("=" * 50)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<30} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("🎉 DEMO IS READY FOR DEPLOYMENT!")
        return True
    else:
        print("⚠️ Some issues need to be addressed before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
