#!/usr/bin/env python3
"""
Comprehensive test script for AQI Prediction System
Tests training pipeline, model loading, and API functionality
"""
import sys
import json
import subprocess
import time
from pathlib import Path
import requests

def test_training_pipeline():
    """Test the model training pipeline"""
    print("\n" + "="*70)
    print("TEST 1: Training Pipeline")
    print("="*70)
    
    try:
        # Check if models already exist
        models_dir = Path("models")
        model_files = list(models_dir.glob("lat_*_best.pkl"))
        
        if model_files:
            print("✅ Models already trained")
            print(f"   Found {len(model_files)} location model(s)")
            
            # Check metrics file
            for model_file in model_files:
                metrics_file = model_file.parent / model_file.name.replace('_best.pkl', '_metrics.json')
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    print(f"   {model_file.stem}: {metrics['best_model']} (MAE={metrics['metrics']['mae']})")
                    return True
        else:
            print("❌ No trained models found")
            print("   Run: python train_models.py --data data/sample_data.csv")
            return False
            
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        return False


def test_model_loading():
    """Test loading models and making predictions"""
    print("\n" + "="*70)
    print("TEST 2: Model Loading")
    print("="*70)
    
    try:
        import joblib
        from pathlib import Path
        
        models_dir = Path("models")
        
        # Test loading scaler
        scaler_path = models_dir / "scaler.pkl"
        if not scaler_path.exists():
            print("❌ Scaler not found")
            return False
        
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully")
        
        # Test loading best model
        model_files = list(models_dir.glob("lat_*_best.pkl"))
        if not model_files:
            print("❌ No model files found")
            return False
        
        model = joblib.load(model_files[0])
        print(f"✅ Model loaded successfully: {model_files[0].name}")
        
        # Test prediction
        import numpy as np
        test_data = np.random.rand(1, 161)  # 161 features (adjust based on actual model)
        pred = model.predict(test_data)
        print(f"✅ Test prediction: AQI = {pred[0]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_health(base_url="http://localhost:8000"):
    """Test API health endpoint"""
    print("\n" + "="*70)
    print("TEST 3: API Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API")
        print("   Make sure API is running: uvicorn api.main:app")
        return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False


def test_api_prediction(base_url="http://localhost:8000"):
    """Test API prediction endpoint"""
    print("\n" + "="*70)
    print("TEST 4: API Prediction")
    print("="*70)
    
    try:
        payload = {
            "lat": 10.7828,
            "lon": 106.6953,
            "current_data": {
                "co": 1842.5,
                "no": 34.87,
                "no2": 54.84,
                "o3": 0,
                "so2": 60.56,
                "pm2_5": 67.8,
                "pm10": 81.92,
                "nh3": 10.77
            },
            "datetime": "2022-10-12 20:00:00"
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful")
            print(f"   Location: ({data['location']['lat']}, {data['location']['lon']})")
            print(f"   Predicted AQI: {data['predicted_aqi']}")
            print(f"   Model used: {data['model_used']}")
            print(f"   Confidence: {data['confidence_score']}")
            print(f"   MAE: {data['metrics']['mae']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False


def test_api_models(base_url="http://localhost:8000"):
    """Test API models list endpoint"""
    print("\n" + "="*70)
    print("TEST 5: API Models List")
    print("="*70)
    
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models list retrieved")
            print(f"   Total locations: {data['total_locations']}")
            for model in data['models']:
                loc = model['location']
                print(f"   - Location ({loc['lat']}, {loc['lon']}): {model['model_type']}")
            return True
        else:
            print(f"❌ Models list failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Models list test failed: {e}")
        return False


def test_api_comparison(base_url="http://localhost:8000"):
    """Test API model comparison endpoint"""
    print("\n" + "="*70)
    print("TEST 6: API Model Comparison")
    print("="*70)
    
    try:
        # First get available locations
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code != 200:
            print("❌ Could not get models list")
            return False
        
        models_data = response.json()
        if models_data['total_locations'] == 0:
            print("❌ No models available")
            return False
        
        # Get first location
        location = models_data['models'][0]['location']
        lat = location['lat']
        lon = location['lon']
        
        # Get comparison
        response = requests.get(
            f"{base_url}/model-comparison/{lat}/{lon}",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model comparison retrieved")
            print(f"   Location: ({data['location']['lat']}, {data['location']['lon']})")
            print(f"   Best model: {data['best_model']}")
            print(f"   Models compared: {len(data['models'])}")
            for model_name, metrics in data['models'].items():
                print(f"   - {model_name}: MAE={metrics['mae']}, R²={metrics.get('r2', 'N/A')}")
            return True
        else:
            print(f"❌ Model comparison failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model comparison test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 AQI PREDICTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Training Pipeline
    results.append(("Training Pipeline", test_training_pipeline()))
    
    # Test 2: Model Loading
    results.append(("Model Loading", test_model_loading()))
    
    # Test 3-6: API Tests (only if API is running)
    api_running = False
    try:
        requests.get("http://localhost:8000/health", timeout=2)
        api_running = True
    except:
        print("\n⚠️  API is not running. Skipping API tests.")
        print("   To run API tests, start the API with:")
        print("   uvicorn api.main:app --reload")
    
    if api_running:
        results.append(("API Health", test_api_health()))
        results.append(("API Prediction", test_api_prediction()))
        results.append(("API Models List", test_api_models()))
        results.append(("API Model Comparison", test_api_comparison()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
