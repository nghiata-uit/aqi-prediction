#!/usr/bin/env python3
"""
Quick verification script for AQI Prediction System
Tests all major components to ensure the system is working correctly
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from src.data_preprocessing import preprocess_data, check_data_quality
        from src.feature_engineering import engineer_features
        from src.model_training import AQIModelTrainer
        from src.model_evaluation import calculate_metrics
        from src.prediction import predict_next_24h
        print("✅ All imports successful\n")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}\n")
        return False

def test_data_loading():
    """Test data loading"""
    print("Testing data loading...")
    try:
        from src.data_preprocessing import preprocess_data
        df = preprocess_data('data/sample_data.csv')
        assert df.shape[0] > 0, "No data loaded"
        assert 'aqi' in df.columns, "Missing AQI column"
        print(f"✅ Data loaded successfully: {df.shape}\n")
        return True, df
    except Exception as e:
        print(f"❌ Data loading failed: {e}\n")
        return False, None

def test_feature_engineering(df):
    """Test feature engineering"""
    print("Testing feature engineering...")
    try:
        from src.feature_engineering import engineer_features
        df_feat = engineer_features(df)
        assert df_feat.shape[1] > df.shape[1], "No new features created"
        print(f"✅ Features engineered: {df.shape[1]} → {df_feat.shape[1]}\n")
        return True
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}\n")
        return False

def test_models():
    """Test that models exist and can be loaded"""
    print("Testing model loading...")
    try:
        import joblib
        rf = joblib.load('models/random_forest.pkl')
        xgb = joblib.load('models/xgboost.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ All models loaded successfully\n")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}\n")
        return False

def test_results():
    """Test that result files exist"""
    print("Testing result files...")
    try:
        required_files = [
            'results/24h_predictions.csv',
            'results/model_comparison.csv',
            'results/feature_importance.csv',
            'results/rf_predictions.png',
            'results/xgb_predictions.png',
            'results/model_comparison.png',
            'results/feature_importance.png',
            'results/24h_forecast.png'
        ]
        
        for file in required_files:
            path = Path(file)
            assert path.exists(), f"Missing file: {file}"
        
        print("✅ All result files present\n")
        return True
    except Exception as e:
        print(f"❌ Result files check failed: {e}\n")
        return False

def test_documentation():
    """Test that documentation exists"""
    print("Testing documentation...")
    try:
        readme = Path('README.md')
        notebook = Path('notebooks/AQI_Prediction_Analysis.ipynb')
        
        assert readme.exists(), "README.md not found"
        assert notebook.exists(), "Jupyter notebook not found"
        
        readme_size = readme.stat().st_size
        notebook_size = notebook.stat().st_size
        
        print(f"✅ README.md exists ({readme_size:,} bytes)")
        print(f"✅ Jupyter notebook exists ({notebook_size:,} bytes)\n")
        return True
    except Exception as e:
        print(f"❌ Documentation check failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("AQI PREDICTION SYSTEM - VERIFICATION TEST")
    print("=" * 70 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Documentation", test_documentation),
        ("Models", test_models),
        ("Results", test_results),
    ]
    
    # Data-dependent tests
    success, df = test_data_loading()
    if success and df is not None:
        tests.insert(2, ("Feature Engineering", lambda: test_feature_engineering(df)))
    
    results = []
    for test_name, test_func in tests:
        if callable(test_func):
            result = test_func() if test_name not in ["Feature Engineering"] else test_func()
        else:
            result = False
        results.append((test_name, result))
    
    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("=" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All verification tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
