"""
Sample code minh họa cách sử dụng Global Spatial Model

Ví dụ này cho thấy:
1. Cách train global model với spatial features
2. Cách load model và artifacts đã train
3. Cách predict AQI cho một location mới
4. Cách sử dụng API để predict
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib

# ============================================================================
# EXAMPLE 1: Training Global Spatial Model
# ============================================================================

def example_train_model():
    """
    Ví dụ train global model với spatial features
    """
    print("="*80)
    print("EXAMPLE 1: Training Global Spatial Model")
    print("="*80)
    
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # 1. Load data
    print("\n1. Loading data...")
    data_path = project_root / "data" / "sample_data.csv"
    df = preprocess_data(str(data_path))
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # 2. Engineer features (bao gồm spatial features)
    print("\n2. Engineering features...")
    df_featured, spatial_scaler = engineer_features(df)
    print(f"   Created {len(df_featured.columns)} features")
    print(f"   Spatial scaler fitted for lat/lon")
    
    # 3. Prepare X and y
    print("\n3. Preparing features and target...")
    exclude_cols = ['datetime', 'aqi', 'lat', 'lon']
    feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
    
    X = df_featured[feature_cols].values
    y = df_featured['aqi'].values
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # 4. Time-based split
    print("\n4. Time-based split (70/30)...")
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # 5. Train model
    print("\n5. Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 6. Evaluate
    print("\n6. Evaluating model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n   Performance Metrics:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    # 7. Save artifacts (optional - chỉ để demo)
    print("\n7. Model và artifacts có thể được lưu với joblib:")
    print("   joblib.dump(model, 'models/xgboost_global.pkl')")
    print("   joblib.dump(feature_cols, 'models/feature_columns_global.pkl')")
    print("   joblib.dump(spatial_scaler, 'models/spatial_scaler.pkl')")
    
    return model, feature_cols, spatial_scaler


# ============================================================================
# EXAMPLE 2: Loading Model và Predicting
# ============================================================================

def example_load_and_predict():
    """
    Ví dụ load model đã train và predict cho location mới
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Loading Model và Predicting")
    print("="*80)
    
    from src.data_preprocessing import preprocess_data
    from src.feature_engineering import engineer_features
    
    # 1. Load artifacts
    print("\n1. Loading model artifacts...")
    models_dir = project_root / "models"
    
    try:
        model = joblib.load(models_dir / "xgboost_global.pkl")
        feature_cols = joblib.load(models_dir / "feature_columns_global.pkl")
        spatial_scaler = joblib.load(models_dir / "spatial_scaler.pkl")
        print(f"   ✅ Loaded model")
        print(f"   ✅ Loaded {len(feature_cols)} feature columns")
        print(f"   ✅ Loaded spatial scaler")
    except FileNotFoundError:
        print("   ❌ Artifacts not found. Run training script first:")
        print("      python scripts/train_global_model.py")
        return
    
    # 2. Load historical data (cần cho lag/rolling features)
    print("\n2. Loading historical data...")
    data_path = project_root / "data" / "sample_data.csv"
    historical_data = pd.read_csv(data_path)
    historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
    print(f"   Loaded {len(historical_data)} historical records")
    
    # 3. Prepare new input
    print("\n3. Preparing new input for prediction...")
    new_input = pd.DataFrame([{
        'datetime': pd.Timestamp('2020-11-25 12:00:00'),
        'lat': 106.7075,      # Ho Chi Minh City area
        'lon': 10.804,
        'co': 700.0,
        'no': 8.0,
        'no2': 22.0,
        'o3': 60.0,
        'so2': 20.0,
        'pm2_5': 25.0,
        'pm10': 60.0,
        'nh3': 9.0,
        'aqi': 0  # Dummy value
    }])
    
    print("   Input data:")
    print(f"   - Location: lat={new_input['lat'].iloc[0]}, lon={new_input['lon'].iloc[0]}")
    print(f"   - Datetime: {new_input['datetime'].iloc[0]}")
    print(f"   - CO: {new_input['co'].iloc[0]}, NO: {new_input['no'].iloc[0]}")
    
    # 4. Combine with historical data
    print("\n4. Combining with historical data...")
    combined = pd.concat([historical_data, new_input], ignore_index=True)
    
    # 5. Engineer features với spatial_scaler đã load
    print("\n5. Engineering features...")
    featured, _ = engineer_features(combined, spatial_scaler=spatial_scaler)
    
    # 6. Get last row (new input) và select features
    print("\n6. Preparing features for prediction...")
    last_row = featured.iloc[-1:][feature_cols]
    
    # 7. Predict
    print("\n7. Predicting AQI...")
    prediction = model.predict(last_row.values)[0]
    
    print(f"\n   🎯 Predicted AQI: {prediction:.2f}")
    
    # Interpret AQI level
    if prediction < 2:
        level = "Good (Tốt)"
    elif prediction < 3:
        level = "Fair (Trung bình)"
    elif prediction < 4:
        level = "Moderate (Kém)"
    else:
        level = "Poor (Xấu)"
    
    print(f"   📊 AQI Level: {level}")
    
    return prediction


# ============================================================================
# EXAMPLE 3: Using API
# ============================================================================

def example_api_usage():
    """
    Ví dụ sử dụng API để predict
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Using API for Prediction")
    print("="*80)
    
    print("\n1. Start API server:")
    print("   cd api && python app.py")
    print("   (Server sẽ chạy tại http://localhost:8000)")
    
    print("\n2. Check health status:")
    print("   curl http://localhost:8000/health")
    
    print("\n3. Make prediction request:")
    print("""   curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "lat": 106.7075,
       "lon": 10.804,
       "co": 704.51,
       "no": 8.31,
       "no2": 21.89,
       "o3": 63.35,
       "so2": 21.33,
       "pm2_5": 25.13,
       "pm10": 63.95,
       "nh3": 9.5
     }'""")
    
    print("\n4. Response example:")
    print("""   {
     "predicted_aqi": 3.02,
     "model_name": "xgboost_global"
   }""")
    
    print("\n5. Using Python requests library:")
    print("""
import requests

url = "http://localhost:8000/predict"
data = {
    "lat": 106.7075,
    "lon": 10.804,
    "co": 704.51,
    "no": 8.31,
    "no2": 21.89,
    "o3": 63.35,
    "so2": 21.33,
    "pm2_5": 25.13,
    "pm10": 63.95,
    "nh3": 9.5
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted AQI: {result['predicted_aqi']}")
""")


# ============================================================================
# EXAMPLE 4: Predict cho nhiều locations
# ============================================================================

def example_batch_predictions():
    """
    Ví dụ predict cho nhiều locations khác nhau
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Predictions for Multiple Locations")
    print("="*80)
    
    from src.feature_engineering import engineer_features
    
    # Load artifacts
    print("\n1. Loading artifacts...")
    models_dir = project_root / "models"
    
    try:
        model = joblib.load(models_dir / "xgboost_global.pkl")
        feature_cols = joblib.load(models_dir / "feature_columns_global.pkl")
        spatial_scaler = joblib.load(models_dir / "spatial_scaler.pkl")
    except FileNotFoundError:
        print("   ❌ Artifacts not found. Run training script first.")
        return
    
    # Multiple locations
    print("\n2. Predicting for multiple locations...")
    locations = [
        {"name": "Ho Chi Minh City", "lat": 106.7075, "lon": 10.804},
        {"name": "Hanoi", "lat": 105.8342, "lon": 21.0278},
        {"name": "Da Nang", "lat": 108.2022, "lon": 16.0544},
    ]
    
    # Load historical data
    data_path = project_root / "data" / "sample_data.csv"
    historical_data = pd.read_csv(data_path)
    historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
    
    results = []
    
    for loc in locations:
        # Create input for this location
        new_input = pd.DataFrame([{
            'datetime': pd.Timestamp('2020-11-25 12:00:00'),
            'lat': loc['lat'],
            'lon': loc['lon'],
            'co': 700.0,
            'no': 8.0,
            'no2': 22.0,
            'o3': 60.0,
            'so2': 20.0,
            'pm2_5': 25.0,
            'pm10': 60.0,
            'nh3': 9.0,
            'aqi': 0
        }])
        
        # Combine and engineer features
        combined = pd.concat([historical_data, new_input], ignore_index=True)
        featured, _ = engineer_features(combined, spatial_scaler=spatial_scaler)
        
        # Predict
        last_row = featured.iloc[-1:][feature_cols]
        prediction = model.predict(last_row.values)[0]
        
        results.append({
            'location': loc['name'],
            'lat': loc['lat'],
            'lon': loc['lon'],
            'predicted_aqi': prediction
        })
    
    # Display results
    print("\n   Results:")
    print("   " + "-"*60)
    for result in results:
        print(f"   {result['location']:20s} | AQI: {result['predicted_aqi']:.2f}")
    print("   " + "-"*60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GLOBAL SPATIAL MODEL - COMPLETE EXAMPLES")
    print("="*80)
    
    # Chọn example nào muốn chạy
    import argparse
    parser = argparse.ArgumentParser(description='Global Spatial Model Examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4],
        help='Example number to run (1-4). If not specified, shows all examples.'
    )
    args = parser.parse_args()
    
    if args.example == 1 or args.example is None:
        example_train_model()
    
    if args.example == 2 or args.example is None:
        example_load_and_predict()
    
    if args.example == 3 or args.example is None:
        example_api_usage()
    
    if args.example == 4 or args.example is None:
        example_batch_predictions()
    
    print("\n" + "="*80)
    print("✅ Examples completed!")
    print("="*80)
    print("\nTo run specific example:")
    print("  python examples/global_spatial_example.py --example 1")
    print("  python examples/global_spatial_example.py --example 2")
    print("  python examples/global_spatial_example.py --example 3")
    print("  python examples/global_spatial_example.py --example 4")
    print()
