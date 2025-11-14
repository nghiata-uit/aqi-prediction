"""
Training pipeline for AQI prediction models
Trains multiple models for each unique location (lat, lon)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import joblib
import json
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.models import (
    train_lstm_model,
    train_gru_model,
    train_prophet_model,
    predict_with_prophet,
    train_xgboost_model,
    train_ensemble_model
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_sequences(X, y, seq_length=24):
    """
    Prepare sequences for LSTM/GRU
    
    Args:
        X: Feature array
        y: Target array
        seq_length: Length of sequences
        
    Returns:
        X_seq, y_seq: Sequenced data
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    return np.array(X_seq), np.array(y_seq)


def train_random_forest_model(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    logger.info("🌲 Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    logger.info(f"   Train R² score: {train_score:.4f}")
    logger.info(f"   Validation R² score: {val_score:.4f}")
    logger.info("✅ Random Forest training completed")
    
    return model


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'r2': round(float(r2), 4)
    }


def train_models_for_location(df_location, location_id, output_dir):
    """
    Train all models for a specific location
    
    Args:
        df_location: DataFrame with data for this location
        location_id: String identifier for the location (lat_X_lon_Y)
        output_dir: Directory to save models
    """
    logger.info("\n" + "="*70)
    logger.info(f"Training models for location: {location_id}")
    logger.info("="*70)
    
    # Prepare data
    exclude_cols = ['datetime', 'aqi', 'lon', 'lat']
    feature_cols = [col for col in df_location.columns if col not in exclude_cols]
    
    X = df_location[feature_cols].values
    y = df_location['aqi'].values
    
    # Time-based split (70/15/15)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler (globally, used for all locations)
    scaler_path = output_dir / 'scaler.pkl'
    if not scaler_path.exists():
        joblib.dump(scaler, scaler_path)
        logger.info(f"✅ Scaler saved to {scaler_path}")
    
    # Dictionary to store all models and their metrics
    all_models = {}
    all_metrics = {}
    
    # 1. Train Random Forest
    try:
        rf_model = train_random_forest_model(X_train_scaled, y_train, X_val_scaled, y_val)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_metrics = calculate_metrics(y_test, rf_pred)
        all_models['RandomForest'] = rf_model
        all_metrics['RandomForest'] = rf_metrics
        logger.info(f"RandomForest metrics: {rf_metrics}")
    except Exception as e:
        logger.error(f"RandomForest training failed: {str(e)}")
    
    # 2. Train XGBoost
    try:
        xgb_model = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_metrics = calculate_metrics(y_test, xgb_pred)
        all_models['XGBoost'] = xgb_model
        all_metrics['XGBoost'] = xgb_metrics
        logger.info(f"XGBoost metrics: {xgb_metrics}")
    except Exception as e:
        logger.error(f"XGBoost training failed: {str(e)}")
    
    # 3. Train LSTM
    try:
        seq_length = 24
        X_train_seq, y_train_seq = prepare_sequences(X_train_scaled, y_train, seq_length)
        X_val_seq, y_val_seq = prepare_sequences(X_val_scaled, y_val, seq_length)
        X_test_seq, y_test_seq = prepare_sequences(X_test_scaled, y_test, seq_length)
        
        lstm_model = train_lstm_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        if lstm_model is not None:
            lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
            lstm_metrics = calculate_metrics(y_test_seq, lstm_pred)
            all_models['LSTM'] = lstm_model
            all_metrics['LSTM'] = lstm_metrics
            logger.info(f"LSTM metrics: {lstm_metrics}")
    except Exception as e:
        logger.error(f"LSTM training failed: {str(e)}")
    
    # 4. Train GRU
    try:
        if 'X_train_seq' in locals():
            gru_model = train_gru_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            if gru_model is not None:
                gru_pred = gru_model.predict(X_test_seq, verbose=0).flatten()
                gru_metrics = calculate_metrics(y_test_seq, gru_pred)
                all_models['GRU'] = gru_model
                all_metrics['GRU'] = gru_metrics
                logger.info(f"GRU metrics: {gru_metrics}")
    except Exception as e:
        logger.error(f"GRU training failed: {str(e)}")
    
    # 5. Train Prophet
    try:
        df_train_prophet = df_location.iloc[:train_size].copy()
        df_val_prophet = df_location.iloc[train_size:train_size+val_size].copy()
        
        prophet_model = train_prophet_model(df_train_prophet, df_val_prophet)
        if prophet_model is not None:
            df_test_prophet = df_location.iloc[train_size+val_size:].copy()
            prophet_pred = predict_with_prophet(prophet_model, df_test_prophet)
            if prophet_pred is not None:
                prophet_metrics = calculate_metrics(y_test, prophet_pred)
                all_models['Prophet'] = prophet_model
                all_metrics['Prophet'] = prophet_metrics
                logger.info(f"Prophet metrics: {prophet_metrics}")
    except Exception as e:
        logger.error(f"Prophet training failed: {str(e)}")
    
    # 6. Train Ensemble (exclude sequence models for simplicity)
    try:
        ensemble_models = {
            name: model for name, model in all_models.items() 
            if name not in ['LSTM', 'GRU', 'Prophet']
        }
        
        if len(ensemble_models) >= 2:
            ensemble_model = train_ensemble_model(ensemble_models, X_val_scaled, y_val)
            if ensemble_model is not None:
                ensemble_pred = ensemble_model.predict(X_test_scaled)
                ensemble_metrics = calculate_metrics(y_test, ensemble_pred)
                all_models['Ensemble'] = ensemble_model
                all_metrics['Ensemble'] = ensemble_metrics
                logger.info(f"Ensemble metrics: {ensemble_metrics}")
    except Exception as e:
        logger.error(f"Ensemble training failed: {str(e)}")
    
    # Select best model based on validation MAE
    if all_metrics:
        best_model_name = min(all_metrics.items(), key=lambda x: x[1]['mae'])[0]
        best_model = all_models[best_model_name]
        best_metrics = all_metrics[best_model_name]
        
        logger.info(f"\n🏆 Best model: {best_model_name}")
        logger.info(f"   Metrics: {best_metrics}")
        
        # Save best model
        model_path = output_dir / f"{location_id}_best.pkl"
        
        if best_model_name in ['LSTM', 'GRU']:
            # Save Keras model
            best_model.save(output_dir / f"{location_id}_best.h5")
            logger.info(f"✅ Best model saved to {output_dir / f'{location_id}_best.h5'}")
        else:
            joblib.dump(best_model, model_path)
            logger.info(f"✅ Best model saved to {model_path}")
        
        # Save metrics
        metrics_data = {
            'best_model': best_model_name,
            'metrics': best_metrics,
            'all_models': all_metrics
        }
        
        metrics_path = output_dir / f"{location_id}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"✅ Metrics saved to {metrics_path}")
        
        return True
    else:
        logger.error("No models were successfully trained")
        return False


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train AQI prediction models')
    parser.add_argument('--data', type=str, default='data/sample_data.csv',
                      help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models/',
                      help='Output directory for models')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 AQI PREDICTION MODEL TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Setup paths
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    logger.info("📥 Loading and preprocessing data...")
    df = preprocess_data(str(data_path))
    
    # Feature engineering
    logger.info("🔧 Engineering features...")
    df_featured = engineer_features(df)
    
    # Group by unique locations
    locations = df_featured.groupby(['lat', 'lon'])
    num_locations = len(locations)
    
    logger.info(f"\n📍 Found {num_locations} unique location(s)")
    
    # Train models for each location
    success_count = 0
    for (lat, lon), df_location in locations:
        location_id = f"lat_{lat}_lon_{lon}"
        
        try:
            success = train_models_for_location(df_location, location_id, output_dir)
            if success:
                success_count += 1
        except Exception as e:
            logger.error(f"Failed to train models for {location_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED")
    print("="*70)
    print(f"Locations processed: {num_locations}")
    print(f"Successfully trained: {success_count}")
    print(f"Models directory: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
