"""
Prediction logic for the API
"""
import logging
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from .utils import (
    load_model_for_location,
    load_scaler,
    prepare_features,
    calculate_confidence_score
)

logger = logging.getLogger(__name__)


def predict_aqi(
    lat: float,
    lon: float,
    current_data: dict,
    datetime_str: Optional[str],
    models_dir: Path
) -> Optional[Dict]:
    """
    Predict AQI for a given location and current conditions
    
    Args:
        lat: Latitude
        lon: Longitude
        current_data: Dictionary of current pollutant values
        datetime_str: Optional datetime string
        models_dir: Path to models directory
        
    Returns:
        Dictionary with prediction results or None
    """
    try:
        # Load model for location
        result = load_model_for_location(lat, lon, models_dir)
        
        if result is None:
            logger.error(f"No model available for location ({lat}, {lon})")
            return None
        
        model, metrics, model_lat, model_lon = result
        
        # Load scaler
        scaler = load_scaler(models_dir)
        
        # Prepare features
        features_df = prepare_features(current_data, datetime_str)
        
        # Get feature columns (match training features)
        # This is a simplified approach - in production, you'd save the feature list
        feature_cols = [col for col in features_df.columns if col not in ['datetime', 'aqi', 'lon', 'lat']]
        
        # Scale features if scaler is available
        if scaler is not None:
            try:
                # Get the features in the right order
                X = features_df[feature_cols].values
                X_scaled = scaler.transform(X)
            except Exception as e:
                logger.warning(f"Error scaling features, using unscaled: {str(e)}")
                X_scaled = features_df[feature_cols].values
        else:
            X_scaled = features_df[feature_cols].values
        
        # Make prediction
        predicted_aqi = model.predict(X_scaled)[0]
        
        # Clip to valid AQI range
        predicted_aqi = float(np.clip(predicted_aqi, 1, 5))
        
        # Calculate prediction time
        if datetime_str:
            current_time = pd.to_datetime(datetime_str)
        else:
            current_time = pd.Timestamp.now()
        
        prediction_time = current_time + timedelta(hours=24)
        
        # Calculate confidence score
        confidence = calculate_confidence_score(metrics.get('metrics', metrics), predicted_aqi)
        
        # Prepare result
        result = {
            'location': {
                'lat': model_lat,
                'lon': model_lon
            },
            'predicted_aqi': round(predicted_aqi, 2),
            'prediction_for': prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': metrics.get('best_model', 'Unknown'),
            'confidence_score': confidence,
            'metrics': {
                'mae': metrics.get('metrics', {}).get('mae', metrics.get('mae', 0.0)),
                'rmse': metrics.get('metrics', {}).get('rmse', metrics.get('rmse', 0.0)),
                'r2': metrics.get('metrics', {}).get('r2', metrics.get('r2', None))
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
