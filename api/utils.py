"""
Utility functions for the API
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def find_closest_model(lat: float, lon: float, models_dir: Path) -> Optional[Tuple[Path, float, float]]:
    """
    Find the closest available model for a given location
    
    Args:
        lat: Target latitude
        lon: Target longitude
        models_dir: Directory containing trained models
        
    Returns:
        Tuple of (model_path, model_lat, model_lon) or None if no models found
    """
    try:
        # Find all model files
        model_files = list(models_dir.glob("lat_*_lon_*_best.pkl"))
        
        if not model_files:
            return None
        
        # Extract coordinates from filenames and find closest
        min_distance = float('inf')
        closest_model = None
        closest_lat = None
        closest_lon = None
        
        for model_file in model_files:
            # Parse filename: lat_X_lon_Y_best.pkl
            filename = model_file.stem
            parts = filename.split('_')
            
            try:
                model_lat = float(parts[1])
                model_lon = float(parts[3])
                
                # Calculate Euclidean distance
                distance = np.sqrt((lat - model_lat)**2 + (lon - model_lon)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_model = model_file
                    closest_lat = model_lat
                    closest_lon = model_lon
            except (IndexError, ValueError):
                continue
        
        if closest_model:
            return closest_model, closest_lat, closest_lon
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding closest model: {str(e)}")
        return None


def load_model_for_location(lat: float, lon: float, models_dir: Path) -> Optional[Tuple[object, Dict, float, float]]:
    """
    Load the best model and its metrics for a specific location
    
    Args:
        lat: Latitude
        lon: Longitude
        models_dir: Directory containing trained models
        
    Returns:
        Tuple of (model, metrics_dict, actual_lat, actual_lon) or None
    """
    try:
        # Find closest model
        result = find_closest_model(lat, lon, models_dir)
        
        if result is None:
            logger.error(f"No model found for location ({lat}, {lon})")
            return None
        
        model_path, model_lat, model_lon = result
        
        # Load the model
        model = joblib.load(model_path)
        
        # Load metrics
        metrics_path = model_path.parent / model_path.name.replace('_best.pkl', '_metrics.json')
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        
        logger.info(f"Loaded model for location ({model_lat}, {model_lon})")
        return model, metrics, model_lat, model_lon
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def load_scaler(models_dir: Path) -> Optional[object]:
    """
    Load the feature scaler
    
    Args:
        models_dir: Directory containing the scaler
        
    Returns:
        Scaler object or None
    """
    try:
        scaler_path = models_dir / 'scaler.pkl'
        if scaler_path.exists():
            return joblib.load(scaler_path)
        else:
            logger.warning("Scaler not found")
            return None
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        return None


def prepare_features(current_data: dict, datetime_str: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare features from current data for prediction
    
    Args:
        current_data: Dictionary of current pollutant values
        datetime_str: Optional datetime string
        
    Returns:
        DataFrame with features
    """
    try:
        # Parse datetime
        if datetime_str:
            dt = pd.to_datetime(datetime_str)
        else:
            dt = pd.Timestamp.now()
        
        # Create base features
        features = {
            'co': current_data['co'],
            'no': current_data['no'],
            'no2': current_data['no2'],
            'o3': current_data['o3'],
            'so2': current_data['so2'],
            'pm2_5': current_data['pm2_5'],
            'pm10': current_data['pm10'],
            'nh3': current_data['nh3'],
        }
        
        # Add time features
        features['hour'] = dt.hour
        features['day_of_week'] = dt.dayofweek
        features['day'] = dt.day
        features['month'] = dt.month
        features['is_weekend'] = int(dt.dayofweek >= 5)
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
        
        # Create lag features (use current values as approximation)
        pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        lags = [1, 2, 3, 6, 12, 24]
        
        for pollutant in pollutants:
            for lag in lags:
                features[f'{pollutant}_lag_{lag}h'] = current_data[pollutant]
        
        # Create rolling features (use current values as approximation)
        windows = [6, 12, 24]
        for pollutant in pollutants:
            for window in windows:
                features[f'{pollutant}_rolling_mean_{window}h'] = current_data[pollutant]
                features[f'{pollutant}_rolling_std_{window}h'] = 0.0
                features[f'{pollutant}_rolling_min_{window}h'] = current_data[pollutant]
                features[f'{pollutant}_rolling_max_{window}h'] = current_data[pollutant]
        
        df = pd.DataFrame([features])
        return df
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise


def calculate_confidence_score(metrics: Dict, predicted_value: float) -> float:
    """
    Calculate confidence score based on model metrics and prediction
    
    Args:
        metrics: Dictionary containing model metrics
        predicted_value: Predicted AQI value
        
    Returns:
        Confidence score between 0 and 1
    """
    try:
        # Base confidence on R² score
        r2 = metrics.get('r2', 0.5)
        mae = metrics.get('mae', 1.0)
        
        # Adjust for R² score
        confidence = r2
        
        # Adjust for MAE (lower is better)
        if mae < 0.5:
            confidence += 0.1
        elif mae > 1.0:
            confidence -= 0.1
        
        # Adjust for extreme predictions
        if predicted_value < 1 or predicted_value > 5:
            confidence -= 0.2
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return round(confidence, 2)
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {str(e)}")
        return 0.5


def list_available_models(models_dir: Path) -> List[Dict]:
    """
    List all available trained models
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        List of model information dictionaries
    """
    try:
        model_files = list(models_dir.glob("lat_*_lon_*_best.pkl"))
        models_info = []
        
        for model_file in model_files:
            # Parse filename
            filename = model_file.stem
            parts = filename.split('_')
            
            try:
                lat = float(parts[1])
                lon = float(parts[3])
                
                # Load metrics
                metrics_path = model_file.parent / model_file.name.replace('_best.pkl', '_metrics.json')
                
                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                else:
                    metrics = {}
                
                model_info = {
                    'lat': lat,
                    'lon': lon,
                    'model_type': metrics.get('best_model', 'Unknown'),
                    'metrics': metrics.get('metrics', {}),
                    'trained_date': None
                }
                
                models_info.append(model_info)
                
            except (IndexError, ValueError):
                continue
        
        return models_info
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return []


def get_model_comparison(lat: float, lon: float, models_dir: Path) -> Optional[Dict]:
    """
    Get comparison of all models for a specific location
    
    Args:
        lat: Latitude
        lon: Longitude
        models_dir: Directory containing trained models
        
    Returns:
        Dictionary with model comparison or None
    """
    try:
        # Find the location's metrics file
        result = find_closest_model(lat, lon, models_dir)
        
        if result is None:
            return None
        
        model_path, model_lat, model_lon = result
        metrics_path = model_path.parent / model_path.name.replace('_best.pkl', '_metrics.json')
        
        if not metrics_path.exists():
            return None
        
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        return {
            'lat': model_lat,
            'lon': model_lon,
            'models': data.get('all_models', {}),
            'best_model': data.get('best_model', 'Unknown')
        }
        
    except Exception as e:
        logger.error(f"Error getting model comparison: {str(e)}")
        return None
