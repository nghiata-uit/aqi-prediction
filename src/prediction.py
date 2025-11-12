"""
Module dự đoán AQI cho 24h tiếp theo
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
from pathlib import Path
import logging
from typing import Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def predict_next_24h(model: Any, last_data: pd.DataFrame, scaler: Any, 
                     feature_cols: List[str]) -> pd.DataFrame:
    """
    Dự đoán AQI cho 24h tiếp theo
    
    Args:
        model: Model đã train
        last_data: DataFrame chứa dữ liệu gần nhất (với tất cả features)
        scaler: Scaler đã fit
        feature_cols: List các cột features để dùng cho prediction
        
    Returns:
        DataFrame với columns: hour, timestamp, predicted_aqi
    """
    logger.info("🔮 Predicting AQI for next 24 hours...")
    
    try:
        predictions = []
        timestamps = []
        
        # Get last datetime
        if 'datetime' in last_data.columns:
            last_datetime = pd.to_datetime(last_data['datetime'].iloc[-1])
        else:
            last_datetime = pd.Timestamp.now()
        
        # Create working dataframe
        working_data = last_data.copy()
        
        # Predict iteratively for 24 hours
        for hour in range(1, 25):
            # Get current timestamp
            current_timestamp = last_datetime + timedelta(hours=hour)
            
            # Get features from last row
            if len(working_data) > 0:
                # Use the last row for prediction
                last_row = working_data.iloc[-1:].copy()
                
                # Extract features
                try:
                    X = last_row[feature_cols].values
                    
                    # Make prediction
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)[0]
                    else:
                        pred = 3.0  # Default fallback
                    
                    # Clip prediction to valid AQI range
                    pred = np.clip(pred, 1, 5)
                    
                except Exception as e:
                    logger.warning(f"⚠️  Error during prediction at hour {hour}: {str(e)}")
                    pred = 3.0  # Default fallback
                
                predictions.append(round(pred, 2))
                timestamps.append(current_timestamp)
                
                # Update working data for next iteration (simplified approach)
                # In a real scenario, you would update all lag and rolling features properly
                new_row = last_row.copy()
                new_row['aqi'] = pred
                if 'datetime' in new_row.columns:
                    new_row['datetime'] = current_timestamp
                
                working_data = pd.concat([working_data, new_row], ignore_index=True)
            else:
                predictions.append(3.0)
                timestamps.append(current_timestamp)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'hour': range(1, 25),
            'timestamp': timestamps,
            'predicted_aqi': predictions
        })
        
        logger.info(f"✅ Generated predictions for 24 hours")
        logger.info(f"   Predicted AQI range: {min(predictions):.2f} - {max(predictions):.2f}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"❌ Error in 24h prediction: {str(e)}")
        raise


def load_model_and_predict(model_path: str, scaler_path: str, data_path: str) -> pd.DataFrame:
    """
    Load model và thực hiện dự đoán
    
    Args:
        model_path: Đường dẫn đến file model
        scaler_path: Đường dẫn đến file scaler
        data_path: Đường dẫn đến file dữ liệu
        
    Returns:
        DataFrame chứa predictions
    """
    try:
        # Load model
        logger.info(f"Loading model from {model_path}...")
        
        if model_path.endswith('.h5'):
            # TensorFlow/Keras model
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
        else:
            # Scikit-learn or XGBoost model
            model = joblib.load(model_path)
        
        logger.info("✅ Model loaded successfully")
        
        # Load scaler
        logger.info(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
        logger.info("✅ Scaler loaded successfully")
        
        # Load data
        logger.info(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
        
        logger.info("✅ Data loaded successfully")
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = ['datetime', 'aqi', 'lon', 'lat']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Make predictions
        predictions = predict_next_24h(model, data, scaler, feature_cols)
        
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Error in load_model_and_predict: {str(e)}")
        raise


def visualize_predictions(predictions_df: pd.DataFrame, save_path: str) -> None:
    """
    Visualize 24h predictions
    
    Args:
        predictions_df: DataFrame với predictions
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style('whitegrid')
        
        plt.figure(figsize=(15, 6))
        
        # Plot predictions
        plt.plot(predictions_df['hour'], predictions_df['predicted_aqi'], 
                marker='o', linewidth=2, markersize=8, color='#3498db', label='Predicted AQI')
        
        # Add color zones for AQI levels
        plt.axhspan(0, 2, alpha=0.1, color='green', label='Good (1-2)')
        plt.axhspan(2, 3, alpha=0.1, color='yellow', label='Fair (2-3)')
        plt.axhspan(3, 4, alpha=0.1, color='orange', label='Moderate (3-4)')
        plt.axhspan(4, 5, alpha=0.1, color='red', label='Poor (4-5)')
        
        plt.xlabel('Hour Ahead', fontsize=12)
        plt.ylabel('Predicted AQI', fontsize=12)
        plt.title('24-Hour AQI Forecast', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 6)
        
        plt.tight_layout()
        
        # Create directory if not exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Prediction visualization saved to {save_path}")
        
    except Exception as e:
        logger.error(f"❌ Error visualizing predictions: {str(e)}")
        raise
