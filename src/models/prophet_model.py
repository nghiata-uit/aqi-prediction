"""
Prophet model implementation for AQI prediction
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def train_prophet_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame
) -> Optional[object]:
    """
    Train Prophet model for AQI prediction
    
    Args:
        df_train: Training dataframe with datetime and aqi columns
        df_val: Validation dataframe
        
    Returns:
        Trained Prophet model or None if Prophet unavailable
    """
    try:
        from prophet import Prophet
        
        logger.info("📈 Training Prophet model...")
        
        # Prepare data for Prophet
        # Prophet requires columns named 'ds' (datestamp) and 'y' (target)
        train_prophet = df_train[['datetime', 'aqi']].copy()
        train_prophet.columns = ['ds', 'y']
        
        # Add regressors (pollutants)
        pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        available_pollutants = [col for col in pollutants if col in df_train.columns]
        
        for pollutant in available_pollutants:
            train_prophet[pollutant] = df_train[pollutant].values
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        # Add regressors
        for pollutant in available_pollutants:
            model.add_regressor(pollutant)
        
        # Fit model (suppress output)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_prophet)
        
        # Evaluate on validation set
        val_prophet = df_val[['datetime', 'aqi']].copy()
        val_prophet.columns = ['ds', 'y']
        
        for pollutant in available_pollutants:
            val_prophet[pollutant] = df_val[pollutant].values
        
        forecast = model.predict(val_prophet)
        val_predictions = forecast['yhat'].values
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(val_prophet['y'], val_predictions)
        
        logger.info(f"   Validation MAE: {mae:.4f}")
        logger.info("✅ Prophet training completed")
        
        return model
        
    except ImportError:
        logger.warning("⚠️  Prophet not available, skipping Prophet training")
        return None
    except Exception as e:
        logger.error(f"❌ Error training Prophet: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def predict_with_prophet(model: object, df_future: pd.DataFrame) -> np.ndarray:
    """
    Make predictions with Prophet model
    
    Args:
        model: Trained Prophet model
        df_future: Future dataframe with datetime and pollutant values
        
    Returns:
        Array of predictions
    """
    try:
        # Prepare data for Prophet
        future_prophet = df_future[['datetime']].copy()
        future_prophet.columns = ['ds']
        
        # Add regressors
        pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        available_pollutants = [col for col in pollutants if col in df_future.columns]
        
        for pollutant in available_pollutants:
            future_prophet[pollutant] = df_future[pollutant].values
        
        # Make prediction
        forecast = model.predict(future_prophet)
        predictions = forecast['yhat'].values
        
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Error making Prophet prediction: {str(e)}")
        return None
