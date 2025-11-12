"""
Module training các ML models
"""
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AQIModelTrainer:
    """Class để train và quản lý các models"""
    
    def __init__(self):
        self.models = {}
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> RandomForestRegressor:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Trained Random Forest model
        """
        logger.info("🌲 Training Random Forest model...")
        
        try:
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
            
            # Evaluate on validation set
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            logger.info(f"   Train R² score: {train_score:.4f}")
            logger.info(f"   Validation R² score: {val_score:.4f}")
            logger.info("✅ Random Forest training completed")
            
            self.models['random_forest'] = model
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training Random Forest: {str(e)}")
            raise
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBRegressor:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Trained XGBoost model
        """
        logger.info("🚀 Training XGBoost model...")
        
        try:
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate on both sets
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            logger.info(f"   Train R² score: {train_score:.4f}")
            logger.info(f"   Validation R² score: {val_score:.4f}")
            logger.info("✅ XGBoost training completed")
            
            self.models['xgboost'] = model
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training XGBoost: {str(e)}")
            raise
    
    def train_lstm(self, X_train_seq: np.ndarray, y_train_seq: np.ndarray,
                  X_val_seq: np.ndarray, y_val_seq: np.ndarray) -> Any:
        """
        Train LSTM model
        
        Args:
            X_train_seq: Training sequences (3D: samples, timesteps, features)
            y_train_seq: Training targets
            X_val_seq: Validation sequences
            y_val_seq: Validation targets
            
        Returns:
            Trained LSTM model
        """
        logger.info("🧠 Training LSTM model...")
        
        try:
            # Import tensorflow here to avoid loading if not needed
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
            import tensorflow as tf
            
            # Set seed for reproducibility
            tf.random.set_seed(42)
            
            # Build LSTM model
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Early stopping callback
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate
            train_loss, train_mae = model.evaluate(X_train_seq, y_train_seq, verbose=0)
            val_loss, val_mae = model.evaluate(X_val_seq, y_val_seq, verbose=0)
            
            logger.info(f"   Train MAE: {train_mae:.4f}")
            logger.info(f"   Validation MAE: {val_mae:.4f}")
            logger.info("✅ LSTM training completed")
            
            self.models['lstm'] = model
            return model
            
        except ImportError:
            logger.warning("⚠️  TensorFlow not available, skipping LSTM training")
            return None
        except Exception as e:
            logger.error(f"❌ Error training LSTM: {str(e)}")
            raise
    
    def save_model(self, model: Any, model_name: str, model_dir: str) -> None:
        """
        Save trained model
        
        Args:
            model: Trained model object
            model_name: Name for the model file
            model_dir: Directory to save the model
        """
        try:
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save based on model type
            if 'lstm' in model_name.lower():
                # TensorFlow/Keras model
                save_path = model_path / f"{model_name}.h5"
                model.save(save_path)
            else:
                # Scikit-learn or XGBoost model
                save_path = model_path / f"{model_name}.pkl"
                joblib.dump(model, save_path)
            
            logger.info(f"✅ Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
            raise
    
    def get_feature_importance(self, model: Any, feature_names: list) -> dict:
        """
        Get feature importance from trained model
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature names and importance scores
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
                return importance_dict
            else:
                logger.warning("⚠️  Model does not have feature_importances_ attribute")
                return {}
        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {str(e)}")
            return {}
