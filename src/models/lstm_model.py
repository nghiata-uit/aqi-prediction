"""
LSTM model implementation for AQI prediction
"""
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def train_lstm_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray
) -> Optional[object]:
    """
    Train LSTM model for AQI prediction
    
    Args:
        X_train_seq: Training sequences (3D: samples, timesteps, features)
        y_train_seq: Training targets
        X_val_seq: Validation sequences
        y_val_seq: Validation targets
        
    Returns:
        Trained LSTM model or None if TensorFlow unavailable
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        import tensorflow as tf
        
        logger.info("🧠 Training LSTM model...")
        
        # Set seed for reproducibility
        tf.random.set_seed(42)
        
        # Build LSTM architecture
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
        
        return model
        
    except ImportError:
        logger.warning("⚠️  TensorFlow not available, skipping LSTM training")
        return None
    except Exception as e:
        logger.error(f"❌ Error training LSTM: {str(e)}")
        return None
