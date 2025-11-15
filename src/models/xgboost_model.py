"""
XGBoost model implementation for AQI prediction
"""
import xgboost as xgb
import numpy as np
import logging

logger = logging.getLogger(__name__)


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> xgb.XGBRegressor:
    """
    Train XGBoost model for AQI prediction
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        Trained XGBoost model
    """
    logger.info("🚀 Training XGBoost model...")
    
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
    
    return model
