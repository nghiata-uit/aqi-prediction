"""
Script để train global XGBoost model với spatial features (lat, lon)
Sử dụng time-based split 70/30 và lưu artifacts cho API
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_global_model():
    """
    Train global XGBoost model với spatial features và lưu artifacts
    """
    logger.info("="*80)
    logger.info("🚀 Starting Global XGBoost Model Training")
    logger.info("="*80)
    
    # 1. Load và preprocess dữ liệu
    logger.info("\n📂 Loading data from data/sample_data.csv...")
    data_path = project_root / "data" / "sample_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = preprocess_data(str(data_path))
    logger.info(f"✅ Loaded {len(df)} rows")
    logger.info(f"   Columns: {list(df.columns)}")
    
    # 2. Feature engineering với spatial features
    logger.info("\n🔧 Engineering features (including spatial features)...")
    df_featured, spatial_scaler = engineer_features(df)
    logger.info(f"✅ Features created: {len(df_featured.columns)} columns")
    logger.info(f"   Data shape: {df_featured.shape}")
    
    # 3. Chuẩn bị X và y
    logger.info("\n📊 Preparing X and y...")
    
    # Loại bỏ các cột không phải features
    exclude_cols = ['datetime', 'aqi', 'lat', 'lon']  # lat, lon đã được scaled và đổi tên
    feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
    
    X = df_featured[feature_cols].values
    y = df_featured['aqi'].values
    
    logger.info(f"✅ X shape: {X.shape}")
    logger.info(f"   y shape: {y.shape}")
    logger.info(f"   Number of features: {len(feature_cols)}")
    
    # 4. Time-based split 70/30
    logger.info("\n✂️ Performing time-based split (70/30)...")
    split_idx = int(len(X) * 0.7)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    logger.info(f"✅ Training set: {X_train.shape[0]} samples")
    logger.info(f"   Test set: {X_test.shape[0]} samples")
    
    # 5. Train XGBoost model
    logger.info("\n🎯 Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    logger.info("✅ Model training completed")
    
    # 6. Evaluate model
    logger.info("\n📈 Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 MODEL PERFORMANCE METRICS")
    logger.info(f"{'='*50}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R²:   {r2:.4f}")
    logger.info(f"{'='*50}\n")
    
    # 7. Save artifacts
    logger.info("💾 Saving artifacts...")
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "xgboost_global.pkl"
    joblib.dump(model, model_path)
    logger.info(f"✅ Model saved: {model_path}")
    
    # Save feature columns
    feature_cols_path = models_dir / "feature_columns_global.pkl"
    joblib.dump(feature_cols, feature_cols_path)
    logger.info(f"✅ Feature columns saved: {feature_cols_path}")
    
    # Save spatial scaler
    scaler_path = models_dir / "spatial_scaler.pkl"
    joblib.dump(spatial_scaler, scaler_path)
    logger.info(f"✅ Spatial scaler saved: {scaler_path}")
    
    # 8. Summary
    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Artifacts saved in: {models_dir}")
    logger.info(f"  - xgboost_global.pkl (model)")
    logger.info(f"  - feature_columns_global.pkl ({len(feature_cols)} features)")
    logger.info(f"  - spatial_scaler.pkl (lat/lon scaler)")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        train_global_model()
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise
