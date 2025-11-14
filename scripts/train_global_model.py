"""
Script huấn luyện Global XGBoost Model với spatial features

Tính năng:
- Load và preprocess dữ liệu từ data/sample_data.csv
- Tạo features (time, lag, rolling, spatial)
- Train XGBoost model với time-based split (70% train, 30% val)
- Lưu artifacts: xgboost_global.pkl, feature_columns_global.pkl, spatial_scaler.pkl
- In ra validation metrics (MAE, RMSE)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import logging

# Thêm thư mục gốc vào sys.path để import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_global_model():
    """
    Hàm chính để train global XGBoost model
    
    Pipeline:
    1. Load và preprocess dữ liệu
    2. Feature engineering (bao gồm spatial features)
    3. Time-based split (70% train, 30% validation)
    4. Train XGBoost model
    5. Evaluate và print metrics
    6. Save artifacts
    """
    logger.info("="*70)
    logger.info("🚀 TRAINING GLOBAL XGBOOST MODEL WITH SPATIAL FEATURES")
    logger.info("="*70)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / 'data' / 'sample_data.csv'
    model_dir = base_dir / 'models'
    
    # Tạo thư mục models nếu chưa tồn tại
    model_dir.mkdir(exist_ok=True)
    
    # ===== STEP 1: Load và Preprocess Data =====
    logger.info("\n📥 STEP 1: Loading and Preprocessing Data")
    logger.info("-" * 70)
    df = preprocess_data(str(data_path))
    
    # Đổi tên cột datetime thành dt để phù hợp với engineer_features
    # Note: engineer_features supports both 'datetime' and 'dt', but 'dt' is preferred
    # for consistency with the new API and to avoid confusion with Python's datetime module
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'dt'})
    
    # ===== STEP 2: Feature Engineering =====
    logger.info("\n🔧 STEP 2: Feature Engineering with Spatial Features")
    logger.info("-" * 70)
    df_featured, spatial_scaler = engineer_features(df)
    
    # ===== STEP 3: Prepare Data for Training =====
    logger.info("\n📊 STEP 3: Preparing Data for Training")
    logger.info("-" * 70)
    
    # Loại bỏ các cột không dùng để train
    exclude_cols = ['dt', 'datetime', 'aqi', 'lat', 'lon']
    feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
    
    X = df_featured[feature_cols].values
    y = df_featured['aqi'].values
    
    logger.info(f"   Feature shape: {X.shape}")
    logger.info(f"   Target shape: {y.shape}")
    logger.info(f"   Number of features: {len(feature_cols)}")
    logger.info(f"   Feature columns: {feature_cols[:5]}... (showing first 5)")
    
    # Time-based split (70% train, 30% validation) - không shuffle
    train_size = int(0.7 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:]
    y_val = y[train_size:]
    
    logger.info(f"   Train set: {X_train.shape[0]} samples")
    logger.info(f"   Validation set: {X_val.shape[0]} samples")
    
    # ===== STEP 4: Train XGBoost Model =====
    logger.info("\n🤖 STEP 4: Training XGBoost Global Model")
    logger.info("-" * 70)
    
    # Khởi tạo XGBoost model với hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("   Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    logger.info("   ✅ Model training completed")
    
    # ===== STEP 5: Evaluate Model =====
    logger.info("\n📈 STEP 5: Evaluating Model")
    logger.info("-" * 70)
    
    # Predictions trên validation set
    y_val_pred = model.predict(X_val)
    
    # Tính metrics
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    logger.info(f"   Validation Metrics:")
    logger.info(f"   - MAE:  {mae:.4f}")
    logger.info(f"   - RMSE: {rmse:.4f}")
    
    # ===== STEP 6: Save Artifacts =====
    logger.info("\n💾 STEP 6: Saving Model Artifacts")
    logger.info("-" * 70)
    
    # Save XGBoost model
    model_path = model_dir / 'xgboost_global.pkl'
    joblib.dump(model, model_path)
    logger.info(f"   ✅ Saved model to: {model_path}")
    
    # Save feature columns
    feature_cols_path = model_dir / 'feature_columns_global.pkl'
    joblib.dump(feature_cols, feature_cols_path)
    logger.info(f"   ✅ Saved feature columns to: {feature_cols_path}")
    
    # Save spatial scaler
    spatial_scaler_path = model_dir / 'spatial_scaler.pkl'
    joblib.dump(spatial_scaler, spatial_scaler_path)
    logger.info(f"   ✅ Saved spatial scaler to: {spatial_scaler_path}")
    
    # ===== FINAL SUMMARY =====
    logger.info("\n" + "="*70)
    logger.info("✅ GLOBAL MODEL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info("\n📁 Generated Artifacts:")
    logger.info(f"   - {model_path}")
    logger.info(f"   - {feature_cols_path}")
    logger.info(f"   - {spatial_scaler_path}")
    logger.info("\n🎯 Model Performance:")
    logger.info(f"   - MAE:  {mae:.4f}")
    logger.info(f"   - RMSE: {rmse:.4f}")
    logger.info("\n" + "="*70 + "\n")


if __name__ == '__main__':
    train_global_model()
