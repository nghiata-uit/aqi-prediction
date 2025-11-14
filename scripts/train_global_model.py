"""
Script train global XGBoost model với spatial features (lat, lon)
Lưu artifacts vào thư mục models/ để sử dụng cho API inference
"""
import sys
from pathlib import Path

# Thêm thư mục gốc vào Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import logging

from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple:
    """
    Split dữ liệu theo thời gian (time-based split) - không shuffle
    
    Args:
        df: DataFrame với dữ liệu
        train_ratio: Tỷ lệ dữ liệu train (mặc định 70%)
        
    Returns:
        Tuple (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"📊 Time-based split: {train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test")
    logger.info(f"   Train size: {len(train_df)} samples")
    logger.info(f"   Test size: {len(test_df)} samples")
    
    return train_df, test_df


def train_global_xgboost(X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series) -> xgb.XGBRegressor:
    """
    Train XGBoost model với hyperparameters tối ưu
    
    Args:
        X_train: Features training set
        y_train: Target training set
        X_test: Features test set
        y_test: Target test set
        
    Returns:
        Trained XGBoost model
    """
    logger.info("🚀 Training XGBoost global model...")
    
    # Khởi tạo XGBoost với hyperparameters tối ưu
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    logger.info("✅ Model training completed")
    
    return model


def evaluate_model(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Đánh giá model và in ra metrics
    
    Args:
        model: Trained model
        X_test: Features test set
        y_test: Target test set
        
    Returns:
        Dictionary với metrics
    """
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {
        'mae': mae,
        'rmse': rmse
    }
    
    # Print metrics
    logger.info("📊 Model Evaluation Metrics:")
    logger.info(f"   MAE:  {mae:.4f}")
    logger.info(f"   RMSE: {rmse:.4f}")
    
    return metrics


def save_artifacts(model: xgb.XGBRegressor, feature_columns: list, 
                   spatial_scaler, output_dir: Path):
    """
    Lưu model artifacts vào thư mục models/
    
    Args:
        model: Trained XGBoost model
        feature_columns: Danh sách tên các features
        spatial_scaler: StandardScaler cho spatial features
        output_dir: Thư mục output
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "xgboost_global.pkl"
    joblib.dump(model, model_path)
    logger.info(f"✅ Saved model to {model_path}")
    
    # Save feature columns
    feature_cols_path = output_dir / "feature_columns_global.pkl"
    joblib.dump(feature_columns, feature_cols_path)
    logger.info(f"✅ Saved feature columns to {feature_cols_path}")
    
    # Save spatial scaler
    if spatial_scaler is not None:
        scaler_path = output_dir / "spatial_scaler.pkl"
        joblib.dump(spatial_scaler, scaler_path)
        logger.info(f"✅ Saved spatial scaler to {scaler_path}")
    else:
        logger.warning("⚠️  Spatial scaler is None, skipping save")


def main():
    """
    Main training pipeline
    """
    logger.info("="*80)
    logger.info("🌍 TRAINING GLOBAL XGBOOST MODEL WITH SPATIAL FEATURES")
    logger.info("="*80)
    
    # 1. Load và preprocess dữ liệu
    data_path = project_root / "data" / "sample_data.csv"
    logger.info(f"\n📂 Loading data from {data_path}")
    df = preprocess_data(str(data_path))
    
    # 2. Feature engineering (bao gồm spatial features)
    logger.info("\n🔧 Feature Engineering")
    df_featured, spatial_scaler = engineer_features(df, include_spatial=True)
    
    # 3. Chuẩn bị X và y
    # Loại bỏ các cột không phải features
    exclude_cols = ['datetime', 'aqi']
    feature_cols = [col for col in df_featured.columns if col not in exclude_cols]
    
    X = df_featured[feature_cols]
    y = df_featured['aqi']
    
    logger.info(f"\n📊 Dataset Info:")
    logger.info(f"   Total samples: {len(X)}")
    logger.info(f"   Total features: {len(feature_cols)}")
    logger.info(f"   Feature columns: {feature_cols[:5]}... (showing first 5)")
    
    # 4. Time-based split (70/30)
    logger.info("\n✂️  Splitting data")
    # Tạo DataFrame tạm để split theo thời gian
    temp_df = df_featured.copy()
    train_df, test_df = time_based_split(temp_df, train_ratio=0.7)
    
    X_train = train_df[feature_cols]
    y_train = train_df['aqi']
    X_test = test_df[feature_cols]
    y_test = test_df['aqi']
    
    # 5. Train model
    logger.info("\n🤖 Training Model")
    model = train_global_xgboost(X_train, y_train, X_test, y_test)
    
    # 6. Evaluate model
    logger.info("\n📈 Evaluating Model")
    metrics = evaluate_model(model, X_test, y_test)
    
    # 7. Save artifacts
    logger.info("\n💾 Saving Artifacts")
    output_dir = project_root / "models"
    save_artifacts(model, feature_cols, spatial_scaler, output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\n📦 Artifacts saved in: {output_dir}")
    logger.info("   - xgboost_global.pkl (model)")
    logger.info("   - feature_columns_global.pkl (feature names)")
    logger.info("   - spatial_scaler.pkl (spatial scaler)")
    logger.info(f"\n🎯 Final Metrics:")
    logger.info(f"   MAE:  {metrics['mae']:.4f}")
    logger.info(f"   RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
