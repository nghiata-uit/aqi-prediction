"""
Module tạo features cho model dự đoán AQI
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo time-based features
    
    Args:
        df: DataFrame với cột datetime
        
    Returns:
        DataFrame với time features mới
    """
    df_new = df.copy()
    
    if 'datetime' in df_new.columns:
        df_new['hour'] = df_new['datetime'].dt.hour
        df_new['day_of_week'] = df_new['datetime'].dt.dayofweek
        df_new['day'] = df_new['datetime'].dt.day
        df_new['month'] = df_new['datetime'].dt.month
        df_new['is_weekend'] = (df_new['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding cho hour (sin/cos transformation)
        df_new['hour_sin'] = np.sin(2 * np.pi * df_new['hour'] / 24)
        df_new['hour_cos'] = np.cos(2 * np.pi * df_new['hour'] / 24)
        
        # Cyclical encoding cho day_of_week
        df_new['dow_sin'] = np.sin(2 * np.pi * df_new['day_of_week'] / 7)
        df_new['dow_cos'] = np.cos(2 * np.pi * df_new['day_of_week'] / 7)
        
        logger.info("✅ Created time features: hour, day_of_week, day, month, is_weekend, cyclical encodings")
    
    return df_new


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Tạo lag features cho các cột được chỉ định
    
    Args:
        df: DataFrame đầu vào
        columns: List các cột cần tạo lag features
        lags: List các lag periods (số giờ)
        
    Returns:
        DataFrame với lag features mới
    """
    df_new = df.copy()
    
    for col in columns:
        if col in df_new.columns:
            for lag in lags:
                df_new[f'{col}_lag_{lag}h'] = df_new[col].shift(lag)
    
    logger.info(f"✅ Created lag features for {len(columns)} columns with lags: {lags}")
    
    return df_new


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Tạo rolling statistics features
    
    Args:
        df: DataFrame đầu vào
        columns: List các cột cần tạo rolling features
        windows: List các window sizes (số giờ)
        
    Returns:
        DataFrame với rolling features mới
    """
    df_new = df.copy()
    
    for col in columns:
        if col in df_new.columns:
            for window in windows:
                # Rolling mean
                df_new[f'{col}_rolling_mean_{window}h'] = df_new[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                df_new[f'{col}_rolling_std_{window}h'] = df_new[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min/max
                df_new[f'{col}_rolling_min_{window}h'] = df_new[col].rolling(window=window, min_periods=1).min()
                df_new[f'{col}_rolling_max_{window}h'] = df_new[col].rolling(window=window, min_periods=1).max()
    
    logger.info(f"✅ Created rolling features for {len(columns)} columns with windows: {windows}")
    
    return df_new


def create_spatial_features(df: pd.DataFrame, spatial_scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Tạo spatial features và chuẩn hóa tọa độ địa lý (lat, lon)
    
    Args:
        df: DataFrame đầu vào với cột 'lat' và 'lon'
        spatial_scaler: StandardScaler đã fit trước đó (để inference), nếu None sẽ tạo mới
        
    Returns:
        Tuple (DataFrame với spatial features đã chuẩn hóa, StandardScaler đã fit)
    """
    df_new = df.copy()
    
    # Kiểm tra xem có cột lat và lon không
    if 'lat' in df_new.columns and 'lon' in df_new.columns:
        # Nếu chưa có scaler, tạo mới và fit
        if spatial_scaler is None:
            spatial_scaler = StandardScaler()
            df_new[['lat_scaled', 'lon_scaled']] = spatial_scaler.fit_transform(df_new[['lat', 'lon']])
            logger.info("✅ Created and fitted spatial scaler for lat/lon features")
        else:
            # Sử dụng scaler đã fit trước đó (cho inference)
            df_new[['lat_scaled', 'lon_scaled']] = spatial_scaler.transform(df_new[['lat', 'lon']])
            logger.info("✅ Applied existing spatial scaler to lat/lon features")
        
        # Xóa cột lat/lon gốc để tránh data leakage
        df_new = df_new.drop(columns=['lat', 'lon'])
    else:
        logger.warning("⚠️  lat/lon columns not found, skipping spatial features")
        spatial_scaler = None
    
    return df_new, spatial_scaler


def engineer_features(df: pd.DataFrame, spatial_scaler: Optional[StandardScaler] = None, 
                      include_spatial: bool = False) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Pipeline feature engineering hoàn chỉnh
    
    Args:
        df: DataFrame đầu vào
        spatial_scaler: StandardScaler cho spatial features (nếu có), dùng cho inference
        include_spatial: Có tạo spatial features không (mặc định False để tương thích ngược)
        
    Returns:
        Tuple (DataFrame với tất cả features đã được tạo, StandardScaler cho spatial features hoặc None)
        
    Note:
        - Nếu include_spatial=False, trả về (df, None) để tương thích với code cũ
        - Nếu include_spatial=True, tạo spatial features và trả về (df, scaler)
    """
    logger.info("🚀 Starting feature engineering pipeline...")
    
    df_featured = df.copy()
    fitted_spatial_scaler = None
    
    # 1. Create spatial features (nếu được yêu cầu)
    if include_spatial:
        df_featured, fitted_spatial_scaler = create_spatial_features(df_featured, spatial_scaler)
    
    # 2. Create time features
    df_featured = create_time_features(df_featured)
    
    # 3. Define pollutant columns
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    available_pollutants = [col for col in pollutant_cols if col in df_featured.columns]
    
    # 4. Create lag features (1h, 2h, 3h, 6h, 12h, 24h)
    lags = [1, 2, 3, 6, 12, 24]
    df_featured = create_lag_features(df_featured, available_pollutants, lags)
    
    # 5. Create rolling features (6h, 12h, 24h)
    windows = [6, 12, 24]
    df_featured = create_rolling_features(df_featured, available_pollutants, windows)
    
    # 6. Drop rows with NaN values created by lag/rolling features
    original_rows = len(df_featured)
    df_featured = df_featured.dropna().reset_index(drop=True)
    dropped_rows = original_rows - len(df_featured)
    
    logger.info(f"✅ Feature engineering completed")
    logger.info(f"   Original rows: {original_rows}")
    logger.info(f"   Dropped rows (NaN): {dropped_rows}")
    logger.info(f"   Final rows: {len(df_featured)}")
    logger.info(f"   Total features: {len(df_featured.columns)}")
    
    return df_featured, fitted_spatial_scaler
    df_featured = df_featured.dropna().reset_index(drop=True)
    dropped_rows = original_rows - len(df_featured)
    
    logger.info(f"✅ Feature engineering completed")
    logger.info(f"   Original rows: {original_rows}")
    logger.info(f"   Dropped rows (NaN): {dropped_rows}")
    logger.info(f"   Final rows: {len(df_featured)}")
    logger.info(f"   Total features: {len(df_featured.columns)}")
    
    return df_featured, fitted_spatial_scaler
