"""
Module tạo features cho model dự đoán AQI
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo time-based features từ cột datetime hoặc dt
    
    Tính năng: Tạo các đặc trưng thời gian bao gồm hour, day_of_week, day, month,
    is_weekend và cyclical encodings (sin/cos) cho hour và day_of_week.
    
    Args:
        df: DataFrame với cột 'datetime' hoặc 'dt' (datetime column)
        
    Returns:
        DataFrame với time features mới được thêm vào
    """
    df_new = df.copy()
    
    # Hỗ trợ cả 'datetime' và 'dt' column names
    dt_col = None
    if 'dt' in df_new.columns:
        dt_col = 'dt'
    elif 'datetime' in df_new.columns:
        dt_col = 'datetime'
    
    if dt_col:
        # Tạo các time features cơ bản
        df_new['hour'] = df_new[dt_col].dt.hour
        df_new['day_of_week'] = df_new[dt_col].dt.dayofweek
        df_new['day'] = df_new[dt_col].dt.day
        df_new['month'] = df_new[dt_col].dt.month
        df_new['is_weekend'] = (df_new['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding cho hour (sin/cos transformation để giữ tính chu kỳ)
        df_new['hour_sin'] = np.sin(2 * np.pi * df_new['hour'] / 24)
        df_new['hour_cos'] = np.cos(2 * np.pi * df_new['hour'] / 24)
        
        # Cyclical encoding cho day_of_week (tuần có tính chu kỳ)
        df_new['dow_sin'] = np.sin(2 * np.pi * df_new['day_of_week'] / 7)
        df_new['dow_cos'] = np.cos(2 * np.pi * df_new['day_of_week'] / 7)
        
        logger.info("✅ Created time features: hour, day_of_week, day, month, is_weekend, cyclical encodings")
    
    return df_new


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Tạo lag features cho các cột được chỉ định
    
    Tính năng: Tạo các đặc trưng lag (giá trị tại các thời điểm trước đó)
    để model có thể học từ các giá trị lịch sử.
    
    Args:
        df: DataFrame đầu vào
        columns: List các cột cần tạo lag features
        lags: List các lag periods (số giờ, ví dụ: [1, 2, 3, 6, 12, 24])
        
    Returns:
        DataFrame với lag features mới được thêm vào
    """
    df_new = df.copy()
    
    for col in columns:
        if col in df_new.columns:
            for lag in lags:
                # Tạo lag feature với tên dạng: column_lag_Xh
                df_new[f'{col}_lag_{lag}h'] = df_new[col].shift(lag)
    
    logger.info(f"✅ Created lag features for {len(columns)} columns with lags: {lags}")
    
    return df_new


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Tạo rolling statistics features
    
    Tính năng: Tạo các đặc trưng thống kê trượt (rolling mean, std, min, max)
    để capture các xu hướng ngắn hạn và trung hạn của dữ liệu.
    
    Args:
        df: DataFrame đầu vào
        columns: List các cột cần tạo rolling features
        windows: List các window sizes (số giờ, ví dụ: [6, 12, 24])
        
    Returns:
        DataFrame với rolling features mới được thêm vào
    """
    df_new = df.copy()
    
    for col in columns:
        if col in df_new.columns:
            for window in windows:
                # Rolling mean - giá trị trung bình trượt
                df_new[f'{col}_rolling_mean_{window}h'] = df_new[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std - độ lệch chuẩn trượt (đo biến động)
                df_new[f'{col}_rolling_std_{window}h'] = df_new[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min/max - giá trị min/max trong cửa sổ thời gian
                df_new[f'{col}_rolling_min_{window}h'] = df_new[col].rolling(window=window, min_periods=1).min()
                df_new[f'{col}_rolling_max_{window}h'] = df_new[col].rolling(window=window, min_periods=1).max()
    
    logger.info(f"✅ Created rolling features for {len(columns)} columns with windows: {windows}")
    
    return df_new


def create_spatial_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Tạo spatial features từ lat, lon
    
    Tính năng: Tạo các đặc trưng không gian (spatial) bao gồm lat_scaled, lon_scaled
    và lat_lon_interaction để model có thể học từ vị trí địa lý.
    
    Args:
        df: DataFrame với cột 'lat' và 'lon'
        scaler: StandardScaler đã được fit (dùng cho inference), nếu None sẽ tạo mới
        
    Returns:
        Tuple[DataFrame, StandardScaler]: DataFrame với spatial features và scaler đã fit
    """
    df_new = df.copy()
    
    # Kiểm tra xem có cột lat và lon không
    if 'lat' in df_new.columns and 'lon' in df_new.columns:
        # Nếu chưa có scaler, tạo mới và fit
        if scaler is None:
            scaler = StandardScaler()
            spatial_data = df_new[['lat', 'lon']].values
            scaled_data = scaler.fit_transform(spatial_data)
        else:
            # Dùng scaler đã có (cho inference)
            spatial_data = df_new[['lat', 'lon']].values
            scaled_data = scaler.transform(spatial_data)
        
        # Tạo lat_scaled và lon_scaled features
        df_new['lat_scaled'] = scaled_data[:, 0]
        df_new['lon_scaled'] = scaled_data[:, 1]
        
        # Tạo lat_lon_interaction feature (tương tác giữa lat và lon)
        df_new['lat_lon_interaction'] = df_new['lat_scaled'] * df_new['lon_scaled']
        
        logger.info("✅ Created spatial features: lat_scaled, lon_scaled, lat_lon_interaction")
    else:
        logger.warning("⚠️  lat và/hoặc lon columns không tồn tại, bỏ qua spatial features")
    
    return df_new, scaler


def engineer_features(
    df: pd.DataFrame,
    pollutant_cols: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Pipeline feature engineering hoàn chỉnh với spatial features
    
    Tính năng: Tạo toàn bộ features bao gồm time features, lag features,
    rolling features và spatial features cho global model.
    
    Args:
        df: DataFrame đầu vào (phải có 'dt' hoặc 'datetime', 'lat', 'lon', pollutant columns và 'aqi')
        pollutant_cols: List các cột pollutant, mặc định là None (sẽ dùng danh sách chuẩn)
        lags: List các lag periods, mặc định là None (sẽ dùng [1, 2, 3, 6, 12, 24])
        rolling_windows: List các rolling windows, mặc định là None (sẽ dùng [6, 24])
        
    Returns:
        Tuple[DataFrame, StandardScaler]: DataFrame với tất cả features và spatial_scaler
    """
    logger.info("🚀 Starting feature engineering pipeline...")
    
    df_featured = df.copy()
    
    # 1. Create time features
    df_featured = create_time_features(df_featured)
    
    # 2. Define pollutant columns (sử dụng danh sách chuẩn nếu không được cung cấp)
    if pollutant_cols is None:
        pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    available_pollutants = [col for col in pollutant_cols if col in df_featured.columns]
    
    # 3. Create lag features (mặc định: 1h, 2h, 3h, 6h, 12h, 24h)
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]
    df_featured = create_lag_features(df_featured, available_pollutants, lags)
    
    # 4. Create rolling features (mặc định: 6h, 24h)
    if rolling_windows is None:
        rolling_windows = [6, 24]
    df_featured = create_rolling_features(df_featured, available_pollutants, rolling_windows)
    
    # 5. Create spatial features (lat_scaled, lon_scaled, lat_lon_interaction)
    df_featured, spatial_scaler = create_spatial_features(df_featured, scaler=None)
    
    # 6. Drop rows with NaN values created by lag/rolling features
    original_rows = len(df_featured)
    df_featured = df_featured.dropna().reset_index(drop=True)
    dropped_rows = original_rows - len(df_featured)
    
    logger.info(f"✅ Feature engineering completed")
    logger.info(f"   Original rows: {original_rows}")
    logger.info(f"   Dropped rows (NaN): {dropped_rows}")
    logger.info(f"   Final rows: {len(df_featured)}")
    logger.info(f"   Total features: {len(df_featured.columns)}")
    
    return df_featured, spatial_scaler
