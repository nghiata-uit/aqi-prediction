"""
Module xử lý và làm sạch dữ liệu AQI
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dữ liệu từ CSV file
    
    Args:
        file_path: Đường dẫn đến file CSV
        
    Returns:
        DataFrame chứa dữ liệu
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded data from {file_path}")
        logger.info(f"   Shape: {df.shape}")
        
        # Convert datetime column to datetime type
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý missing values
    
    Args:
        df: DataFrame đầu vào
        
    Returns:
        DataFrame đã xử lý missing values
    """
    df_clean = df.copy()
    
    # Kiểm tra missing values
    missing_counts = df_clean.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        logger.warning(f"⚠️  Found missing values in columns: {missing_cols.to_dict()}")
        
        # Xử lý missing values cho các cột số bằng forward fill, sau đó backward fill
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                
        logger.info("✅ Missing values handled using forward/backward fill")
    else:
        logger.info("✅ No missing values found")
    
    return df_clean


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Kiểm tra chất lượng dữ liệu
    
    Args:
        df: DataFrame cần kiểm tra
        
    Returns:
        Dictionary chứa thông tin về chất lượng dữ liệu
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'date_range': None,
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
    }
    
    # Kiểm tra date range
    if 'datetime' in df.columns:
        quality_report['date_range'] = {
            'start': str(df['datetime'].min()),
            'end': str(df['datetime'].max()),
            'total_hours': len(df)
        }
    
    # Kiểm tra outliers cho các pollutants
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    outliers = {}
    for col in pollutant_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outliers[col] = int(outlier_count)
    
    quality_report['outliers'] = outliers
    
    logger.info("📊 Data Quality Report:")
    logger.info(f"   Total rows: {quality_report['total_rows']}")
    logger.info(f"   Total columns: {quality_report['total_columns']}")
    logger.info(f"   Duplicate rows: {quality_report['duplicate_rows']}")
    logger.info(f"   Date range: {quality_report['date_range']}")
    
    return quality_report


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Pipeline xử lý dữ liệu hoàn chỉnh
    
    Args:
        file_path: Đường dẫn đến file CSV
        
    Returns:
        DataFrame đã được xử lý
    """
    logger.info("🚀 Starting data preprocessing pipeline...")
    
    # Load data
    df = load_data(file_path)
    
    # Check data quality
    quality_report = check_data_quality(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove duplicates if any
    if quality_report['duplicate_rows'] > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"✅ Removed {quality_report['duplicate_rows']} duplicate rows")
    
    logger.info("✅ Data preprocessing completed")
    
    return df
