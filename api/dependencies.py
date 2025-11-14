"""
API dependencies module - quản lý loading và caching của model artifacts
"""
import joblib
from pathlib import Path
from typing import Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Global variables để cache artifacts
_model = None
_spatial_scaler = None
_feature_columns = None
_historical_data = None

# Paths to artifacts
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def load_artifacts() -> bool:
    """
    Load tất cả artifacts cần thiết cho API (model, scaler, feature columns)
    
    Returns:
        bool: True nếu load thành công, False nếu thất bại
    """
    global _model, _spatial_scaler, _feature_columns
    
    try:
        # Load model
        model_path = MODELS_DIR / "xgboost_global.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        _model = joblib.load(model_path)
        logger.info(f"✅ Loaded model from {model_path}")
        
        # Load spatial scaler
        scaler_path = MODELS_DIR / "spatial_scaler.pkl"
        if not scaler_path.exists():
            logger.error(f"Spatial scaler file not found: {scaler_path}")
            return False
        _spatial_scaler = joblib.load(scaler_path)
        logger.info(f"✅ Loaded spatial scaler from {scaler_path}")
        
        # Load feature columns
        feature_cols_path = MODELS_DIR / "feature_columns_global.pkl"
        if not feature_cols_path.exists():
            logger.error(f"Feature columns file not found: {feature_cols_path}")
            return False
        _feature_columns = joblib.load(feature_cols_path)
        logger.info(f"✅ Loaded {len(_feature_columns)} feature columns from {feature_cols_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {str(e)}")
        return False


def load_historical_data() -> bool:
    """
    Load historical data cho inference
    
    Returns:
        bool: True nếu load thành công, False nếu thất bại
    """
    global _historical_data
    
    try:
        data_path = DATA_DIR / "sample_data.csv"
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return False
        
        _historical_data = pd.read_csv(data_path)
        logger.info(f"✅ Loaded historical data: {len(_historical_data)} rows")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading historical data: {str(e)}")
        return False


def get_model():
    """
    Lấy model đã được load
    
    Returns:
        XGBoost model hoặc None nếu chưa load
    """
    return _model


def get_spatial_scaler():
    """
    Lấy spatial scaler đã được load
    
    Returns:
        StandardScaler hoặc None nếu chưa load
    """
    return _spatial_scaler


def get_feature_columns() -> Optional[List[str]]:
    """
    Lấy danh sách feature columns đã được load
    
    Returns:
        List[str] feature columns hoặc None nếu chưa load
    """
    return _feature_columns


def get_historical_data() -> Optional[pd.DataFrame]:
    """
    Lấy historical data đã được load
    
    Returns:
        DataFrame hoặc None nếu chưa load
    """
    return _historical_data


def is_ready() -> bool:
    """
    Kiểm tra xem tất cả artifacts đã sẵn sàng chưa
    
    Returns:
        bool: True nếu model, scaler và feature columns đã được load
    """
    return _model is not None and _spatial_scaler is not None and _feature_columns is not None
