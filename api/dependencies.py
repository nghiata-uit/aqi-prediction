"""
Dependencies cho API - quản lý việc load models và artifacts
Module này load các artifacts khi startup và cung cấp các hàm để access
"""
import sys
from pathlib import Path
from typing import Optional, List
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

# Thêm thư mục gốc vào Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables để cache các artifacts
_model = None
_spatial_scaler = None
_feature_columns = None
_historical_data = None


def load_artifacts(models_dir: Path = None):
    """
    Load tất cả artifacts từ thư mục models/ khi startup
    
    Args:
        models_dir: Đường dẫn đến thư mục chứa models (mặc định: project_root/models)
    """
    global _model, _spatial_scaler, _feature_columns
    
    if models_dir is None:
        models_dir = project_root / "models"
    
    try:
        logger.info("🚀 Loading model artifacts...")
        
        # Load XGBoost global model
        model_path = models_dir / "xgboost_global.pkl"
        if model_path.exists():
            _model = joblib.load(model_path)
            logger.info(f"✅ Loaded model from {model_path}")
        else:
            logger.error(f"❌ Model not found at {model_path}")
            
        # Load spatial scaler
        scaler_path = models_dir / "spatial_scaler.pkl"
        if scaler_path.exists():
            _spatial_scaler = joblib.load(scaler_path)
            logger.info(f"✅ Loaded spatial scaler from {scaler_path}")
        else:
            logger.error(f"❌ Spatial scaler not found at {scaler_path}")
            
        # Load feature columns
        feature_cols_path = models_dir / "feature_columns_global.pkl"
        if feature_cols_path.exists():
            _feature_columns = joblib.load(feature_cols_path)
            logger.info(f"✅ Loaded feature columns from {feature_cols_path}")
            logger.info(f"   Total features: {len(_feature_columns)}")
        else:
            logger.error(f"❌ Feature columns not found at {feature_cols_path}")
            
        logger.info("✅ All artifacts loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {str(e)}")
        raise


def load_historical_data(data_path: Path = None):
    """
    Load historical data để sử dụng cho predictions
    
    Args:
        data_path: Đường dẫn đến file dữ liệu (mặc định: project_root/data/sample_data.csv)
    """
    global _historical_data
    
    if data_path is None:
        data_path = project_root / "data" / "sample_data.csv"
    
    try:
        logger.info(f"📂 Loading historical data from {data_path}")
        from src.data_preprocessing import preprocess_data
        _historical_data = preprocess_data(str(data_path))
        logger.info(f"✅ Loaded historical data: {len(_historical_data)} samples")
    except Exception as e:
        logger.error(f"❌ Error loading historical data: {str(e)}")
        raise


def get_model():
    """
    Lấy model đã load
    
    Returns:
        XGBoost model hoặc None nếu chưa load
    """
    return _model


def get_scaler() -> Optional[StandardScaler]:
    """
    Lấy spatial scaler đã load
    
    Returns:
        StandardScaler hoặc None nếu chưa load
    """
    return _spatial_scaler


def get_feature_cols() -> Optional[List[str]]:
    """
    Lấy danh sách feature columns đã load
    
    Returns:
        List feature names hoặc None nếu chưa load
    """
    return _feature_columns


def get_historical_data() -> Optional[pd.DataFrame]:
    """
    Lấy historical data đã load
    
    Returns:
        DataFrame hoặc None nếu chưa load
    """
    return _historical_data


def is_ready() -> bool:
    """
    Kiểm tra xem tất cả artifacts đã được load chưa
    
    Returns:
        True nếu tất cả artifacts đã load, False nếu không
    """
    ready = (_model is not None and 
             _spatial_scaler is not None and 
             _feature_columns is not None)
    
    if not ready:
        logger.warning("⚠️  Not all artifacts are loaded:")
        logger.warning(f"   Model loaded: {_model is not None}")
        logger.warning(f"   Spatial scaler loaded: {_spatial_scaler is not None}")
        logger.warning(f"   Feature columns loaded: {_feature_columns is not None}")
    
    return ready


# Hàm khởi tạo để gọi khi startup
def initialize(models_dir: Path = None, data_path: Path = None):
    """
    Khởi tạo và load tất cả artifacts khi startup
    
    Args:
        models_dir: Đường dẫn đến thư mục chứa models
        data_path: Đường dẫn đến file historical data
    """
    logger.info("="*80)
    logger.info("🌍 INITIALIZING API DEPENDENCIES")
    logger.info("="*80)
    
    load_artifacts(models_dir)
    load_historical_data(data_path)
    
    if is_ready():
        logger.info("="*80)
        logger.info("✅ API READY TO SERVE REQUESTS")
        logger.info("="*80)
    else:
        logger.error("="*80)
        logger.error("❌ API NOT READY - Missing artifacts")
        logger.error("="*80)
        raise RuntimeError("Failed to initialize API dependencies")


if __name__ == "__main__":
    # Test loading
    initialize()
    print(f"\nModel type: {type(get_model())}")
    print(f"Scaler type: {type(get_scaler())}")
    print(f"Feature columns count: {len(get_feature_cols())}")
    print(f"Historical data shape: {get_historical_data().shape}")
    print(f"API ready: {is_ready()}")
