"""
API Dependencies - Quản lý loading và caching models

Tính năng:
- Load global XGBoost model với spatial features
- Load spatial scaler và feature columns
- Maintain backward compatibility với existing models
- Expose model manager cho API endpoints
"""
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Quản lý models và artifacts cho API
    
    Tính năng:
    - Load và cache models (global và legacy models)
    - Load spatial scaler và feature columns
    - Cung cấp methods để lấy model dựa trên location
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Khởi tạo ModelManager
        
        Args:
            models_dir: Đường dẫn đến thư mục chứa models (default: models/)
        """
        if models_dir is None:
            # Mặc định là thư mục models/ ở root của project
            self.models_dir = Path(__file__).parent.parent / 'models'
        else:
            self.models_dir = Path(models_dir)
        
        self.global_model = None
        self.spatial_scaler = None
        self.feature_columns_global = None
        
        # Backward compatibility - legacy models
        self.xgboost_model = None
        self.random_forest_model = None
        self.scaler = None
        
        self._is_ready = False
        
        # Load models khi khởi tạo
        self._load_models()
    
    def _load_models(self):
        """
        Load tất cả models và artifacts có sẵn
        
        Priority: global model > legacy models
        """
        logger.info("🔄 Loading models and artifacts...")
        
        # Load global model artifacts (ưu tiên cao nhất)
        global_model_path = self.models_dir / 'xgboost_global.pkl'
        spatial_scaler_path = self.models_dir / 'spatial_scaler.pkl'
        feature_cols_path = self.models_dir / 'feature_columns_global.pkl'
        
        if global_model_path.exists():
            try:
                self.global_model = joblib.load(global_model_path)
                logger.info(f"✅ Loaded global model from {global_model_path}")
                self._is_ready = True
            except Exception as e:
                logger.error(f"❌ Error loading global model: {e}")
        
        if spatial_scaler_path.exists():
            try:
                self.spatial_scaler = joblib.load(spatial_scaler_path)
                logger.info(f"✅ Loaded spatial scaler from {spatial_scaler_path}")
            except Exception as e:
                logger.error(f"❌ Error loading spatial scaler: {e}")
        
        if feature_cols_path.exists():
            try:
                self.feature_columns_global = joblib.load(feature_cols_path)
                logger.info(f"✅ Loaded feature columns from {feature_cols_path}")
            except Exception as e:
                logger.error(f"❌ Error loading feature columns: {e}")
        
        # Backward compatibility - load legacy models nếu có
        xgb_path = self.models_dir / 'xgboost.pkl'
        rf_path = self.models_dir / 'random_forest.pkl'
        scaler_path = self.models_dir / 'scaler.pkl'
        
        if xgb_path.exists() and self.global_model is None:
            try:
                self.xgboost_model = joblib.load(xgb_path)
                logger.info(f"✅ Loaded legacy XGBoost model from {xgb_path}")
                self._is_ready = True
            except Exception as e:
                logger.error(f"❌ Error loading legacy XGBoost: {e}")
        
        if rf_path.exists():
            try:
                self.random_forest_model = joblib.load(rf_path)
                logger.info(f"✅ Loaded legacy Random Forest model from {rf_path}")
            except Exception as e:
                logger.error(f"❌ Error loading legacy Random Forest: {e}")
        
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Loaded legacy scaler from {scaler_path}")
            except Exception as e:
                logger.error(f"❌ Error loading legacy scaler: {e}")
        
        if not self._is_ready:
            logger.warning("⚠️  No models loaded. Please train a model first.")
    
    @property
    def is_ready(self) -> bool:
        """
        Kiểm tra xem model manager đã sẵn sàng chưa
        
        Returns:
            bool: True nếu có ít nhất một model được load
        """
        return self._is_ready
    
    def get_model_for_location(
        self, 
        lat: float, 
        lon: float
    ) -> Tuple[Any, Optional[Any], Optional[list]]:
        """
        Lấy model phù hợp cho location (latitude, longitude)
        
        Hiện tại trả về global model (trong tương lai có thể mở rộng với location-specific models)
        
        Args:
            lat: Latitude của location
            lon: Longitude của location
            
        Returns:
            Tuple[model, scaler, feature_cols]:
                - model: XGBoost model để dùng cho prediction
                - scaler: Spatial scaler (hoặc legacy scaler)
                - feature_cols: List các feature columns theo đúng thứ tự
        """
        # Ưu tiên global model nếu có
        if self.global_model is not None:
            return (
                self.global_model,
                self.spatial_scaler,
                self.feature_columns_global
            )
        
        # Fallback sang legacy XGBoost model
        if self.xgboost_model is not None:
            return (
                self.xgboost_model,
                self.scaler,
                None  # Legacy model không có feature_columns_global
            )
        
        # Fallback sang Random Forest nếu không có XGBoost
        if self.random_forest_model is not None:
            return (
                self.random_forest_model,
                self.scaler,
                None
            )
        
        raise ValueError("No model available for prediction")
    
    def get_default_model(self) -> Tuple[Any, Optional[Any], Optional[list]]:
        """
        Lấy default model (global model nếu có, nếu không thì legacy model)
        
        Returns:
            Tuple[model, scaler, feature_cols]:
                - model: XGBoost model để dùng cho prediction
                - scaler: Spatial scaler (hoặc legacy scaler)
                - feature_cols: List các feature columns theo đúng thứ tự
        """
        if self.global_model is not None:
            return (
                self.global_model,
                self.spatial_scaler,
                self.feature_columns_global
            )
        
        if self.xgboost_model is not None:
            return (
                self.xgboost_model,
                self.scaler,
                None
            )
        
        if self.random_forest_model is not None:
            return (
                self.random_forest_model,
                self.scaler,
                None
            )
        
        raise ValueError("No model available")


# Global instance của ModelManager
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """
    Dependency injection function để lấy ModelManager instance
    
    Returns:
        ModelManager: Global model manager instance
    """
    return model_manager
