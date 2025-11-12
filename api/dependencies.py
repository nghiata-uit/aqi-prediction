"""
Dependencies cho FastAPI - load models và data
"""
import joblib
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton class để quản lý models"""
    _instance = None
    _models = {}
    _scaler = None
    _feature_cols = None
    _historical_data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def load_models(self, model_dir: str = "models"):
        """Load tất cả trained models"""
        model_path = Path(model_dir)

        try:
            xgb_path = model_path / "xgboost.pkl"
            if xgb_path.exists():
                self._models['xgboost'] = joblib.load(xgb_path)
                logger.info("✅ XGBoost model loaded")

            rf_path = model_path / "random_forest.pkl"
            if rf_path.exists():
                self._models['random_forest'] = joblib.load(rf_path)
                logger.info("✅ Random Forest model loaded")

            scaler_path = model_path / "scaler.pkl"
            if scaler_path.exists():
                self._scaler = joblib.load(scaler_path)
                logger.info("✅ Scaler loaded")

            features_path = model_path / "feature_columns.pkl"
            if features_path.exists():
                self._feature_cols = joblib.load(features_path)
                logger.info(f"✅ Feature columns loaded ({len(self._feature_cols)} features)")

            return True
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False

    def load_historical_data(self, data_path: str = "data/sample_data.csv"):
        """Load historical data cho location-based prediction"""
        try:
            self._historical_data = pd.read_csv(data_path)
            self._historical_data['dt'] = pd.to_datetime(self._historical_data['dt'])
            logger.info(f"✅ Historical data loaded ({len(self._historical_data)} rows)")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False

    def get_model(self, model_name: str = "xgboost"):
        return self._models.get(model_name)

    def get_scaler(self):
        return self._scaler

    def get_feature_cols(self):
        return self._feature_cols

    def get_historical_data(self, lat: float, lon: float, radius: float = 0.1):
        """Get historical data gần location"""
        if self._historical_data is None:
            return None

        data = self._historical_data[
            (abs(self._historical_data['lat'] - lat) <= radius) &
            (abs(self._historical_data['lon'] - lon) <= radius)
            ].copy()

        return data.tail(100) if len(data) > 0 else None

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0 and self._scaler is not None


model_manager = ModelManager()