"""
FastAPI application cho AQI Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))

from api.models import PredictionRequest, PredictionResponse, HealthResponse, LocationRequest
from api.dependencies import model_manager
from api.utils import create_prediction_dataframe
from src.prediction import predict_next_24h
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AQI Prediction API",
    description="API dự đoán chỉ số AQI cho 24-72 giờ tiếp theo",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

startup_time = time.time()


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting AQI Prediction API...")
    model_manager.load_models()
    model_manager.load_historical_data()
    logger.info("✅ API ready!")


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "🌍 AQI Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return {
        "status": "healthy" if model_manager.is_ready else "degraded",
        "model_loaded": model_manager.is_ready,
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - startup_time, 2)
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_aqi(request: PredictionRequest):
    """Dự đoán AQI theo location"""
    try:
        if not model_manager.is_ready:
            raise HTTPException(status_code=503, detail="Models not loaded")

        lat = request.location.latitude
        lon = request.location.longitude
        hours_ahead = request.location.hours_ahead

        logger.info(f"📍 Prediction for ({lat}, {lon})")

        historical_data = model_manager.get_historical_data(lat, lon, radius=1.0)
        if historical_data is None or len(historical_data) < 50:
            raise HTTPException(status_code=404, detail=f"No data near ({lat}, {lon})")

        df_features = engineer_features(historical_data)
        if len(df_features) < 24:
            raise HTTPException(status_code=400, detail="Insufficient data")

        model = model_manager.get_model("xgboost")
        scaler = model_manager.get_scaler()
        feature_cols = model_manager.get_feature_cols()

        predictions_df = predict_next_24h(model, df_features.tail(100), scaler, feature_cols)
        predictions_list = create_prediction_dataframe(
            predictions_df['predicted_aqi'].tolist(),
            datetime.now(),
            hours_ahead
        )

        actual_location = {
            "latitude": float(historical_data.iloc[-1]['lat']),
            "longitude": float(historical_data.iloc[-1]['lon'])
        }

        return {
            "location": {
                "requested": {"latitude": lat, "longitude": lon},
                "actual": actual_location,
                "distance_degrees": round(((lat - actual_location['latitude']) ** 2 +
                                           (lon - actual_location['longitude']) ** 2) ** 0.5, 4)
            },
            "prediction_time": datetime.now(),
            "model_used": "XGBoost",
            "predictions": predictions_list,
            "metadata": {
                "data_points_used": len(historical_data),
                "features_generated": len(feature_cols),
                "last_data_timestamp": historical_data['dt'].max().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{lat}/{lon}", response_model=PredictionResponse, tags=["Prediction"])
async def predict_simple(lat: float, lon: float, hours: int = 24):
    """Simple GET endpoint"""
    return await predict_aqi(PredictionRequest(
        location=LocationRequest(latitude=lat, longitude=lon, hours_ahead=hours)
    ))


@app.get("/models", tags=["Models"])
async def list_models():
    return {
        "available_models": ["xgboost", "random_forest"],
        "default_model": "xgboost",
        "model_loaded": model_manager.is_ready
    }