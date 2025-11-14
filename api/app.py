"""
FastAPI application cho AQI prediction service
Sử dụng global XGBoost model với spatial features
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import logging

from api.dependencies import (
    load_artifacts,
    load_historical_data,
    get_model,
    get_spatial_scaler,
    get_feature_columns,
    get_historical_data,
    is_ready
)
from src.feature_engineering import engineer_features

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(
    title="AQI Prediction API",
    description="API dự đoán chỉ số AQI sử dụng global XGBoost model với spatial features",
    version="1.0.0"
)


class PredictionInput(BaseModel):
    """Schema cho input prediction request"""
    lat: float = Field(..., description="Latitude của vị trí cần dự đoán")
    lon: float = Field(..., description="Longitude của vị trí cần dự đoán")
    co: float = Field(..., ge=0, description="Carbon Monoxide (μg/m³)")
    no: float = Field(..., ge=0, description="Nitrogen Monoxide (μg/m³)")
    no2: float = Field(..., ge=0, description="Nitrogen Dioxide (μg/m³)")
    o3: float = Field(..., ge=0, description="Ozone (μg/m³)")
    so2: float = Field(..., ge=0, description="Sulfur Dioxide (μg/m³)")
    pm2_5: float = Field(..., ge=0, description="PM2.5 (μg/m³)")
    pm10: float = Field(..., ge=0, description="PM10 (μg/m³)")
    nh3: float = Field(..., ge=0, description="Ammonia (μg/m³)")
    
    class Config:
        schema_extra = {
            "example": {
                "lat": 106.7075,
                "lon": 10.804,
                "co": 704.51,
                "no": 8.31,
                "no2": 21.89,
                "o3": 63.35,
                "so2": 21.33,
                "pm2_5": 25.13,
                "pm10": 63.95,
                "nh3": 9.5
            }
        }


class PredictionOutput(BaseModel):
    """Schema cho output prediction response"""
    predicted_aqi: float = Field(..., description="Predicted AQI value")
    model_name: str = Field(default="xgboost_global", description="Model name used for prediction")


class HealthCheck(BaseModel):
    """Schema cho health check response"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    feature_columns_loaded: bool
    num_features: Optional[int]


@app.on_event("startup")
async def startup_event():
    """
    Load artifacts khi khởi động API
    """
    logger.info("🚀 Starting API server...")
    logger.info("📂 Loading model artifacts...")
    
    if load_artifacts():
        logger.info("✅ Artifacts loaded successfully")
    else:
        logger.error("❌ Failed to load artifacts")
    
    logger.info("📂 Loading historical data...")
    if load_historical_data():
        logger.info("✅ Historical data loaded successfully")
    else:
        logger.warning("⚠️ Failed to load historical data (optional)")


@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint
    """
    return {
        "message": "AQI Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint - kiểm tra xem API đã sẵn sàng chưa
    """
    model = get_model()
    scaler = get_spatial_scaler()
    feature_cols = get_feature_columns()
    
    return HealthCheck(
        status="ready" if is_ready() else "not_ready",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        feature_columns_loaded=feature_cols is not None,
        num_features=len(feature_cols) if feature_cols else None
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict AQI cho một location và pollutant data
    
    Args:
        input_data: PredictionInput với lat, lon và các pollutant values
        
    Returns:
        PredictionOutput với predicted AQI
    """
    # Kiểm tra xem artifacts đã được load chưa
    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Model artifacts not loaded."
        )
    
    try:
        # Lấy artifacts
        model = get_model()
        spatial_scaler = get_spatial_scaler()
        feature_columns = get_feature_columns()
        historical_data = get_historical_data()
        
        # Tạo DataFrame từ input
        input_df = pd.DataFrame([{
            'datetime': pd.Timestamp.now(),  # Sử dụng timestamp hiện tại
            'lat': input_data.lat,
            'lon': input_data.lon,
            'co': input_data.co,
            'no': input_data.no,
            'no2': input_data.no2,
            'o3': input_data.o3,
            'so2': input_data.so2,
            'pm2_5': input_data.pm2_5,
            'pm10': input_data.pm10,
            'nh3': input_data.nh3,
            'aqi': 0  # Dummy value, sẽ không được sử dụng
        }])
        
        # Convert datetime column to datetime type
        input_df['datetime'] = pd.to_datetime(input_df['datetime'])
        
        # Kết hợp với historical data để tạo lag và rolling features
        if historical_data is not None:
            # Thêm input vào cuối historical data
            historical_data['datetime'] = pd.to_datetime(historical_data['datetime'])
            combined_df = pd.concat([historical_data, input_df], ignore_index=True)
        else:
            # Nếu không có historical data, chỉ dùng input (lag features sẽ là NaN)
            combined_df = input_df
        
        # Feature engineering với spatial scaler đã được load
        featured_df, _ = engineer_features(combined_df, spatial_scaler=spatial_scaler)
        
        # Lấy dòng cuối cùng (input mới nhất)
        last_row = featured_df.iloc[-1:].copy()
        
        # Chọn các features theo đúng thứ tự đã lưu
        X = last_row[feature_columns].values
        
        # Predict
        prediction = model.predict(X)[0]
        
        return PredictionOutput(
            predicted_aqi=float(prediction),
            model_name="xgboost_global"
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Chạy server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
