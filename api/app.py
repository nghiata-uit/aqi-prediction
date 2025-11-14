"""
FastAPI application cho AQI prediction với global spatial model
Sử dụng artifacts đã train (model, scaler, feature_columns)
"""
import sys
from pathlib import Path

# Thêm thư mục gốc vào Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import logging

from api.dependencies import (
    initialize, get_model, get_scaler, get_feature_cols,
    get_historical_data, is_ready
)
from src.feature_engineering import engineer_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(
    title="AQI Prediction API",
    description="API dự đoán chỉ số chất lượng không khí (AQI) sử dụng XGBoost model với spatial features",
    version="1.0.0"
)


# Pydantic models cho request/response
class PredictionInput(BaseModel):
    """Input cho prediction endpoint"""
    lat: float = Field(..., description="Vĩ độ (latitude)")
    lon: float = Field(..., description="Kinh độ (longitude)")
    co: float = Field(..., description="Carbon Monoxide (μg/m³)")
    no: float = Field(..., description="Nitrogen Monoxide (μg/m³)")
    no2: float = Field(..., description="Nitrogen Dioxide (μg/m³)")
    o3: float = Field(..., description="Ozone (μg/m³)")
    so2: float = Field(..., description="Sulfur Dioxide (μg/m³)")
    pm2_5: float = Field(..., description="Fine Particulate Matter (μg/m³)")
    pm10: float = Field(..., description="Coarse Particulate Matter (μg/m³)")
    nh3: float = Field(..., description="Ammonia (μg/m³)")
    
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


class PredictionResponse(BaseModel):
    """Response cho prediction endpoint"""
    predicted_aqi: float = Field(..., description="Chỉ số AQI dự đoán")
    lat_scaled: float = Field(..., description="Latitude sau khi chuẩn hóa")
    lon_scaled: float = Field(..., description="Longitude sau khi chuẩn hóa")
    model_name: str = Field(..., description="Tên model được sử dụng")


class HealthResponse(BaseModel):
    """Response cho health check endpoint"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    feature_columns_loaded: bool
    total_features: int


@app.on_event("startup")
async def startup_event():
    """Load artifacts khi API startup"""
    try:
        initialize()
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "AQI Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint - kiểm tra xem API đã sẵn sàng chưa
    """
    model = get_model()
    scaler = get_scaler()
    feature_cols = get_feature_cols()
    
    return HealthResponse(
        status="healthy" if is_ready() else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        feature_columns_loaded=feature_cols is not None,
        total_features=len(feature_cols) if feature_cols else 0
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_aqi(input_data: PredictionInput):
    """
    Dự đoán AQI dựa trên dữ liệu pollutants và vị trí địa lý
    
    - **lat**: Vĩ độ (latitude)
    - **lon**: Kinh độ (longitude)
    - **co**: Carbon Monoxide
    - **no**: Nitrogen Monoxide
    - **no2**: Nitrogen Dioxide
    - **o3**: Ozone
    - **so2**: Sulfur Dioxide
    - **pm2_5**: Fine Particulate Matter
    - **pm10**: Coarse Particulate Matter
    - **nh3**: Ammonia
    """
    # Kiểm tra API đã ready chưa
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not loaded. Please check server logs."
        )
    
    try:
        # Lấy model, scaler và feature columns
        model = get_model()
        spatial_scaler = get_scaler()
        feature_cols = get_feature_cols()
        historical_data = get_historical_data()
        
        # Tạo DataFrame từ input để chuẩn hóa spatial features
        input_df = pd.DataFrame([{
            'lat': input_data.lat,
            'lon': input_data.lon,
            'co': input_data.co,
            'no': input_data.no,
            'no2': input_data.no2,
            'o3': input_data.o3,
            'so2': input_data.so2,
            'pm2_5': input_data.pm2_5,
            'pm10': input_data.pm10,
            'nh3': input_data.nh3
        }])
        
        # Apply spatial scaler to lat/lon
        lat_lon_scaled = spatial_scaler.transform(input_df[['lat', 'lon']])
        lat_scaled, lon_scaled = lat_lon_scaled[0]
        
        # Để prediction, ta cần full features engineering
        # Lấy một số samples gần nhất từ historical data để tạo lag/rolling features
        # Trong production, cần caching hoặc database để store features
        
        # Đơn giản hóa: sử dụng historical data tail để tạo context
        df_for_features = historical_data.tail(100).copy()
        
        # Append input data vào cuối (giả sử là next timestamp)
        import datetime
        last_datetime = df_for_features['datetime'].max()
        next_datetime = last_datetime + pd.Timedelta(hours=1)
        
        new_row = input_df.copy()
        new_row['datetime'] = next_datetime
        new_row['aqi'] = 0  # Placeholder, sẽ không dùng
        
        df_combined = pd.concat([df_for_features, new_row], ignore_index=True)
        
        # Feature engineering với spatial scaler đã có
        df_featured, _ = engineer_features(df_combined, spatial_scaler=spatial_scaler)
        
        # Lấy row cuối cùng (là prediction row)
        last_row = df_featured.iloc[-1:][feature_cols]
        
        # Đảm bảo features theo đúng thứ tự trong feature_columns_global.pkl
        X_pred = last_row[feature_cols]
        
        # Predict
        prediction = model.predict(X_pred)[0]
        
        return PredictionResponse(
            predicted_aqi=float(prediction),
            lat_scaled=float(lat_scaled),
            lon_scaled=float(lon_scaled),
            model_name="xgboost_global"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """
    Lấy thông tin về model và features
    """
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    feature_cols = get_feature_cols()
    
    return {
        "model_name": "xgboost_global",
        "model_type": "XGBoost Regressor",
        "total_features": len(feature_cols),
        "features": feature_cols,
        "spatial_features": ["lat_scaled", "lon_scaled"],
        "description": "Global XGBoost model với spatial features (lat, lon)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
