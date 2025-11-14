"""
FastAPI Application - AQI Prediction API

Tính năng:
- REST API endpoints cho AQI prediction
- Support global model với spatial features
- Backward compatible với legacy models
"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from api.dependencies import ModelManager, get_model_manager
from src.feature_engineering import create_spatial_features

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(
    title="AQI Prediction API",
    description="API để dự đoán chỉ số chất lượng không khí (AQI) sử dụng Machine Learning",
    version="2.0.0"
)


# ===== Pydantic Models =====

class PollutantData(BaseModel):
    """
    Model cho dữ liệu các chất ô nhiễm
    """
    co: float = Field(..., description="Carbon Monoxide (μg/m³)", ge=0)
    no: float = Field(..., description="Nitrogen Monoxide (μg/m³)", ge=0)
    no2: float = Field(..., description="Nitrogen Dioxide (μg/m³)", ge=0)
    o3: float = Field(..., description="Ozone (μg/m³)", ge=0)
    so2: float = Field(..., description="Sulfur Dioxide (μg/m³)", ge=0)
    pm2_5: float = Field(..., description="Fine Particulate Matter (μg/m³)", ge=0)
    pm10: float = Field(..., description="Coarse Particulate Matter (μg/m³)", ge=0)
    nh3: float = Field(..., description="Ammonia (μg/m³)", ge=0)


class PredictionRequest(BaseModel):
    """
    Request model cho AQI prediction
    """
    lat: float = Field(..., description="Latitude", ge=-90, le=90)
    lon: float = Field(..., description="Longitude", ge=-180, le=180)
    pollutants: PollutantData
    timestamp: Optional[datetime] = Field(None, description="Timestamp (mặc định: hiện tại)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "lat": 10.804,
                "lon": 106.7075,
                "pollutants": {
                    "co": 704.51,
                    "no": 8.31,
                    "no2": 21.89,
                    "o3": 63.35,
                    "so2": 21.33,
                    "pm2_5": 25.13,
                    "pm10": 63.95,
                    "nh3": 9.5
                },
                "timestamp": "2020-11-01T00:00:00"
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model cho AQI prediction
    """
    aqi_prediction: float = Field(..., description="Predicted AQI value")
    lat: float
    lon: float
    timestamp: datetime
    model_type: str = Field(..., description="Model type used (global/xgboost/random_forest)")


class HealthResponse(BaseModel):
    """
    Response model cho health check
    """
    status: str
    model_loaded: bool
    model_type: str


# ===== API Endpoints =====

@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint - thông tin về API
    """
    return {
        "message": "AQI Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(manager: ModelManager = Depends(get_model_manager)):
    """
    Health check endpoint
    
    Kiểm tra xem API và models có sẵn sàng không
    """
    model_type = "none"
    if manager.global_model is not None:
        model_type = "global"
    elif manager.xgboost_model is not None:
        model_type = "xgboost"
    elif manager.random_forest_model is not None:
        model_type = "random_forest"
    
    return HealthResponse(
        status="healthy" if manager.is_ready else "not_ready",
        model_loaded=manager.is_ready,
        model_type=model_type
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_aqi(
    request: PredictionRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Predict AQI dựa trên pollutant data và location
    
    Args:
        request: PredictionRequest với pollutants, lat, lon và timestamp
        manager: ModelManager dependency
        
    Returns:
        PredictionResponse với AQI prediction
    """
    try:
        # Kiểm tra xem có model nào sẵn sàng không
        if not manager.is_ready:
            raise HTTPException(
                status_code=503,
                detail="No model available. Please train a model first."
            )
        
        # Lấy model và scaler cho location này
        model, scaler, feature_cols = manager.get_model_for_location(
            request.lat, 
            request.lon
        )
        
        # Xác định timestamp (dùng hiện tại nếu không có)
        timestamp = request.timestamp if request.timestamp else datetime.now()
        
        # Tạo DataFrame từ request data
        data = {
            'lat': [request.lat],
            'lon': [request.lon],
            'co': [request.pollutants.co],
            'no': [request.pollutants.no],
            'no2': [request.pollutants.no2],
            'o3': [request.pollutants.o3],
            'so2': [request.pollutants.so2],
            'pm2_5': [request.pollutants.pm2_5],
            'pm10': [request.pollutants.pm10],
            'nh3': [request.pollutants.nh3],
            'dt': [timestamp]
        }
        df = pd.DataFrame(data)
        
        # Xác định model type để xử lý features đúng cách
        model_type = "global" if manager.global_model is not None else "legacy"
        
        if model_type == "global" and scaler is not None:
            # Global model - cần spatial features
            # Apply spatial scaler
            df_with_spatial, _ = create_spatial_features(df, scaler=scaler)
            
            # Chuẩn bị features theo đúng thứ tự trong feature_columns_global
            if feature_cols is not None:
                # Tạo các features cần thiết (simplified version cho real-time prediction)
                # Note: Trong production, cần implement đầy đủ feature engineering
                # hoặc cache các giá trị lag/rolling từ historical data
                
                # Đơn giản hóa: chỉ dùng current pollutants + spatial features + time features
                from src.feature_engineering import create_time_features
                df_with_features = create_time_features(df_with_spatial)
                
                # Lấy các features có sẵn
                available_features = {}
                for col in feature_cols:
                    if col in df_with_features.columns:
                        available_features[col] = df_with_features[col].values[0]
                    else:
                        # Nếu feature không có (lag/rolling), dùng giá trị mặc định
                        # Trong production, cần lấy từ historical data
                        available_features[col] = 0.0
                
                # Tạo feature vector theo đúng thứ tự
                X = np.array([[available_features[col] for col in feature_cols]])
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Feature columns not available for global model"
                )
        else:
            # Legacy model - không có spatial features
            # Simplified feature engineering cho real-time prediction
            raise HTTPException(
                status_code=501,
                detail="Legacy model prediction not fully implemented. Please use global model."
            )
        
        # Predict AQI
        aqi_pred = model.predict(X)[0]
        
        return PredictionResponse(
            aqi_prediction=float(aqi_pred),
            lat=request.lat,
            lon=request.lon,
            timestamp=timestamp,
            model_type=model_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ===== Main =====

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
