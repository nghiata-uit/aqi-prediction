"""
Pydantic models cho API requests và responses
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


class LocationRequest(BaseModel):
    """Request model cho prediction endpoint"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    hours_ahead: int = Field(24, ge=1, le=72, description="Number of hours to predict (1-72)")

    @field_validator('latitude')
    @classmethod
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @field_validator('longitude')
    @classmethod
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v


class PollutantData(BaseModel):
    """Model cho dữ liệu pollutant hiện tại"""
    co: float = Field(..., ge=0, description="Carbon Monoxide (μg/m³)")
    no: float = Field(..., ge=0, description="Nitrogen Monoxide (μg/m³)")
    no2: float = Field(..., ge=0, description="Nitrogen Dioxide (μg/m³)")
    o3: float = Field(..., ge=0, description="Ozone (μg/m³)")
    so2: float = Field(..., ge=0, description="Sulfur Dioxide (μg/m³)")
    pm2_5: float = Field(..., ge=0, description="PM2.5 (μg/m³)")
    pm10: float = Field(..., ge=0, description="PM10 (μg/m³)")
    nh3: float = Field(..., ge=0, description="Ammonia (μg/m³)")


class PredictionRequest(BaseModel):
    """Request model với pollutant data"""
    location: LocationRequest
    current_pollutants: Optional[PollutantData] = None


class AQIPrediction(BaseModel):
    """Model cho một prediction point"""
    timestamp: datetime
    hour_ahead: int
    predicted_aqi: float
    aqi_category: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response model cho predictions"""
    location: dict
    prediction_time: datetime
    model_used: str
    predictions: List[AQIPrediction]
    metadata: dict


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float