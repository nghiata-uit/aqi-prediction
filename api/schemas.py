"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict
from datetime import datetime


class CurrentData(BaseModel):
    """Current pollutant data"""
    co: float = Field(..., description="Carbon Monoxide level")
    no: float = Field(..., description="Nitrogen Monoxide level")
    no2: float = Field(..., description="Nitrogen Dioxide level")
    o3: float = Field(..., description="Ozone level")
    so2: float = Field(..., description="Sulfur Dioxide level")
    pm2_5: float = Field(..., description="PM2.5 Particulate Matter level")
    pm10: float = Field(..., description="PM10 Particulate Matter level")
    nh3: float = Field(..., description="Ammonia level")

    @field_validator('co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3')
    @classmethod
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError('Pollutant values must be non-negative')
        return v


class PredictionRequest(BaseModel):
    """Request schema for AQI prediction"""
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    current_data: CurrentData
    datetime: Optional[str] = Field(None, description="Optional datetime in format 'YYYY-MM-DD HH:MM:SS'")

    @field_validator('lat')
    @classmethod
    def validate_lat(cls, v):
        if v < -90 or v > 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v

    @field_validator('lon')
    @classmethod
    def validate_lon(cls, v):
        if v < -180 or v > 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v


class LocationInfo(BaseModel):
    """Location information"""
    lat: float
    lon: float


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    mae: float
    rmse: float
    r2: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response schema for AQI prediction"""
    location: LocationInfo
    predicted_aqi: float
    prediction_for: str
    model_used: str
    confidence_score: float
    metrics: ModelMetrics


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: int


class ModelInfo(BaseModel):
    """Information about a trained model"""
    location: LocationInfo
    model_type: str
    metrics: ModelMetrics
    trained_date: Optional[str] = None


class ModelsListResponse(BaseModel):
    """Response for listing all available models"""
    total_locations: int
    models: list[ModelInfo]


class ModelComparisonResponse(BaseModel):
    """Response for model comparison at a location"""
    location: LocationInfo
    models: Dict[str, ModelMetrics]
    best_model: str
