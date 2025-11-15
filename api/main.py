"""
FastAPI application for AQI prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
from typing import List

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelsListResponse,
    ModelInfo,
    ModelComparisonResponse,
    LocationInfo
)
from .predict import predict_aqi
from .utils import list_available_models, get_model_comparison

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AQI Prediction API",
    description="API for predicting Air Quality Index 24 hours ahead",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models directory
MODELS_DIR = Path(__file__).parent.parent / "models"


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "AQI Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns system status and number of loaded models
    """
    try:
        models = list_available_models(MODELS_DIR)
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded=len(models)
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict AQI 24 hours ahead for a given location
    
    Args:
        request: Prediction request with location and current pollutant data
        
    Returns:
        Prediction response with AQI forecast and model information
    """
    try:
        logger.info(f"Prediction request for location ({request.lat}, {request.lon})")
        
        # Convert current_data to dict
        current_data = request.current_data.model_dump()
        
        # Make prediction
        result = predict_aqi(
            lat=request.lat,
            lon=request.lon,
            current_data=current_data,
            datetime_str=request.datetime,
            models_dir=MODELS_DIR
        )
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for location ({request.lat}, {request.lon})"
            )
        
        # Convert to response model
        response = PredictionResponse(**result)
        
        logger.info(f"Prediction successful: AQI={response.predicted_aqi}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """
    List all available trained models and their locations
    
    Returns:
        List of models with their locations and performance metrics
    """
    try:
        models_info = list_available_models(MODELS_DIR)
        
        models = []
        for info in models_info:
            model = ModelInfo(
                location=LocationInfo(lat=info['lat'], lon=info['lon']),
                model_type=info['model_type'],
                metrics={
                    "mae": info['metrics'].get('mae', 0.0),
                    "rmse": info['metrics'].get('rmse', 0.0),
                    "r2": info['metrics'].get('r2', None)
                },
                trained_date=info.get('trained_date')
            )
            models.append(model)
        
        return ModelsListResponse(
            total_locations=len(models),
            models=models
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/model-comparison/{lat}/{lon}", response_model=ModelComparisonResponse, tags=["Models"])
async def get_model_comparison_endpoint(lat: float, lon: float):
    """
    Get comparison of all models for a specific location
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Comparison of all models trained for this location
    """
    try:
        comparison = get_model_comparison(lat, lon, MODELS_DIR)
        
        if comparison is None:
            raise HTTPException(
                status_code=404,
                detail=f"No model comparison data found for location ({lat}, {lon})"
            )
        
        # Convert to response format
        response = ModelComparisonResponse(
            location=LocationInfo(lat=comparison['lat'], lon=comparison['lon']),
            models=comparison['models'],
            best_model=comparison['best_model']
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model comparison: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model comparison: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
