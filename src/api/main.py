"""FastAPI application for watch price forecasting."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .forecast_service import ForecastService
from .cloud_forecast_service import CloudForecastService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Watch Price Forecasting API",
    description="API for predicting luxury watch prices using machine learning models",
    version="1.0.0"
)

# Global service instance (will be initialized on first request)
_service: Optional[ForecastService] = None


def get_service() -> ForecastService:
    """Get or create the forecast service instance."""
    global _service
    if _service is None:
        # Check if running in cloud environment
        environment = os.getenv("ENVIRONMENT", "local")
        bucket_name = os.getenv("MODEL_BUCKET", "timepiece-watch-models")

        if environment == "production":
            # Use cloud service for production
            _service = CloudForecastService(bucket_name=bucket_name)
            logger.info("Initialized CloudForecastService with bucket: %s", bucket_name)
        else:
            # Use local service for development
            model_dir = "data/output/models"
            data_path = "data/output/featured_data.csv"
            _service = ForecastService(model_dir=model_dir, data_path=data_path)
            logger.info("Initialized local ForecastService")
    return _service


# Request/Response models
class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    watch_id: str = Field(..., description="The ID of the watch to predict for")
    horizon: int = Field(default=7, ge=1, le=30, description="Number of days ahead to predict")
    model_name: str = Field(default="lightgbm", description="Name of the model to use")


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    watch_id: str
    model_name: str
    horizon_days: int
    feature_date: str
    prediction_date: str
    prediction: float
    current_price: Optional[float] = None


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    watch_ids: List[str] = Field(..., description="List of watch IDs to predict for")
    horizon: int = Field(default=7, ge=1, le=30, description="Number of days ahead to predict")
    model_name: str = Field(default="lightgbm", description="Name of the model to use")


class BatchPredictResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    predictions: List[PredictResponse]
    total_requested: int
    successful_predictions: int


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    available_watches: Optional[int] = None
    available_models: Optional[int] = None
    model_directory: Optional[str] = None
    data_path: Optional[str] = None
    error: Optional[str] = None


class WatchListResponse(BaseModel):
    """Response model for available watches endpoint."""
    watches: List[str]
    total_count: int


class ModelListResponse(BaseModel):
    """Response model for available models endpoint."""
    models: List[str]
    total_count: int


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Watch Price Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        service = get_service()
        health_data = service.health_check()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error("Health check failed: %s", str(e))
        return HealthResponse(status="unhealthy", error=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Generate a price prediction for a single watch."""
    try:
        service = get_service()
        result = service.predict_single(
            watch_id=request.watch_id,
            horizon=request.horizon,
            model_name=request.model_name
        )
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed for watch %s: %s", request.watch_id, str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Generate price predictions for multiple watches."""
    try:
        service = get_service()
        results = service.predict_multiple(
            watch_ids=request.watch_ids,
            horizon=request.horizon,
            model_name=request.model_name
        )

        predictions = [PredictResponse(**result) for result in results]

        return BatchPredictResponse(
            predictions=predictions,
            total_requested=len(request.watch_ids),
            successful_predictions=len(predictions)
        )
    except Exception as e:
        logger.error("Batch prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/watches", response_model=WatchListResponse)
async def get_available_watches():
    """Get list of available watch IDs."""
    try:
        service = get_service()
        watches = service.get_available_watches()
        return WatchListResponse(watches=watches, total_count=len(watches))
    except Exception as e:
        logger.error("Failed to get available watches: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models", response_model=ModelListResponse)
async def get_available_models():
    """Get list of available model names."""
    try:
        service = get_service()
        models = service.get_available_models()
        return ModelListResponse(models=models, total_count=len(models))
    except Exception as e:
        logger.error("Failed to get available models: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {"detail": "Endpoint not found"}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error("Internal server error: %s", str(exc))
    return {"detail": "Internal server error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)