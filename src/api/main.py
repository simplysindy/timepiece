"""FastAPI application for watch price forecasting."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .cloud_forecast_service import CloudForecastService
from .forecast_service import ForecastService
from .logging_config import configure_logging

configure_logging()
logger = logging.getLogger("watch_forecast.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every incoming request with latency and trace metadata."""

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)
        self.environment = os.getenv("ENVIRONMENT", "local")
        self.project_id = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCLOUD_PROJECT")
            or os.getenv("PROJECT_ID")
        )

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        trace_header = request.headers.get("X-Cloud-Trace-Context", "")
        trace_id = trace_header.split("/")[0] if trace_header else ""

        extra: Dict[str, Any] = {
            "method": request.method,
            "path": request.url.path,
            "environment": self.environment,
        }

        if trace_id and self.project_id:
            extra["trace"] = f"projects/{self.project_id}/traces/{trace_id}"

        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - defensive logging
            latency_ms = (time.perf_counter() - start_time) * 1000
            extra.update({
                "status_code": 500,
                "latency_ms": round(latency_ms, 2),
            })
            logger.exception("request.failed", extra=extra)
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000
        extra.update({
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        })

        if response.status_code >= 500:
            logger.error("request.error", extra=extra)
        elif response.status_code >= 400:
            logger.warning("request.client_error", extra=extra)
        else:
            logger.info("request.completed", extra=extra)

        return response


# Initialize FastAPI app
# TODO: Configure custom domain with Cloud Endpoints for production deployment
# TODO: Set up Cloud Build trigger connection to repository for automated deployments
app = FastAPI(
    title="Watch Price Forecasting API",
    description="API for predicting luxury watch prices using machine learning models",
    version="1.0.0",
)
app.add_middleware(RequestLoggingMiddleware)

# Global service instance (will be initialized on first request)
_service: Optional[ForecastService] = None


def get_service() -> ForecastService:
    """Get or create the forecast service instance."""
    global _service
    if _service is None:
        environment = os.getenv("ENVIRONMENT", "local")
        bucket_name = os.getenv("MODEL_BUCKET", "timepiece-watch-models")

        if environment == "production":
            _service = CloudForecastService(bucket_name=bucket_name)
            logger.info(
                "Initialized CloudForecastService", extra={"bucket_name": bucket_name}
            )
        else:
            model_dir = "data/output/models"
            data_path = "data/output/featured_data.csv"
            _service = ForecastService(model_dir=model_dir, data_path=data_path)
            logger.info("Initialized local ForecastService")
    return _service


# Request/Response models
class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    watch_id: str = Field(
        default="Philippe_Nautilus_5711_Stainless_Steel_5711_1A",
        description="The ID of the watch to predict for"
    )
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
    watch_ids: List[str] = Field(
        default=[
            "Philippe_Nautilus_5711_Stainless_Steel_5711_1A",
            "Philippe_Aquanaut_5167_Stainless_Steel_5167A"
        ],
        description="List of watch IDs to predict for"
    )
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
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        service = get_service()
        health_data = service.health_check()
        return HealthResponse(**health_data)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("health.check_failed", extra={"error": str(exc)})
        return HealthResponse(status="unhealthy", error=str(exc))


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Generate a price prediction for a single watch."""
    try:
        service = get_service()
        result = service.predict_single(
            watch_id=request.watch_id,
            horizon=request.horizon,
            model_name=request.model_name,
        )
        logger.info(
            "prediction.success",
            extra={
                "watch_id": request.watch_id,
                "model_name": request.model_name,
                "horizon": request.horizon,
            },
        )
        return PredictResponse(**result)
    except ValueError as exc:
        logger.warning(
            "prediction.validation_error",
            extra={
                "watch_id": request.watch_id,
                "model_name": request.model_name,
                "horizon": request.horizon,
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "prediction.failure",
            extra={
                "watch_id": request.watch_id,
                "model_name": request.model_name,
                "horizon": request.horizon,
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Generate price predictions for multiple watches."""
    try:
        service = get_service()
        results = service.predict_multiple(
            watch_ids=request.watch_ids,
            horizon=request.horizon,
            model_name=request.model_name,
        )

        predictions = [PredictResponse(**result) for result in results]
        logger.info(
            "prediction.batch_success",
            extra={
                "model_name": request.model_name,
                "horizon": request.horizon,
                "requested": len(request.watch_ids),
                "returned": len(predictions),
            },
        )

        return BatchPredictResponse(
            predictions=predictions,
            total_requested=len(request.watch_ids),
            successful_predictions=len(predictions),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "prediction.batch_failure",
            extra={
                "model_name": request.model_name,
                "horizon": request.horizon,
                "requested": len(request.watch_ids),
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/watches", response_model=WatchListResponse)
async def get_available_watches():
    """Get list of available watch IDs."""
    try:
        service = get_service()
        watches = service.get_available_watches()
        return WatchListResponse(watches=watches, total_count=len(watches))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("watches.fetch_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/models", response_model=ModelListResponse)
async def get_available_models():
    """Get list of available model names."""
    try:
        service = get_service()
        models = service.get_available_models()
        return ModelListResponse(models=models, total_count=len(models))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("models.fetch_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Capture request validation errors with useful logging."""
    logger.warning(
        "request.validation_error",
        extra={
            "path": request.url.path,
            "error_count": len(exc.errors()),
        },
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors."""
    logger.warning(
        "request.not_found",
        extra={"method": request.method, "path": request.url.path},
    )
    return JSONResponse(status_code=404, content={"detail": "Endpoint not found"})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle uncaught errors."""
    logger.exception(
        "request.unhandled_error",
        extra={"method": request.method, "path": request.url.path},
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
