"""Smoke tests for the FastAPI watch forecasting service."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.api import main


class StubForecastService:
    """Lightweight stand-in for ForecastService used in tests."""

    def __init__(self) -> None:
        self.raise_value_error = False

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "available_watches": 2,
            "available_models": 2,
            "model_directory": "data/output/models",
            "data_path": "data/output/featured_data.csv",
        }

    def predict_single(self, watch_id: str, horizon: int, model_name: str) -> Dict[str, Any]:
        if self.raise_value_error:
            raise ValueError("invalid watch id provided")

        return {
            "watch_id": watch_id,
            "model_name": model_name,
            "horizon_days": horizon,
            "feature_date": "2024-01-01",
            "prediction_date": "2024-01-08",
            "prediction": 123.45,
            "current_price": 120.00,
        }

    def predict_multiple(self, watch_ids: List[str], horizon: int, model_name: str) -> List[Dict[str, Any]]:
        return [
            self.predict_single(watch_id=watch_id, horizon=horizon, model_name=model_name)
            for watch_id in watch_ids
        ]

    def get_available_watches(self) -> List[str]:
        return ["watch_a", "watch_b"]

    def get_available_models(self) -> List[str]:
        return ["lightgbm", "ridge"]


@pytest.fixture()
def stub_service(monkeypatch) -> StubForecastService:
    stub = StubForecastService()

    monkeypatch.setattr(main, "_service", stub)
    monkeypatch.setattr(main, "get_service", lambda: stub)

    return stub


@pytest.fixture()
def client(stub_service: StubForecastService) -> TestClient:
    return TestClient(main.app)


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Watch Price Forecasting API"
    assert body["docs"] == "/docs"


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["available_watches"] == 2


def test_predict_success(client: TestClient) -> None:
    payload = {"watch_id": "watch_a", "horizon": 7, "model_name": "lightgbm"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["watch_id"] == "watch_a"
    assert body["model_name"] == "lightgbm"
    assert body["horizon_days"] == 7


def test_predict_validation_error(client: TestClient, stub_service: StubForecastService) -> None:
    stub_service.raise_value_error = True
    payload = {"watch_id": "bad_watch", "horizon": 7, "model_name": "lightgbm"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "invalid watch" in response.json()["detail"]


def test_batch_predict(client: TestClient) -> None:
    payload = {
        "watch_ids": ["watch_a", "watch_b"],
        "horizon": 14,
        "model_name": "ridge",
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["total_requested"] == 2
    assert body["successful_predictions"] == 2
    assert all(result["model_name"] == "ridge" for result in body["predictions"])


def test_models_listing(client: TestClient) -> None:
    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == ["lightgbm", "ridge"]


def test_watches_listing(client: TestClient) -> None:
    response = client.get("/watches")

    assert response.status_code == 200
    body = response.json()
    assert body["watches"] == ["watch_a", "watch_b"]
