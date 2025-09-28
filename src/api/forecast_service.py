"""Forecast service for watch price predictions."""

from __future__ import annotations

import logging
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig

from ..training.features import prepare_features

logger = logging.getLogger(__name__)

ModelCache = Dict[Tuple[str, int], Any]


class ForecastService:
    """Service class for generating watch price predictions."""

    def __init__(self, model_dir: str | Path, data_path: str | Path):
        """Initialize the forecast service.

        Parameters
        ----------
        model_dir : str | Path
            Directory containing pickled models saved by the training pipeline.
        data_path : str | Path
            Path to the feature-rich dataset (same schema as training input).
        """
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        self.model_cache: ModelCache = {}

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load the dataset once during initialization
        self.df = pd.read_csv(self.data_path)
        if self.df.empty:
            raise ValueError("Inference dataset is empty")

        # Prepare feature matrix once
        self.feature_matrix, _ = prepare_features(self.df.copy(), target_column="target")

        logger.info("ForecastService initialized with %d watches", len(self.df["watch_id"].unique()))

    def _resolve_model_path(self, model_name: str, horizon: int) -> Tuple[Optional[Path], bool]:
        """Return best matching model path for the given horizon.

        Returns
        -------
        Tuple[path, is_horizon_specific]
            path is None when no model is discovered.
            is_horizon_specific indicates whether the located model encodes the
            requested horizon explicitly.
        """
        candidates = []
        if horizon:
            candidates.extend([
                (self.model_dir / f"{model_name}__h{horizon}.pkl", True),
                (self.model_dir / f"{model_name}_h{horizon}.pkl", True),
            ])

        candidates.append((self.model_dir / f"{model_name}.pkl", False))

        for path, is_specific in candidates:
            if path.exists():
                return path, is_specific

        return None, False

    def _load_model(self, model_name: str, horizon: int) -> Optional[Any]:
        """Load a model from disk with caching."""
        cache_key = (model_name, horizon)

        if cache_key not in self.model_cache:
            model_path, is_specific = self._resolve_model_path(model_name, horizon)

            if model_path is None:
                logger.warning(
                    "Model file not found for %s with %d-day horizon in %s",
                    model_name, horizon, self.model_dir
                )
                return None

            with open(model_path, "rb") as fh:
                self.model_cache[cache_key] = pickle.load(fh)

            logger.info(
                "Loaded model %s for %d-day horizon from %s",
                model_name, horizon, model_path
            )

        return self.model_cache.get(cache_key)

    def predict_single(
        self,
        watch_id: str,
        horizon: int = 7,
        model_name: str = "lightgbm",
        asset_column: str = "watch_id",
        timestamp_column: str = "date"
    ) -> Dict[str, Any]:
        """Generate prediction for a single watch.

        Parameters
        ----------
        watch_id : str
            The ID of the watch to predict for.
        horizon : int, default=7
            Number of days ahead to predict.
        model_name : str, default="lightgbm"
            Name of the model to use for prediction.
        asset_column : str, default="watch_id"
            Column that uniquely identifies an asset/watch in the dataset.
        timestamp_column : str, default="date"
            Column containing the timestamp used to order observations.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing prediction results.
        """
        # Validate inputs
        if watch_id not in self.df[asset_column].values:
            raise ValueError(f"Watch ID '{watch_id}' not found in dataset")

        # Load the model
        model = self._load_model(model_name, horizon)
        if model is None:
            raise ValueError(f"Model '{model_name}' not available for {horizon}-day horizon")

        # Prepare data for this specific watch
        self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column])
        self.df["asset_id"] = self.df[asset_column]

        # Get the latest observation for this watch
        watch_data = self.df[self.df[asset_column] == watch_id].copy()
        if watch_data.empty:
            raise ValueError(f"No data found for watch '{watch_id}'")

        latest_row = watch_data.sort_values(timestamp_column).iloc[-1]
        latest_idx = latest_row.name

        # Get features for the latest observation
        X_latest = self.feature_matrix.loc[[latest_idx]]

        # Handle feature selection if model has specific requirements
        features = X_latest
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            missing = [col for col in expected if col not in features.columns]
            if missing:
                raise ValueError(f"Model '{model_name}' expects missing features: {missing}")
            features = features[expected]

        # Generate prediction
        prediction = model.predict(features)[0]

        # Calculate dates
        feature_date = latest_row[timestamp_column]
        prediction_date = feature_date + timedelta(days=horizon)

        # Prepare result
        result = {
            "watch_id": watch_id,
            "model_name": model_name,
            "horizon_days": horizon,
            "feature_date": feature_date.date().isoformat(),
            "prediction_date": prediction_date.date().isoformat(),
            "prediction": float(prediction),
        }

        # Add current price if available
        if "price(SGD)" in latest_row:
            result["current_price"] = float(latest_row["price(SGD)"])

        logger.info(
            "Generated prediction for %s: %.2f (horizon=%d days, model=%s)",
            watch_id, prediction, horizon, model_name
        )

        return result

    def predict_multiple(
        self,
        watch_ids: List[str],
        horizon: int = 7,
        model_name: str = "lightgbm",
        asset_column: str = "watch_id",
        timestamp_column: str = "date"
    ) -> List[Dict[str, Any]]:
        """Generate predictions for multiple watches.

        Parameters
        ----------
        watch_ids : List[str]
            List of watch IDs to predict for.
        horizon : int, default=7
            Number of days ahead to predict.
        model_name : str, default="lightgbm"
            Name of the model to use for prediction.
        asset_column : str, default="watch_id"
            Column that uniquely identifies an asset/watch in the dataset.
        timestamp_column : str, default="date"
            Column containing the timestamp used to order observations.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing prediction results.
        """
        results = []
        for watch_id in watch_ids:
            try:
                result = self.predict_single(
                    watch_id=watch_id,
                    horizon=horizon,
                    model_name=model_name,
                    asset_column=asset_column,
                    timestamp_column=timestamp_column
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to predict for watch %s: %s", watch_id, str(e))
                # Continue with other watches
                continue

        return results

    def get_available_watches(self, asset_column: str = "watch_id") -> List[str]:
        """Get list of available watch IDs in the dataset.

        Parameters
        ----------
        asset_column : str, default="watch_id"
            Column that uniquely identifies an asset/watch in the dataset.

        Returns
        -------
        List[str]
            List of available watch IDs.
        """
        return sorted(self.df[asset_column].unique().tolist())

    def get_available_models(self) -> List[str]:
        """Get list of available models in the model directory.

        Returns
        -------
        List[str]
            List of available model names.
        """
        model_files = list(self.model_dir.glob("*.pkl"))
        model_names = set()

        for model_file in model_files:
            name = model_file.stem
            # Remove horizon suffix if present
            if "__h" in name:
                name = name.split("__h")[0]
            elif "_h" in name:
                name = name.split("_h")[0]
            model_names.add(name)

        return sorted(list(model_names))

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the service.

        Returns
        -------
        Dict[str, Any]
            Health check results.
        """
        try:
            available_watches = len(self.get_available_watches())
            available_models = len(self.get_available_models())

            return {
                "status": "healthy",
                "available_watches": available_watches,
                "available_models": available_models,
                "model_directory": str(self.model_dir),
                "data_path": str(self.data_path)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }