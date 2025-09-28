"""Cloud-enabled forecast service for watch price predictions."""

from __future__ import annotations

import logging
import os
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google.cloud import storage
from omegaconf import DictConfig

from ..training.features import prepare_features

logger = logging.getLogger(__name__)

ModelCache = Dict[Tuple[str, int], Any]


class CloudForecastService:
    """Cloud-enabled service class for generating watch price predictions."""

    def __init__(self, bucket_name: str = "timepiece-watch-models", local_cache_dir: str = "/tmp"):
        """Initialize the cloud forecast service.

        Parameters
        ----------
        bucket_name : str
            Name of the Cloud Storage bucket containing models and data.
        local_cache_dir : str
            Local directory to cache downloaded models and data.
        """
        self.bucket_name = bucket_name
        self.local_cache_dir = Path(local_cache_dir)
        self.models_dir = self.local_cache_dir / "models"
        self.data_dir = self.local_cache_dir / "data"
        self.model_cache: ModelCache = {}

        # Create cache directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Cloud Storage client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

        # Download and prepare data
        self._download_data()
        self._prepare_data()

        logger.info("CloudForecastService initialized with bucket: %s", bucket_name)

    def _download_data(self):
        """Download featured data from Cloud Storage if not cached."""
        local_data_path = self.data_dir / "featured_data.csv"

        if not local_data_path.exists():
            logger.info("Downloading featured_data.csv from Cloud Storage...")
            blob = self.bucket.blob("data/featured_data.csv")
            blob.download_to_filename(str(local_data_path))
            logger.info("Downloaded featured_data.csv to %s", local_data_path)

    def _prepare_data(self):
        """Load and prepare the dataset."""
        data_path = self.data_dir / "featured_data.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.df = pd.read_csv(data_path)
        if self.df.empty:
            raise ValueError("Inference dataset is empty")

        # Prepare feature matrix once
        self.feature_matrix, _ = prepare_features(self.df.copy(), target_column="target")

        logger.info("Prepared data with %d watches", len(self.df["watch_id"].unique()))

    def _download_model(self, model_name: str, horizon: int) -> Optional[Path]:
        """Download a model from Cloud Storage if not cached locally."""
        # Try different model file naming patterns
        possible_names = []
        if horizon:
            possible_names.extend([
                f"{model_name}__h{horizon}.pkl",
                f"{model_name}_h{horizon}.pkl",
            ])
        possible_names.append(f"{model_name}.pkl")

        for model_filename in possible_names:
            local_model_path = self.models_dir / model_filename

            # Check if already cached locally
            if local_model_path.exists():
                logger.debug("Using cached model: %s", local_model_path)
                return local_model_path

            # Try to download from Cloud Storage
            blob = self.bucket.blob(f"models/{model_filename}")
            if blob.exists():
                logger.info("Downloading model %s from Cloud Storage...", model_filename)
                blob.download_to_filename(str(local_model_path))
                logger.info("Downloaded model to %s", local_model_path)
                return local_model_path

        return None

    def _resolve_model_path(self, model_name: str, horizon: int) -> Tuple[Optional[Path], bool]:
        """Return best matching model path for the given horizon.

        Returns
        -------
        Tuple[path, is_horizon_specific]
            path is None when no model is discovered.
            is_horizon_specific indicates whether the located model encodes the
            requested horizon explicitly.
        """
        # Try horizon-specific models first
        if horizon:
            horizon_specific_path = self._download_model(model_name, horizon)
            if horizon_specific_path:
                return horizon_specific_path, True

        # Fall back to general model
        general_path = self._download_model(model_name, 0)  # 0 means no horizon
        if general_path:
            return general_path, False

        return None, False

    def _load_model(self, model_name: str, horizon: int) -> Optional[Any]:
        """Load a model from Cloud Storage with local caching."""
        cache_key = (model_name, horizon)

        if cache_key not in self.model_cache:
            model_path, is_specific = self._resolve_model_path(model_name, horizon)

            if model_path is None:
                logger.warning(
                    "Model file not found for %s with %d-day horizon in bucket %s",
                    model_name, horizon, self.bucket_name
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
        """Get list of available models in Cloud Storage.

        Returns
        -------
        List[str]
            List of available model names.
        """
        blobs = self.bucket.list_blobs(prefix="models/")
        model_names = set()

        for blob in blobs:
            if blob.name.endswith(".pkl"):
                name = Path(blob.name).stem
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
                "bucket_name": self.bucket_name,
                "cache_directory": str(self.local_cache_dir)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }