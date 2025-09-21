"""Lean inference helpers for watch price models."""

from __future__ import annotations

import logging
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..training.features import prepare_features


logger = logging.getLogger(__name__)

ModelCache = Dict[Tuple[str, int], Any]


def _resolve_model_path(
    model_root: Path,
    model_name: str,
    horizon: int,
) -> Tuple[Optional[Path], bool]:
    """Return best matching model path for the given horizon.

    Returns
    -------
    Tuple[path, is_horizon_specific]
        path is None when no model is discovered.
        is_horizon_specific indicates whether the located model encodes the
        requested horizon explicitly (used to warn about fallbacks).
    """

    candidates = []
    if horizon:
        candidates.extend(
            [
                (model_root / f"{model_name}__h{horizon}.pkl", True),
                (model_root / f"{model_name}_h{horizon}.pkl", True),
            ]
        )

    candidates.append((model_root / f"{model_name}.pkl", False))

    for path, is_specific in candidates:
        if path.exists():
            return path, is_specific

    return None, False


def generate_predictions(
    data_path: str | Path,
    model_dir: str | Path,
    models: Iterable[str] | None = None,
    horizons: Iterable[int] | None = None,
    asset_column: str = "watch_id",
    timestamp_column: str = "date",
    target_column: str = "target",
    include_actuals: bool = True,
) -> pd.DataFrame:
    """Create forward predictions for the latest observation of each asset.

    Parameters
    ----------
    data_path:
        Path to the feature-rich dataset (same schema as training input).
    model_dir:
        Directory containing pickled models saved by the training pipeline.
    models:
        Iterable of model names to score. Defaults to ["ridge"].
    horizons:
        Iterable of day offsets to add to the feature timestamp when labelling
        prediction dates. Defaults to [7].
    asset_column:
        Column that uniquely identifies an asset/watch in the dataset.
    timestamp_column:
        Column containing the timestamp used to order observations.
    target_column:
        Column used as the training target. Included in outputs when available.
    include_actuals:
        When True, copy the target value from the feature row into the output
        for reference (may be NaN for the latest row).

    Returns
    -------
    pd.DataFrame
        A tidy frame with one row per asset per model containing
        `prediction`, `feature_date`, and `prediction_date`.
    """

    if models:
        model_names = list(dict.fromkeys(list(models)))
    else:
        model_names = ["ridge"]

    if horizons:
        horizon_list = sorted({int(h) for h in horizons})
    else:
        horizon_list = [7]

    data_source_path = Path(to_absolute_path(str(data_path)))
    model_root = Path(to_absolute_path(str(model_dir)))

    if not data_source_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_source_path}")
    if not model_root.exists():
        raise FileNotFoundError(f"Model directory not found: {model_root}")

    df = pd.read_csv(data_source_path)
    if df.empty:
        raise ValueError("Inference dataset is empty")

    if asset_column not in df.columns:
        if "asset_id" in df.columns:
            asset_column = "asset_id"
        else:
            raise ValueError(
                f"Asset column '{asset_column}' not found. Update inference config."
            )

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' not found. Update inference config."
        )

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df["asset_id"] = df[asset_column]

    feature_matrix, _ = prepare_features(df.copy(), target_column=target_column)

    latest_idx = (
        df.sort_values(["asset_id", timestamp_column])
        .groupby("asset_id")
        .tail(1)
        .index
    )

    X_latest = feature_matrix.loc[latest_idx]
    meta = df.loc[latest_idx, :]

    results_rows = []
    model_cache: ModelCache = {}
    multi_horizon_requested = len(horizon_list) > 1

    for model_name in model_names:
        for horizon_days in horizon_list:
            cache_key = (model_name, horizon_days)

            if cache_key not in model_cache:
                model_path, is_specific = _resolve_model_path(
                    model_root, model_name, horizon_days
                )

                if model_path is None:
                    logger.warning(
                        "Skipping %s-day horizon for %s: model file not found in %s",
                        horizon_days,
                        model_name,
                        model_root,
                    )
                    continue

                if not is_specific and multi_horizon_requested:
                    logger.warning(
                        "Skipping %s-day horizon for %s: horizon-specific model missing (expected %s__h%s.pkl)",
                        horizon_days,
                        model_name,
                        model_name,
                        horizon_days,
                    )
                    continue

                with open(model_path, "rb") as fh:
                    model_cache[cache_key] = pickle.load(fh)

                logger.info(
                    "Loaded model %s for %s-day horizon from %s",
                    model_name,
                    horizon_days,
                    model_path,
                )

            model = model_cache.get(cache_key)
            if model is None:
                continue

            features = X_latest
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                missing = [col for col in expected if col not in features.columns]
                if missing:
                    raise ValueError(
                        f"Model '{model_name}' expects missing features: {missing}"
                    )
                features = features[expected]

            predictions = model.predict(features)

            for idx, prediction in zip(features.index, predictions):
                row = meta.loc[idx]
                feature_date = row[timestamp_column]
                prediction_date = feature_date + timedelta(days=horizon_days)

                record = {
                    "model_name": model_name,
                    "asset_id": row["asset_id"],
                    "horizon_days": horizon_days,
                    "feature_date": feature_date.date().isoformat(),
                    "prediction_date": prediction_date.date().isoformat(),
                    "prediction": float(prediction),
                }

                if "price(SGD)" in row:
                    record["current_price"] = float(row["price(SGD)"])

                # For true forward predictions, we shouldn't include actual targets
                # since we can't know future prices. Only include actuals for backtesting.
                # This can be controlled via include_actuals parameter.

                results_rows.append(record)

            logger.info(
                "Generated %s predictions with %s for horizon %s days",
                len(predictions),
                model_name,
                horizon_days,
            )

    if not results_rows:
        logger.warning("No predictions generated. Check model names and inputs.")
        return pd.DataFrame()

    return pd.DataFrame(results_rows)


@hydra.main(version_base=None, config_path="../../conf", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Hydra entry-point to run inference from the CLI."""

    logger.info("🚀 Starting inference run")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    predictions = generate_predictions(
        data_path=cfg.data.path,
        model_dir=cfg.inference.model_dir,
        models=cfg.inference.models,
        horizons=cfg.inference.prediction_horizon_days,
        asset_column=cfg.data.asset_column,
        timestamp_column=cfg.data.timestamp_column,
        target_column=cfg.data.target_column,
        include_actuals=cfg.inference.get("include_actuals", True),
    )

    if predictions.empty:
        return

    output_path = cfg.inference.get("output_path")
    if output_path:
        output_file = Path(to_absolute_path(str(output_path)))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_file, index=False)
        logger.info("Saved predictions to %s", output_file)
    else:
        logger.info("No output path configured; skipping file write.")


if __name__ == "__main__":
    main()
