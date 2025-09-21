"""
Main training orchestration with Hydra configuration management.

Single entry point for watch price prediction model training with
proper temporal train/val/test splits per watch.

Usage:
    python -m src.training.training
    python -m src.training.training training.models=["lightgbm","xgboost"]
    python -m src.training.training training.horizons=[7] data.path="custom/path"
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

from .features import prepare_features, encode_categorical
from .metrics import calculate_metrics
from .models.statistical import LinearModel, RidgeModel, LassoModel
from .models.tree_based import RandomForestModel, XGBoostModel, LightGBMModel  
from .models.neural import LSTMModel

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="training")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    
    logger.info("🚀 Starting watch price prediction training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Load data
    logger.info("📊 Loading data...")
    df = load_data(cfg)
    
    price_column = cfg.data.get("price_column", "price(SGD)")
    horizons: List[int] = list(cfg.training.get("horizons", [])) or [1]

    overall_results: Dict[int, Dict[str, Dict[str, Any]]] = {}
    successful_models: List[str] = []

    for horizon in sorted(set(horizons)):
        logger.info("🔄 Preparing temporal splits for %s-day horizon", horizon)

        try:
            df_horizon, target_column = prepare_horizon_dataset(
                df, horizon, price_column, cfg.data.target_column
            )
        except ValueError as exc:
            logger.error("Skipping horizon %s: %s", horizon, exc)
            continue

        try:
            X_train, X_val, X_test, y_train, y_val, y_test = create_temporal_splits(
                df_horizon, cfg, target_column
            )
        except RuntimeError as exc:
            logger.error("Skipping horizon %s due to splitting error: %s", horizon, exc)
            continue

        logger.info("🎯 Training models for %s-day horizon...", horizon)
        horizon_results: Dict[str, Dict[str, Any]] = {}

        for model_name in cfg.training.models:
            logger.info("[%sd] Training %s...", horizon, model_name)
            try:
                model = train_model(model_name, X_train, y_train, X_val, y_val, cfg)
                metrics = evaluate_model(model, X_test, y_test)

                horizon_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'success': True,
                    'horizon': horizon,
                }

                successful_models.append(f"{model_name}@{horizon}d")
                logger.info(
                    "✅ [%sd] %s - RMSE: %.4f, MAE: %.4f, R²: %.4f",
                    horizon,
                    model_name,
                    metrics['rmse'],
                    metrics['mae'],
                    metrics['r2'],
                )

            except Exception as exc:
                logger.error("❌ [%sd] %s failed: %s", horizon, model_name, exc)
                horizon_results[model_name] = {
                    'model': None,
                    'metrics': {},
                    'success': False,
                    'error': str(exc),
                    'horizon': horizon,
                }

        overall_results[horizon] = horizon_results

    # Save results
    if cfg.output.save_models and overall_results:
        save_results(overall_results, cfg)

    total_attempts = len(cfg.training.models) * len(set(horizons))
    logger.info(
        "🎉 Training complete! %s/%s model runs successful",
        len(successful_models),
        total_attempts,
    )


def load_data(cfg: DictConfig) -> pd.DataFrame:
    """Load training dataset from unified file or combine individual processed files."""
    
    data_path = Path(cfg.data.path)
    
    # First try to load as unified file
    if data_path.exists():
        logger.info(f"Loading unified data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        # If unified file doesn't exist, create from individual processed files
        logger.info(f"Unified file not found at {data_path}, combining individual processed files")
        df = _combine_individual_files(cfg)
    
    # Basic validation
    if cfg.data.target_column not in df.columns:
        # Check if 'target' column exists (from processed files)
        if 'target' not in df.columns:
            raise ValueError(f"Target column '{cfg.data.target_column}' or 'target' not found in data")
        logger.info("Using 'target' column as target variable")
        cfg.data.target_column = 'target'
    
    # Check for asset identifier column
    asset_col = None
    for col in ['asset_id', 'watch_id']:
        if col in df.columns:
            asset_col = col
            break
    
    if asset_col is None:
        raise ValueError("Asset identifier column ('asset_id' or 'watch_id') required for temporal splits")
    
    # Standardize asset column name
    if asset_col != 'asset_id':
        df['asset_id'] = df[asset_col]
        logger.info(f"Using '{asset_col}' as asset identifier")
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    logger.info(f"Unique assets: {df['asset_id'].nunique()}")
    
    # Limit samples if requested
    if cfg.training.max_samples:
        if len(df) > cfg.training.max_samples:
            logger.info(f"Sampling {cfg.training.max_samples} from {len(df)} samples")
            df = df.sample(n=cfg.training.max_samples, random_state=cfg.training.random_state)
    
    return df


def _combine_individual_files(cfg: DictConfig) -> pd.DataFrame:
    """Combine individual processed watch files into unified dataset."""
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    logger.info(f"Combining individual files from {processed_dir}")
    
    # Find all CSV files in processed directory (excluding summary subdirectory)
    csv_files = [f for f in processed_dir.glob("*.csv") if f.is_file()]
    
    if not csv_files:
        raise FileNotFoundError(f"No processed CSV files found in {processed_dir}")
    
    logger.info(f"Found {len(csv_files)} processed watch files")
    
    combined_data = []
    successful_files = 0
    
    for file_path in csv_files:
        try:
            # Extract watch ID from filename
            watch_id = file_path.stem
            
            # Load the file
            df = pd.read_csv(file_path)
            
            # Skip empty files
            if df.empty:
                logger.warning(f"Skipping empty file: {file_path}")
                continue
            
            # Add watch identifier column if not present
            if 'asset_id' not in df.columns and 'watch_id' not in df.columns:
                df['asset_id'] = watch_id
            
            combined_data.append(df)
            successful_files += 1
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    if not combined_data:
        raise RuntimeError("No valid processed files could be loaded")
    
    # Combine all dataframes
    logger.info(f"Successfully loaded {successful_files} watch files")
    unified_df = pd.concat(combined_data, ignore_index=True)
    
    # Save the unified dataset for future use
    output_path = Path(cfg.data.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(output_path, index=False)
    logger.info(f"Created and saved unified dataset with {len(unified_df)} rows to {output_path}")

    return unified_df


def prepare_horizon_dataset(
    df: pd.DataFrame,
    horizon: int,
    price_column: str,
    base_target_column: str,
) -> Tuple[pd.DataFrame, str]:
    """Create a copy of the dataset with horizon-specific target column."""

    df_horizon = df.copy()

    if horizon < 1:
        raise ValueError("Horizon must be a positive integer")

    target_column = (
        base_target_column
        if horizon == 1 and base_target_column in df_horizon.columns
        else f"{base_target_column}_h{horizon}"
    )

    if target_column not in df_horizon.columns:
        if price_column not in df_horizon.columns:
            raise ValueError(
                f"Price column '{price_column}' required to derive target for horizon {horizon}"
            )

        df_horizon[target_column] = df_horizon[price_column].shift(-horizon)

    # Drop rows where the horizon target is not available yet
    df_horizon = df_horizon.dropna(subset=[target_column])

    return df_horizon, target_column


def create_temporal_splits(
    df: pd.DataFrame,
    cfg: DictConfig,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create train/val/test splits using temporal split per watch."""

    logger.info("Creating temporal splits per asset_id using timestamp...")

    df_local = df.copy()

    # Ensure we have an asset identifier
    if 'asset_id' not in df_local.columns:
        raise ValueError("Asset identifier column 'asset_id' required for temporal splits")

    # Ensure we have a timestamp column (derive from 'date' if needed)
    if 'timestamp' not in df_local.columns:
        if 'date' in df_local.columns:
            try:
                df_local['timestamp'] = pd.to_datetime(
                    df_local['date'], errors='coerce', utc=True
                )
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse 'date' into 'timestamp': {exc}"
                ) from exc
        else:
            raise ValueError("No 'timestamp' or 'date' column found for temporal ordering")

    # Prepare features and target (raw date/timestamp are excluded in features module)
    X, y = prepare_features(df_local, target_column)

    valid_mask = y.notna()
    if not valid_mask.any():
        raise RuntimeError(f"Target column '{target_column}' has no valid rows")

    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    df_local = df_local.loc[valid_mask]

    train_data: list = []
    val_data: list = []
    test_data: list = []

    # Process each asset individually with strict temporal order
    for asset_id, grp in df_local[['asset_id', 'timestamp']].dropna().groupby('asset_id'):
        order_idx = grp.sort_values('timestamp').index

        asset_X = X.loc[order_idx]
        asset_y = y.loc[order_idx]

        # Skip assets with insufficient data
        n_samples = len(asset_X)
        if n_samples < 20:  # configurable minimum
            logger.warning(f"Skipping asset {asset_id}: only {n_samples} samples")
            continue

        # Calculate split points (train | val | test by position)
        test_start = int(n_samples * (1 - cfg.training.test_size))
        val_start = int(n_samples * (1 - cfg.training.test_size - cfg.training.val_size))

        # Guard against edge cases where splits collapse
        if val_start <= 0 or test_start <= val_start:
            logger.warning(
                "Skipping asset %s: not enough samples after horizon filtering (val_start=%s, test_start=%s)",
                asset_id,
                val_start,
                test_start,
            )
            continue

        # Slice
        train_X = asset_X.iloc[:val_start]
        train_y = asset_y.iloc[:val_start]

        val_X = asset_X.iloc[val_start:test_start]
        val_y = asset_y.iloc[val_start:test_start]

        test_X = asset_X.iloc[test_start:]
        test_y = asset_y.iloc[test_start:]

        # Collect non-empty partitions
        if len(train_X) > 0:
            train_data.append((train_X, train_y))
        if len(val_X) > 0:
            val_data.append((val_X, val_y))
        if len(test_X) > 0:
            test_data.append((test_X, test_y))

    if not train_data or not val_data or not test_data:
        raise RuntimeError("Insufficient data to form train/val/test splits across assets")

    # Combine all assets
    X_train = pd.concat([data[0] for data in train_data], ignore_index=True)
    y_train = pd.concat([data[1] for data in train_data], ignore_index=True)

    X_val = pd.concat([data[0] for data in val_data], ignore_index=True)
    y_val = pd.concat([data[1] for data in val_data], ignore_index=True)

    X_test = pd.concat([data[0] for data in test_data], ignore_index=True)
    y_test = pd.concat([data[1] for data in test_data], ignore_index=True)

    logger.info("Temporal splits created:")
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                X_val: pd.DataFrame, y_val: pd.Series, cfg: DictConfig) -> Any:
    """Train a single model."""
    
    # Model factory
    model_classes = {
        'linear': LinearModel,
        'ridge': RidgeModel, 
        'lasso': LassoModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'lstm': LSTMModel
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get model hyperparameters
    model_params = cfg.models.get(model_name, {})
    model_params = OmegaConf.to_container(model_params, resolve=True)
    
    # Create and train model
    model_class = model_classes[model_name]
    model = model_class(**model_params)
    
    # Train with validation data for models that support it
    if hasattr(model, 'fit_with_validation'):
        model.fit_with_validation(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train)
    
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model performance."""
    
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test.values, y_pred)
    
    return metrics


def save_results(results: Dict[int, Dict[str, Dict]], cfg: DictConfig) -> None:
    """Save training results and models grouped by prediction horizon."""

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary: Dict[str, Dict[str, Any]] = {}

    for horizon, horizon_results in results.items():
        horizon_key = f"horizon_{horizon}d"
        metrics_summary[horizon_key] = {}

        for model_name, result in horizon_results.items():
            model_obj = result.get('model')
            success = result.get('success', False)

            if success and model_obj is not None:
                model_filename = f"{model_name}__h{horizon}.pkl"
                model_path = output_dir / model_filename

                try:
                    import pickle

                    with open(model_path, 'wb') as handle:
                        pickle.dump(model_obj, handle)
                    logger.info(
                        "Saved %s model for %s-day horizon to %s",
                        model_name,
                        horizon,
                        model_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to save %s model for horizon %s: %s",
                        model_name,
                        horizon,
                        exc,
                    )

            metrics_summary[horizon_key][model_name] = (
                result['metrics'] if success else {'error': result.get('error', 'Unknown error')}
            )

    import json

    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, 'w') as handle:
        json.dump(metrics_summary, handle, indent=2)

    logger.info("Saved metrics summary to %s", metrics_path)


if __name__ == "__main__":
    main()
