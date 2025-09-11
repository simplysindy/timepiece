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
    
    # Create temporal splits
    logger.info("🔄 Creating temporal train/val/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_temporal_splits(df, cfg)
    
    # Train models
    logger.info("🎯 Training models...")
    results = {}
    
    for model_name in cfg.training.models:
        logger.info(f"Training {model_name}...")
        try:
            model = train_model(model_name, X_train, y_train, X_val, y_val, cfg)
            metrics = evaluate_model(model, X_test, y_test)
            
            results[model_name] = {
                'model': model,
                'metrics': metrics,
                'success': True
            }
            
            logger.info(f"✅ {model_name} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {str(e)}")
            results[model_name] = {
                'model': None,
                'metrics': {},
                'success': False,
                'error': str(e)
            }
    
    # Save results
    if cfg.output.save_models:
        save_results(results, cfg)
    
    # Summary
    successful = [name for name, result in results.items() if result['success']]
    logger.info(f"🎉 Training complete! {len(successful)}/{len(cfg.training.models)} models successful")


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


def create_temporal_splits(df: pd.DataFrame, cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create train/val/test splits using temporal split per watch."""
    logger.info("Creating temporal splits per asset_id using timestamp...")

    # Ensure we have an asset identifier
    if 'asset_id' not in df.columns:
        raise ValueError("Asset identifier column 'asset_id' required for temporal splits")

    # Ensure we have a timestamp column (derive from 'date' if needed)
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            try:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            except Exception as e:
                raise ValueError(f"Failed to parse 'date' into 'timestamp': {e}")
        else:
            raise ValueError("No 'timestamp' or 'date' column found for temporal ordering")

    # Prepare features and target (raw date/timestamp are excluded in features module)
    X, y = prepare_features(df, cfg.data.target_column)

    train_data: list = []
    val_data: list = []
    test_data: list = []

    # Process each asset individually with strict temporal order
    for asset_id, grp in df[['asset_id', 'timestamp']].dropna().groupby('asset_id'):
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


def save_results(results: Dict[str, Dict], cfg: DictConfig) -> None:
    """Save training results and models."""
    
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    for model_name, result in results.items():
        if result['success'] and result['model'] is not None:
            model_path = output_dir / f"{model_name}.pkl"
            
            try:
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
                logger.info(f"Saved {model_name} to {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save {model_name}: {e}")
    
    # Save metrics summary
    metrics_summary = {}
    for model_name, result in results.items():
        if result['success']:
            metrics_summary[model_name] = result['metrics']
        else:
            metrics_summary[model_name] = {'error': result.get('error', 'Unknown error')}
    
    import json
    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    logger.info(f"Saved metrics summary to {metrics_path}")


if __name__ == "__main__":
    main()
