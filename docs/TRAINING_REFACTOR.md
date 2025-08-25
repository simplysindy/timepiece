# Training Pipeline Refactor

## Overview

The ML training pipeline has been successfully refactored from a complex 2,500+ line structure to a simplified ~600 line implementation focused on watch price prediction with proper temporal train/val/test splits.

## New Directory Structure

```
src/
├── training/
│   ├── __init__.py
│   ├── training.py          # Main training orchestration with Hydra
│   ├── models/
│   │   ├── __init__.py
│   │   ├── statistical.py    # Linear, Ridge, Lasso, ARIMA, SARIMA
│   │   ├── tree_based.py     # RandomForest, XGBoost, LightGBM  
│   │   └── neural.py         # LSTM and other neural models
│   ├── features.py       # Feature preprocessing (simplified from feature_handler.py)
│   └── metrics.py        # Evaluation metrics (simplified from metrics.py)
conf/
└── training.yaml # Hydra configuration file
```

## Key Improvements

### 1. **Simplified Architecture**
- **Before**: 2,500+ lines across multiple complex files with inheritance hierarchies
- **After**: ~600 lines in 6 main files with simple function-based approach
- **Removed**: Complex abstractions like `ModelTrainer`, `UnifiedModelTrainer` classes
- **Replaced**: With direct implementation using functions instead of classes where possible

### 2. **Proper Data Splitting** ⚠️ **CRITICAL IMPROVEMENT**
- **Before**: Risk of data leakage with random splits across watches
- **After**: Temporal splits for each watch individually:
  - For each watch: use first 70% for training, next 10% for validation, last 20% for test
  - This ensures no future data leaks into training
  - All watches contribute to train/val/test sets with their respective temporal splits

### 3. **Hydra Integration**
- Full configuration management with CLI overrides
- Single command execution: `python -m src.training.training`
- Easy parameter tuning via command line or config file

### 4. **Streamlined Models**
- **statistical.py**: Combined linear models (Linear, Ridge, Lasso) with time series models (ARIMA, SARIMA)
- **tree_based.py**: Ensemble models (RandomForest, XGBoost, LightGBM) with early stopping support
- **neural.py**: Simplified LSTM implementation
- Each model is a simple class with `fit()` and `predict()` methods
- No complex inheritance - each model is self-contained

### 5. **Simplified Feature Processing**
- Extracted core functionality from complex `feature_handler.py`
- Removed global singleton pattern
- Simple functions: `prepare_features()`, `encode_categorical()`
- Focus only on watch data features with simple label encoding

### 6. **Essential Metrics**
- Focus on key metrics: RMSE, MAE, R², MAPE, directional accuracy
- Simple functions like `calculate_metrics(y_true, y_pred)`
- Removed complex rolling metrics and statistical tests

## Usage

### Basic Training
```bash
# Train default models (lightgbm, xgboost, random_forest)
python -m src.training.training

# Train specific models
python -m src.training.training training.models=["lightgbm","xgboost"]

# Use different horizons
python -m src.training.training training.horizons=[7]

# Custom data path
python -m src.training.training data.path="custom/path/data.csv"
```

### Configuration Override Examples
```bash
# Customize model hyperparameters
python -m src.training.training models.lightgbm.n_estimators=200

# Adjust data splits
python -m src.training.training training.test_size=0.3 training.val_size=0.15

# Limit training data
python -m src.training.training training.max_samples=10000
```

## Configuration File (conf/training.yaml)

```yaml
# Data settings
data:
  path: "data/output/featured_data.csv"
  target_column: "target"
  
# Training settings  
training:
  horizons: [7, 14, 30]
  models: ["lightgbm", "xgboost", "random_forest"]
  test_size: 0.2      # Last 20% of each watch's data
  val_size: 0.1       # 10% before test set
  random_state: 42
  max_samples: null   # Set to limit data size
  
# Model hyperparameters
models:
  lightgbm:
    n_estimators: 100
    learning_rate: 0.1
    early_stopping_rounds: 10
  # ... other model configs
```

## Data Requirements

The pipeline expects a unified dataset with:
- `target` column: The prediction target
- `asset_id` column: Watch identifier for temporal splits
- Feature columns: Technical indicators, price lags, etc.
- Categorical features: Should be pre-encoded (e.g., `brand_encoded`, `model_encoded`)

## Testing

Run the test suite to verify the pipeline:
```bash
python test_training_pipeline.py
```

## Key Benefits

1. **No Data Leakage**: Proper temporal splits ensure no future information leaks into training
2. **Maintainable**: Simplified codebase that's easy to understand and modify
3. **Flexible**: Hydra configuration allows easy experimentation
4. **Fast**: Reduced complexity leads to faster training and debugging
5. **Focused**: Optimized specifically for watch price prediction

## Migration Notes

- **Removed**: All complex validation classes (`TimeSeriesValidator`, `ValidationResult`, etc.)
- **Removed**: Hyperparameter tuning functionality (can add later if needed)
- **Removed**: Multi-asset training support - focus on single unified model
- **Removed**: Complex logging configs - uses simple Python logging
- **Removed**: Backward compatibility code and base.py file

## Function Signatures

### training.py
```python
def load_data(cfg: DictConfig) -> pd.DataFrame
def create_temporal_splits(df: pd.DataFrame, cfg: DictConfig) -> Tuple[...]
def train_model(model_name: str, X_train, y_train, X_val, y_val, cfg) -> Model
def evaluate_model(model, X_test, y_test) -> Dict[str, float]
```

### features.py
```python
def prepare_features(df: pd.DataFrame, target_column: str = 'target') -> Tuple[pd.DataFrame, pd.Series]
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame
```

### metrics.py
```python
def calculate_metrics(y_true, y_pred) -> Dict[str, float]
```

The refactored pipeline is now ready for production use with proper temporal validation and simplified maintenance.