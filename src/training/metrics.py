"""
Evaluation metrics for watch price prediction models.

Streamlined metrics focused on key regression performance indicators.
"""

import logging
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics.
    
    Parameters:
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    -------
    Dict[str, float]
        Dictionary with metric names and values
    """
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        logger.warning("No valid predictions for metric calculation")
        return {}
    
    metrics = {}
    
    # Core regression metrics
    metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
    metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
    
    # Mean Absolute Percentage Error (handle division by zero)
    try:
        mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
        if not np.isnan(mape) and not np.isinf(mape):
            metrics['mape'] = mape * 100  # Convert to percentage
    except:
        logger.warning("MAPE calculation failed (likely due to zero values)")
    
    # Additional useful metrics
    residuals = y_true_clean - y_pred_clean
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    metrics['max_error'] = np.max(np.abs(residuals))
    
    # Directional accuracy (if enough samples)
    if len(y_true_clean) >= 2:
        directional_acc = calculate_directional_accuracy(y_true_clean, y_pred_clean)
        if not np.isnan(directional_acc):
            metrics['directional_accuracy'] = directional_acc
    
    logger.info(f"Calculated metrics for {len(y_true_clean)} samples:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    if 'mape' in metrics:
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Parameters:
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    -------
    float
        Directional accuracy as percentage
    """
    
    if len(y_true) < 2:
        return np.nan
    
    # Calculate actual and predicted directions
    actual_direction = np.diff(y_true) > 0
    predicted_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    correct_directions = np.sum(actual_direction == predicted_direction)
    total_directions = len(actual_direction)
    
    return (correct_directions / total_directions) * 100 if total_directions > 0 else np.nan


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """
    Format metrics for display.
    
    Parameters:
    ----------
    metrics : Dict[str, float]
        Raw metrics dictionary
    precision : int
        Number of decimal places
        
    Returns:
    -------
    Dict[str, str]
        Formatted metrics
    """
    
    formatted = {}
    
    for name, value in metrics.items():
        if np.isnan(value) or np.isinf(value):
            formatted[name] = "N/A"
        elif 'percentage' in name or 'accuracy' in name or 'mape' in name:
            # Percentage metrics
            formatted[name] = f"{value:.2f}%"
        else:
            # Regular metrics
            formatted[name] = f"{value:.{precision}f}"
    
    return formatted


def compare_models(model_metrics: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Compare multiple models and identify best performer.
    
    Parameters:
    ----------
    model_metrics : Dict[str, Dict[str, float]]
        Nested dictionary: {model_name: {metric: value}}
        
    Returns:
    -------
    Dict[str, str]
        Summary of best models per metric
    """
    
    if not model_metrics:
        return {}
    
    # Metrics where lower is better
    lower_better = ['mae', 'mse', 'rmse', 'mape', 'max_error', 'std_residual']
    # Metrics where higher is better  
    higher_better = ['r2', 'directional_accuracy']
    
    best_models = {}
    
    # Find best model for each metric
    for metric in ['rmse', 'mae', 'r2', 'mape']:
        if all(metric in metrics for metrics in model_metrics.values()):
            model_values = {name: metrics[metric] for name, metrics in model_metrics.items() 
                          if not np.isnan(metrics[metric])}
            
            if model_values:
                if metric in lower_better:
                    best_model = min(model_values, key=model_values.get)
                else:
                    best_model = max(model_values, key=model_values.get)
                
                best_models[metric] = f"{best_model} ({model_values[best_model]:.4f})"
    
    return best_models


def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
    """
    Validate prediction arrays.
    
    Parameters:
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    -------
    Dict[str, any]
        Validation results
    """
    
    validation = {
        'valid': True,
        'issues': [],
        'n_samples': len(y_true),
        'n_valid': 0
    }
    
    # Check array lengths
    if len(y_true) != len(y_pred):
        validation['valid'] = False
        validation['issues'].append(f"Array length mismatch: {len(y_true)} vs {len(y_pred)}")
    
    # Check for NaN values
    n_nan_true = np.isnan(y_true).sum()
    n_nan_pred = np.isnan(y_pred).sum()
    
    if n_nan_true > 0:
        validation['issues'].append(f"True values contain {n_nan_true} NaN values")
    
    if n_nan_pred > 0:
        validation['issues'].append(f"Predictions contain {n_nan_pred} NaN values")
    
    # Count valid samples
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    validation['n_valid'] = mask.sum()
    
    if validation['n_valid'] == 0:
        validation['valid'] = False
        validation['issues'].append("No valid prediction pairs found")
    
    # Check for infinite values
    if np.isinf(y_true).any():
        validation['issues'].append("True values contain infinite values")
    
    if np.isinf(y_pred).any():
        validation['issues'].append("Predictions contain infinite values")
    
    return validation