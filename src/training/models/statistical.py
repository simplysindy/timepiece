"""
Statistical models for watch price prediction.

Includes linear models and time series models (ARIMA, SARIMA).
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso

logger = logging.getLogger(__name__)


class LinearModel:
    """Simple linear regression model."""
    
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize Linear Regression model.
        
        Parameters:
        ----------
        fit_intercept : bool
            Whether to calculate intercept
        """
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model."""
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        logger.info(f"Linear model fitted with {len(X)} samples")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_array)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get model coefficients as feature importance."""
        if not self.is_fitted:
            return None
        
        # Use coefficient magnitudes as importance
        importance = {f'feature_{i}': abs(coef) for i, coef in enumerate(self.model.coef_)}
        return importance
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {'fit_intercept': self.fit_intercept}


class RidgeModel:
    """Ridge regression with L2 regularization."""
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, max_iter: int = 1000):
        """
        Initialize Ridge regression model.
        
        Parameters:
        ----------
        alpha : float
            Regularization strength
        fit_intercept : bool
            Whether to calculate intercept
        max_iter : int
            Maximum iterations
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model."""
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        logger.info(f"Ridge model fitted with {len(X)} samples, alpha={self.alpha}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_array)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get regularized coefficients as feature importance."""
        if not self.is_fitted:
            return None
        
        importance = {f'feature_{i}': abs(coef) for i, coef in enumerate(self.model.coef_)}
        return importance
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter
        }


class LassoModel:
    """Lasso regression with L1 regularization and feature selection."""
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, max_iter: int = 5000):
        """
        Initialize Lasso regression model.
        
        Parameters:
        ----------
        alpha : float
            Regularization strength
        fit_intercept : bool
            Whether to calculate intercept
        max_iter : int
            Maximum iterations
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model."""
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        
        # Log sparsity info
        n_selected = np.sum(np.abs(self.model.coef_) > 1e-10)
        logger.info(f"Lasso model fitted with {len(X)} samples, alpha={self.alpha}")
        logger.info(f"Selected {n_selected}/{len(self.model.coef_)} features")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_array)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get non-zero coefficients as feature importance."""
        if not self.is_fitted:
            return None
        
        # Only return features with non-zero coefficients
        importance = {}
        for i, coef in enumerate(self.model.coef_):
            if abs(coef) > 1e-10:
                importance[f'feature_{i}'] = abs(coef)
        
        return importance
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter
        }


class ARIMAModel:
    """Simple ARIMA model for time series forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize ARIMA model.
        
        Parameters:
        ----------
        order : Tuple[int, int, int]
            ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.model_fit = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit ARIMA model using target series."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            logger.error("statsmodels not installed. Install with: pip install statsmodels")
            raise
        
        # ARIMA uses only target variable, not features
        ts_data = y.dropna()
        
        if len(ts_data) < 10:
            raise ValueError("Insufficient data for ARIMA (need at least 10 observations)")
        
        try:
            self.model = ARIMA(ts_data, order=self.order)
            self.model_fit = self.model.fit()
            self.is_fitted = True
            logger.info(f"ARIMA{self.order} fitted with {len(ts_data)} samples")
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ARIMA predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Forecast the requested number of steps
        steps = len(X)
        forecast = self.model_fit.forecast(steps=steps)
        
        if hasattr(forecast, 'values'):
            return forecast.values
        return np.array(forecast)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """ARIMA parameters as 'importance'."""
        if not self.is_fitted:
            return None
        
        params = self.model_fit.params
        return {f'param_{i}': abs(float(val)) for i, val in enumerate(params)}
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get ARIMA hyperparameters."""
        return {'order': self.order}


class SARIMAModel:
    """Simple SARIMA model for seasonal time series forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        """
        Initialize SARIMA model.
        
        Parameters:
        ----------
        order : Tuple[int, int, int]
            Non-seasonal ARIMA order (p, d, q)
        seasonal_order : Tuple[int, int, int, int]
            Seasonal order (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit SARIMA model."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            logger.error("statsmodels not installed. Install with: pip install statsmodels")
            raise
        
        ts_data = y.dropna()
        
        # Need more data for seasonal models
        min_obs = max(24, 2 * self.seasonal_order[3])
        if len(ts_data) < min_obs:
            raise ValueError(f"Insufficient data for SARIMA (need at least {min_obs} observations)")
        
        try:
            self.model = SARIMAX(ts_data, order=self.order, seasonal_order=self.seasonal_order)
            self.model_fit = self.model.fit(disp=False)
            self.is_fitted = True
            logger.info(f"SARIMA{self.order}x{self.seasonal_order} fitted with {len(ts_data)} samples")
        except Exception as e:
            logger.error(f"SARIMA fitting failed: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make SARIMA predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        steps = len(X)
        forecast = self.model_fit.forecast(steps=steps)
        
        if hasattr(forecast, 'values'):
            return forecast.values
        return np.array(forecast)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """SARIMA parameters as 'importance'."""
        if not self.is_fitted:
            return None
        
        params = self.model_fit.params
        return {str(param): abs(float(val)) for param, val in params.items()}
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get SARIMA hyperparameters."""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }