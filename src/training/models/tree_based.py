"""
Tree-based ensemble models for watch price prediction.

Includes RandomForest, XGBoost, and LightGBM implementations.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest model for time series forecasting."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = "sqrt"):
        """
        Initialize Random Forest model.
        
        Parameters:
        ----------
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split node
        min_samples_leaf : int
            Minimum samples required at leaf node  
        max_features : str
            Number of features to consider for best split
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit Random Forest model."""
        from sklearn.ensemble import RandomForestRegressor
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=42,
            n_jobs=-1
        )
        
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        logger.info(f"RandomForest fitted with {len(X)} samples, {self.n_estimators} trees")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_array)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_fitted:
            return None
        
        importance = self.model.feature_importances_
        return {f'feature_{i}': float(imp) for i, imp in enumerate(importance)}
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features
        }


class XGBoostModel:
    """XGBoost model with early stopping support."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 1.0,
                 colsample_bytree: float = 1.0, reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0, early_stopping_rounds: int = 10):
        """
        Initialize XGBoost model.
        
        Parameters:
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum depth of trees
        learning_rate : float
            Learning rate
        subsample : float
            Subsample ratio of training instances
        colsample_bytree : float
            Subsample ratio of columns for each tree
        reg_alpha : float
            L1 regularization
        reg_lambda : float
            L2 regularization
        early_stopping_rounds : int
            Early stopping rounds for validation
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise
        
        xgb_params: Dict[str, Any] = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        if self.early_stopping_rounds and self.early_stopping_rounds > 0:
            xgb_params['early_stopping_rounds'] = self.early_stopping_rounds

        self.model = xgb.XGBRegressor(**xgb_params)
        
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        logger.info(f"XGBoost fitted with {len(X)} samples")
    
    def fit_with_validation(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Fit XGBoost with early stopping using validation data."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise
        
        params: Dict[str, Any] = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        if self.early_stopping_rounds and self.early_stopping_rounds > 0:
            params['early_stopping_rounds'] = self.early_stopping_rounds

        self.model = xgb.XGBRegressor(**params)
        
        X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_array = X_val.values if hasattr(X_val, 'values') else X_val
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        
        eval_set = [(X_train_array, y_train_array), (X_val_array, y_val_array)]

        self.model.fit(
            X_train_array, y_train_array,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_fitted = True
        logger.info(f"XGBoost fitted with early stopping on {len(X_train)} training samples")
        
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            logger.info(f"Early stopped at iteration {self.model.best_iteration}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_array)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_fitted:
            return None
        
        importance = self.model.feature_importances_
        return {f'feature_{i}': float(imp) for i, imp in enumerate(importance)}
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'early_stopping_rounds': self.early_stopping_rounds
        }


class LightGBMModel:
    """LightGBM model with categorical feature support."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = -1, num_leaves: int = 31,
                 early_stopping_rounds: int = 10):
        """
        Initialize LightGBM model.
        
        Parameters:
        ----------
        n_estimators : int
            Number of boosting iterations
        learning_rate : float
            Learning rate
        max_depth : int
            Maximum tree depth (-1 for no limit)
        num_leaves : int
            Maximum number of leaves in one tree
        early_stopping_rounds : int
            Early stopping rounds for validation
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. Install with: pip install lightgbm")
            raise
        
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        # Convert to numpy arrays for sklearn compatibility
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X_array, y_array)
        self.is_fitted = True
        logger.info(f"LightGBM fitted with {len(X)} samples")
    
    def fit_with_validation(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Fit LightGBM with early stopping using validation data."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. Install with: pip install lightgbm")
            raise
        
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_array = X_val.values if hasattr(X_val, 'values') else X_val
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        
        eval_set = [(X_train_array, y_train_array), (X_val_array, y_val_array)]
        
        self.model.fit(
            X_train_array, y_train_array,
            eval_set=eval_set,
            eval_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
        )
        
        self.is_fitted = True
        logger.info(f"LightGBM fitted with early stopping on {len(X_train)} training samples")
        
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            logger.info(f"Early stopped at iteration {self.model.best_iteration}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_array)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_fitted:
            return None
        
        importance = self.model.feature_importances_
        return {f'feature_{i}': float(imp) for i, imp in enumerate(importance)}
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'early_stopping_rounds': self.early_stopping_rounds
        }
