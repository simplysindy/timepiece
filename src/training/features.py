"""
Feature preprocessing for watch price prediction models.

Simplified feature handling focused on watch data preparation.
"""

import logging
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame, target_column: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for training, return X and y.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe with all columns
    target_column : str
        Name of target column
        
    Returns:
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target series y
    """
    
    logger.info(f"Preparing features from {len(df)} samples")
    
    # Separate target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    
    y = df[target_column].copy()
    
    # Define columns to exclude from features
    excluded_columns = {
        target_column,
        'asset_id',   # Keep for grouping but not as feature
        'brand',      # Use encoded version instead
        'model',      # Use encoded version instead
        'date',       # Do not use raw date strings as features
        'timestamp',  # Keep time explicit but not as raw numeric feature
    }

    # Drop any future-looking targets from the feature set to avoid leakage
    excluded_columns.update(col for col in df.columns if col.startswith('target'))
    
    # Select feature columns
    feature_columns = [col for col in df.columns if col not in excluded_columns]
    X = df[feature_columns].copy()
    
    # Encode categorical features
    X = encode_categorical(X)
    
    # Handle missing values
    # Fill numeric columns with 0
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].fillna(0)
    
    # Convert all to numeric (sklearn compatibility)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    logger.info(f"Prepared features:")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Missing values: {X.isna().sum().sum()}")
    logger.info(f"  Feature columns: {len(X.columns)}")
    
    return X, y


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple categorical encoding using label encoding.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    -------
    pd.DataFrame
        Dataframe with encoded categorical features
    """
    
    df_encoded = df.copy()
    
    # Identify categorical columns (object type or low cardinality integers)
    categorical_columns = []
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            categorical_columns.append(col)
        elif df_encoded[col].dtype in ['int64', 'int32'] and df_encoded[col].nunique() < 50:
            categorical_columns.append(col)
    
    logger.info(f"Encoding {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    # Apply label encoding
    for col in categorical_columns:
        if df_encoded[col].dtype == 'object':
            # Handle string categories
            le = LabelEncoder()
            
            # Handle missing values
            non_null_mask = df_encoded[col].notna()
            if non_null_mask.any():
                df_encoded.loc[non_null_mask, col] = le.fit_transform(df_encoded.loc[non_null_mask, col])
                df_encoded.loc[~non_null_mask, col] = -1  # Missing value indicator
            else:
                df_encoded[col] = -1
        
        # Convert to numeric
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(-1)
    
    return df_encoded


def get_feature_names(X: pd.DataFrame) -> List[str]:
    """
    Get list of feature names.
    
    Parameters:
    ----------
    X : pd.DataFrame
        Feature dataframe
        
    Returns:
    -------
    List[str]
        List of feature names
    """
    return X.columns.tolist()


def validate_features(X: pd.DataFrame, y: pd.Series) -> Dict[str, any]:
    """
    Validate feature matrix and target.
    
    Parameters:
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target series
        
    Returns:
    -------
    Dict[str, any]
        Validation results
    """
    
    validation = {
        'n_samples': len(X),
        'n_features': len(X.columns),
        'target_range': [float(y.min()), float(y.max())],
        'missing_features': X.isna().sum().sum(),
        'missing_target': y.isna().sum(),
        'feature_types': X.dtypes.value_counts().to_dict(),
        'valid': True,
        'warnings': []
    }
    
    # Check for issues
    if validation['missing_features'] > 0:
        validation['warnings'].append(f"Features have {validation['missing_features']} missing values")
    
    if validation['missing_target'] > 0:
        validation['warnings'].append(f"Target has {validation['missing_target']} missing values")
        validation['valid'] = False
    
    if len(X) != len(y):
        validation['warnings'].append("Feature matrix and target have different lengths")
        validation['valid'] = False
    
    if len(X) == 0:
        validation['warnings'].append("No samples found")
        validation['valid'] = False
    
    return validation
