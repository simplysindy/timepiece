"""
Consolidated data processing pipeline for watch price data.

This module combines data loading, processing, feature engineering, and watch-specific
logic into a single comprehensive module for simplified watch-only data preparation.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats, interpolate
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from .config import DataPrepConfig

logger = logging.getLogger(__name__)


class WatchDataProcessor:
    """Comprehensive watch data processing with all functionality consolidated."""
    
    def __init__(self, config: DataPrepConfig):
        self.config = config
        
    def process_all_watches(self) -> Dict[str, Any]:
        """
        Process all watch data files through the complete pipeline.
        
        Returns:
        -------
        Dict[str, Any]
            Complete processing results
        """
        logger.info("Starting complete watch data processing pipeline")
        
        # Step 1: Load all watch data
        raw_data = self.load_watch_data()
        if not raw_data:
            logger.error("No watch data loaded")
            return {"success": False, "error": "No data loaded"}
        
        logger.info(f"Loaded {len(raw_data)} watch files")
        
        # Step 2: Clean and validate data
        processed_data = self.clean_and_validate_data(raw_data)
        logger.info(f"Processed {len(processed_data)} valid watches")
        
        # Step 3: Engineer features
        featured_data = self.engineer_all_features(processed_data)
        logger.info(f"Featured data for {len(featured_data)} watches")
        
        # Step 4: Create combined dataset
        combined_data = self.create_combined_dataset(featured_data)
        logger.info(f"Combined dataset: {len(combined_data)} records")
        
        # Step 5: Save outputs
        output_files = self.save_outputs(combined_data, featured_data)
        
        return {
            "success": True,
            "raw_count": len(raw_data),
            "processed_count": len(processed_data),
            "featured_count": len(featured_data),
            "combined_records": len(combined_data),
            "output_files": output_files
        }
    
    def load_watch_data(self) -> Dict[str, pd.DataFrame]:
        """Load all watch CSV files from the input directory."""
        
        input_path = Path(self.config.data.input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_path}")
            return {}
        
        # Find all CSV files
        csv_files = list(input_path.glob("*.csv"))
        if self.config.data.max_files:
            csv_files = csv_files[:self.config.data.max_files]
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        raw_data = {}
        
        for file_path in csv_files:
            try:
                # Extract watch name from filename
                watch_name = file_path.stem
                
                # Load CSV
                df = pd.read_csv(file_path)
                
                # Validate required columns
                if self.config.watch.date_column not in df.columns:
                    logger.warning(f"Missing date column in {watch_name}")
                    continue
                if self.config.watch.price_column not in df.columns:
                    logger.warning(f"Missing price column in {watch_name}")
                    continue
                
                # Convert date to datetime and set as index
                df[self.config.watch.date_column] = pd.to_datetime(df[self.config.watch.date_column])
                df.set_index(self.config.watch.date_column, inplace=True)
                df.sort_index(inplace=True)
                
                # Basic validation
                if len(df) < self.config.processing.min_data_points:
                    logger.warning(f"Insufficient data points for {watch_name}: {len(df)}")
                    continue
                
                raw_data[watch_name] = df
                logger.debug(f"Loaded {watch_name}: {len(df)} records")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                continue
        
        return raw_data
    
    def clean_and_validate_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean, validate, and preprocess watch data."""
        
        processed_data = {}
        
        for watch_name, df in raw_data.items():
            try:
                df_clean = df.copy()
                price_col = self.config.watch.price_column
                
                # Remove invalid prices
                df_clean = df_clean[df_clean[price_col] > 0]
                
                # Remove outliers
                df_clean = self._remove_outliers(df_clean, price_col)
                
                # Handle missing values
                df_clean = self._handle_missing_values(df_clean, price_col)
                
                # Ensure sufficient data remains
                if len(df_clean) < self.config.processing.min_data_points:
                    logger.warning(f"Insufficient data after cleaning for {watch_name}")
                    continue
                
                # Watch-specific validation
                validation_results = self._validate_watch_data(df_clean, watch_name)
                if not validation_results['valid']:
                    logger.warning(f"Validation failed for {watch_name}: {validation_results['errors']}")
                    continue
                
                processed_data[watch_name] = df_clean
                logger.debug(f"Cleaned {watch_name}: {len(df_clean)} records")
                
            except Exception as e:
                logger.error(f"Failed to clean {watch_name}: {str(e)}")
                continue
        
        return processed_data
    
    def _remove_outliers(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Remove outliers using the configured method."""
        
        method = self.config.processing.outlier_method
        threshold = self.config.processing.outlier_threshold
        
        if method == "iqr":
            Q1 = df[price_col].quantile(0.25)
            Q3 = df[price_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return df[(df[price_col] >= lower_bound) & (df[price_col] <= upper_bound)]
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(df[price_col]))
            return df[z_scores < threshold]
        
        elif method == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[[price_col]])
            return df[outlier_labels == 1]
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Handle missing values using the configured method."""
        
        method = self.config.processing.interpolation_method
        
        if method == "backfill":
            df[price_col] = df[price_col].bfill()
        elif method == "forward":
            df[price_col] = df[price_col].ffill()
        elif method == "linear":
            df[price_col] = df[price_col].interpolate(method='linear')
        elif method == "spline":
            df[price_col] = df[price_col].interpolate(method='spline', order=2)
        
        # Drop any remaining NaN values
        df.dropna(subset=[price_col], inplace=True)
        
        return df
    
    def _validate_watch_data(self, df: pd.DataFrame, watch_name: str) -> Dict[str, Any]:
        """Perform watch-specific validation."""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'watch_metadata': {}
        }
        
        price_col = self.config.watch.price_column
        prices = df[price_col].dropna()
        
        if len(prices) == 0:
            validation_results['errors'].append("No valid prices")
            validation_results['valid'] = False
            return validation_results
        
        # Price range validation
        min_price = prices.min()
        max_price = prices.max()
        mean_price = prices.mean()
        
        if min_price < 100:  # SGD
            validation_results['warnings'].append(f"Low watch price: {min_price} SGD")
        if max_price > 1000000:  # 1M SGD
            validation_results['warnings'].append(f"High watch price: {max_price} SGD")
        
        # Classify luxury tier
        luxury_tier = self._classify_luxury_tier(mean_price)
        
        # Parse watch metadata
        metadata = self._parse_watch_metadata(watch_name)
        
        validation_results['watch_metadata'] = {
            'luxury_tier': luxury_tier,
            'price_stats': {'min': min_price, 'max': max_price, 'mean': mean_price},
            'metadata': metadata
        }
        
        return validation_results
    
    def _classify_luxury_tier(self, mean_price: float) -> str:
        """Classify watch into luxury tier based on price."""
        
        for tier_name, tier_config in self.config.watch.luxury_tiers.items():
            if tier_config.min_price <= mean_price < tier_config.max_price:
                return tier_name
        return 'unknown'
    
    def _parse_watch_metadata(self, watch_name: str) -> Dict[str, str]:
        """Parse watch metadata from filename."""
        
        # Try to parse Brand-Model-ID format
        parts = watch_name.split('-')
        if len(parts) >= 3:
            return {
                'brand': parts[0].replace('_', ' '),
                'model': '-'.join(parts[1:-1]).replace('_', ' '),
                'id': parts[-1],
                'full_name': f"{parts[0]} {'-'.join(parts[1:-1])}"
            }
        elif len(parts) == 2:
            return {
                'brand': parts[0].replace('_', ' '),
                'model': parts[1].replace('_', ' '),
                'id': 'unknown',
                'full_name': f"{parts[0]} {parts[1]}"
            }
        else:
            return {
                'brand': watch_name,
                'model': 'unknown',
                'id': 'unknown',
                'full_name': watch_name
            }
    
    def engineer_all_features(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Engineer features for all watches."""
        
        featured_data = {}
        
        for watch_name, df in processed_data.items():
            try:
                df_featured = self._engineer_watch_features(df, watch_name)
                featured_data[watch_name] = df_featured
                logger.debug(f"Engineered features for {watch_name}: {len(df_featured.columns)} features")
                
            except Exception as e:
                logger.error(f"Failed to engineer features for {watch_name}: {str(e)}")
                continue
        
        return featured_data
    
    def _engineer_watch_features(self, df: pd.DataFrame, watch_name: str) -> pd.DataFrame:
        """Engineer comprehensive features for a single watch."""
        
        df_features = df.copy()
        price_col = self.config.watch.price_column
        
        # Temporal features
        if self.config.features.include_temporal:
            df_features = self._add_temporal_features(df_features)
        
        # Lag features
        df_features = self._add_lag_features(df_features, price_col)
        
        # Rolling window features
        df_features = self._add_rolling_features(df_features, price_col)
        
        # Momentum features
        if self.config.features.include_momentum:
            df_features = self._add_momentum_features(df_features, price_col)
        
        # Volatility features
        if self.config.features.include_volatility:
            df_features = self._add_volatility_features(df_features, price_col)
        
        # Technical indicators
        if self.config.features.include_technical:
            df_features = self._add_technical_indicators(df_features, price_col)
        
        # Watch-specific features
        df_features = self._add_watch_specific_features(df_features, price_col, watch_name)
        
        # Target variable
        df_features = self._add_target_variable(df_features, price_col)
        
        # Drop NaN values
        df_features.dropna(inplace=True)
        
        return df_features
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        
        df_temp = df.copy()
        
        # Basic temporal
        df_temp['day_of_week'] = df_temp.index.dayofweek
        df_temp['day_of_month'] = df_temp.index.day
        df_temp['month'] = df_temp.index.month
        df_temp['quarter'] = df_temp.index.quarter
        df_temp['year'] = df_temp.index.year
        
        # Cyclical encoding
        df_temp['day_of_week_sin'] = np.sin(2 * np.pi * df_temp['day_of_week'] / 7)
        df_temp['day_of_week_cos'] = np.cos(2 * np.pi * df_temp['day_of_week'] / 7)
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        
        # Boolean features
        df_temp['is_weekend'] = df_temp['day_of_week'] >= 5
        df_temp['is_month_start'] = df_temp.index.is_month_start
        df_temp['is_month_end'] = df_temp.index.is_month_end
        df_temp['is_quarter_start'] = df_temp.index.is_quarter_start
        df_temp['is_quarter_end'] = df_temp.index.is_quarter_end
        
        return df_temp
    
    def _add_lag_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add lagged price features."""
        
        df_lag = df.copy()
        
        for lag in self.config.features.lag_periods:
            df_lag[f'price_lag_{lag}'] = df_lag[price_col].shift(lag)
        
        return df_lag
    
    def _add_rolling_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add rolling window features."""
        
        df_roll = df.copy()
        
        for window in self.config.features.rolling_windows:
            df_roll[f'rolling_mean_{window}'] = df_roll[price_col].rolling(window=window).mean()
            df_roll[f'rolling_std_{window}'] = df_roll[price_col].rolling(window=window).std()
            df_roll[f'rolling_min_{window}'] = df_roll[price_col].rolling(window=window).min()
            df_roll[f'rolling_max_{window}'] = df_roll[price_col].rolling(window=window).max()
            df_roll[f'rolling_median_{window}'] = df_roll[price_col].rolling(window=window).median()
            
            # Price position within rolling window
            df_roll[f'price_position_{window}'] = (
                (df_roll[price_col] - df_roll[f'rolling_min_{window}']) /
                (df_roll[f'rolling_max_{window}'] - df_roll[f'rolling_min_{window}'])
            ).replace([np.inf, -np.inf], np.nan)
        
        return df_roll
    
    def _add_momentum_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add momentum features."""
        
        df_mom = df.copy()
        
        # Price changes
        for period in [1, 3, 7, 14, 21, 30]:
            df_mom[f'price_change_{period}'] = df_mom[price_col].pct_change(period)
            df_mom[f'price_change_abs_{period}'] = df_mom[price_col].diff(period)
        
        # Momentum indicators
        for window in [5, 10, 20]:
            df_mom[f'momentum_{window}'] = df_mom[price_col] / df_mom[price_col].shift(window)
            df_mom[f'roc_{window}'] = df_mom[price_col].pct_change(window)
        
        return df_mom
    
    def _add_volatility_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add volatility features."""
        
        df_vol = df.copy()
        
        # Rolling volatility
        returns = df_vol[price_col].pct_change()
        for window in [5, 10, 20, 30]:
            df_vol[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
            df_vol[f'volatility_ratio_{window}'] = (
                returns.rolling(window=window//2).std() /
                returns.rolling(window=window).std()
            ).replace([np.inf, -np.inf], np.nan)
        
        return df_vol
    
    def _add_technical_indicators(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add technical indicators."""
        
        df_tech = df.copy()
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df_tech[f'sma_{window}'] = df_tech[price_col].rolling(window=window).mean()
            df_tech[f'price_over_sma_{window}'] = df_tech[price_col] / df_tech[f'sma_{window}']
        
        # Exponential Moving Averages
        for span in [5, 10, 20]:
            df_tech[f'ema_{span}'] = df_tech[price_col].ewm(span=span).mean()
            df_tech[f'price_over_ema_{span}'] = df_tech[price_col] / df_tech[f'ema_{span}']
        
        # Bollinger Bands
        for window in [10, 20]:
            sma = df_tech[price_col].rolling(window=window).mean()
            std = df_tech[price_col].rolling(window=window).std()
            df_tech[f'bb_upper_{window}'] = sma + (2 * std)
            df_tech[f'bb_lower_{window}'] = sma - (2 * std)
            df_tech[f'bb_position_{window}'] = (
                (df_tech[price_col] - df_tech[f'bb_lower_{window}']) /
                (df_tech[f'bb_upper_{window}'] - df_tech[f'bb_lower_{window}'])
            ).replace([np.inf, -np.inf], np.nan)
        
        # RSI (Relative Strength Index)
        delta = df_tech[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_tech['rsi'] = 100 - (100 / (1 + rs))
        
        return df_tech
    
    def _add_watch_specific_features(self, df: pd.DataFrame, price_col: str, watch_name: str) -> pd.DataFrame:
        """Add watch-specific luxury market features."""
        
        df_watch = df.copy()
        
        # Parse watch metadata
        metadata = self._parse_watch_metadata(watch_name)
        brand = metadata['brand'].lower().replace(' ', '_')
        model = metadata['model'].lower()
        
        # Luxury tier features
        mean_price = df_watch[price_col].mean()
        luxury_tier = self._classify_luxury_tier(mean_price)
        
        # Initialize tier indicators
        for tier in self.config.watch.luxury_tiers.keys():
            df_watch[f'tier_{tier}'] = 1 if tier == luxury_tier else 0
        
        # Brand tier features
        for tier_name, tier_brands in self.config.watch.brand_tiers.items():
            df_watch[f'brand_tier_{tier_name}'] = int(any(tb in brand for tb in tier_brands))
        
        # Watch seasonality features
        df_watch['is_holiday_season'] = df_watch.index.month.isin([11, 12, 1])
        df_watch['is_watch_fair_season'] = df_watch.index.month.isin([3, 4])
        
        # Luxury market volatility
        returns = df_watch[price_col].pct_change()
        for window in [3, 5, 10]:
            df_watch[f'luxury_vol_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Premium pricing indicators
        rolling_30 = df_watch[price_col].rolling(window=30)
        df_watch['price_premium_30d'] = (
            (df_watch[price_col] - rolling_30.min()) / rolling_30.min()
        ).replace([np.inf, -np.inf], np.nan)
        
        # Watch category features
        sports_keywords = ['speedmaster', 'submariner', 'daytona', 'gmt', 'diver', 'chrono']
        df_watch['is_sports_watch'] = int(any(kw in model for kw in sports_keywords))
        
        dress_keywords = ['calatrava', 'master', 'patrimony', 'royal_oak', 'nautilus']
        df_watch['is_dress_watch'] = int(any(kw in model for kw in dress_keywords))
        
        return df_watch
    
    def _add_target_variable(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Add target variable for prediction."""
        
        df_target = df.copy()
        
        # Future price (shifted by target_shift days)
        shift = self.config.features.target_shift
        df_target['target'] = df_target[price_col].shift(shift)
        
        # Target as percentage change
        df_target['target_pct_change'] = df_target[price_col].pct_change(periods=abs(shift))
        
        return df_target
    
    def create_combined_dataset(self, featured_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create combined dataset from all watches."""
        
        combined_data = []
        
        for watch_name, df in featured_data.items():
            df_copy = df.copy()
            
            # Add watch identifier
            df_copy['watch_id'] = watch_name
            
            # Parse metadata and add as features
            metadata = self._parse_watch_metadata(watch_name)
            df_copy['brand'] = metadata['brand']
            df_copy['model'] = metadata['model']
            
            combined_data.append(df_copy)
        
        if not combined_data:
            return pd.DataFrame()
        
        # Combine all data
        final_df = pd.concat(combined_data, ignore_index=True)
        
        # Add categorical encodings
        if len(final_df) > 0:
            # Brand encoding
            brand_encoder = LabelEncoder()
            final_df['brand_encoded'] = brand_encoder.fit_transform(final_df['brand'])
            
            # Model encoding
            model_encoder = LabelEncoder()
            final_df['model_encoded'] = model_encoder.fit_transform(final_df['model'])
            
            # Add frequency encodings
            brand_counts = final_df['brand'].value_counts()
            final_df['brand_frequency'] = final_df['brand'].map(brand_counts)
            
            model_counts = final_df['model'].value_counts()
            final_df['model_frequency'] = final_df['model'].map(model_counts)
        
        return final_df
    
    def save_outputs(self, combined_data: pd.DataFrame, featured_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Save processed data to output files."""
        
        output_path = Path(self.config.data.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        # Save combined dataset
        if self.config.output.save_combined_file and len(combined_data) > 0:
            combined_file = output_path / "watch_data_processed.csv"
            combined_data.to_csv(combined_file, index=False)
            output_files.append(str(combined_file))
            logger.info(f"Saved combined dataset: {len(combined_data)} records")
        
        # Save individual files if requested
        if self.config.output.save_individual_files:
            for watch_name, df in featured_data.items():
                individual_file = output_path / f"{watch_name}_processed.csv"
                df.to_csv(individual_file, index=True)
                output_files.append(str(individual_file))
        
        # Save metadata summary
        if self.config.output.include_metadata and len(combined_data) > 0:
            metadata_summary = self._create_metadata_summary(combined_data)
            metadata_file = output_path / "watch_metadata_summary.csv"
            metadata_summary.to_csv(metadata_file, index=False)
            output_files.append(str(metadata_file))
            logger.info("Saved metadata summary")
        
        return output_files
    
    def _create_metadata_summary(self, combined_data: pd.DataFrame) -> pd.DataFrame:
        """Create summary of watch metadata."""
        
        summary_data = []
        
        for watch_id in combined_data['watch_id'].unique():
            watch_data = combined_data[combined_data['watch_id'] == watch_id]
            
            # Calculate date range more safely
            date_range_days = 0
            if 'date' in watch_data.columns and len(watch_data) > 1:
                try:
                    date_col = pd.to_datetime(watch_data['date'])
                    date_range_days = (date_col.max() - date_col.min()).days
                except:
                    date_range_days = len(watch_data)  # fallback to record count
            
            summary = {
                'watch_id': watch_id,
                'brand': watch_data['brand'].iloc[0],
                'model': watch_data['model'].iloc[0],
                'total_records': len(watch_data),
                'date_range_days': date_range_days,
                'avg_price': watch_data[self.config.watch.price_column].mean() if self.config.watch.price_column in watch_data.columns else 0,
                'price_volatility': watch_data[self.config.watch.price_column].std() if self.config.watch.price_column in watch_data.columns else 0,
                'luxury_tier': self._classify_luxury_tier(watch_data[self.config.watch.price_column].mean()) if self.config.watch.price_column in watch_data.columns else 'unknown'
            }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)