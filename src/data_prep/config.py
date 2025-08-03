"""
Configuration schemas for the data preparation pipeline using Hydra.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class DataConfig:
    """Data input/output configuration."""

    input_dir: str = "data/final"
    output_dir: str = "data/processed"
    max_files: Optional[int] = None
    file_pattern: str = "*.csv"


@dataclass
class OutputSubdirs:
    """Output subdirectory structure."""
    
    processed: str = "processed"
    summary: str = "summary"


@dataclass
class OutputConfig:
    """Output structure configuration."""

    save_individual: bool = True
    save_summary: bool = True
    subdirs: OutputSubdirs = field(default_factory=OutputSubdirs)
    file_format: str = "csv"
    include_index: bool = True


@dataclass
class ProcessingConfig:
    """Data processing configuration."""

    frequency: str = "D"
    min_data_points: int = 30
    interpolation_method: str = "linear"
    fill_limit: int = 5
    
    # Outlier detection
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0
    handle_outliers: str = "interpolate"
    
    # Data quality
    max_missing_percent: float = 20.0
    remove_duplicates: bool = True
    sort_index: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Lag features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14, 30])
    
    # Rolling window features
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 21, 30, 60, 90])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "median"])
    
    # Price change periods (for momentum features)
    price_change_periods: List[int] = field(default_factory=lambda: [1, 3, 7, 14, 21, 30])
    
    # Technical indicators
    include_technical: bool = True
    technical_indicators: List[str] = field(default_factory=lambda: ["sma", "ema", "rsi", "bollinger"])
    
    # Watch-specific features
    include_watch_features: bool = True
    watch_features: List[str] = field(default_factory=lambda: ["luxury_tier", "seasonality", "price_stability", "brand_features"])
    
    # Time-based features
    include_temporal: bool = True
    temporal_features: List[str] = field(default_factory=lambda: ["day_of_week", "month", "quarter", "is_weekend", "is_month_end"])
    
    # Target variable configuration
    target_shift: int = -1  # Days to shift for target variable (negative = future)


@dataclass
class WatchConfig:
    """Watch-specific configuration."""

    price_column: str = "price(SGD)"
    date_column: str = "date"
    
    # Luxury tier thresholds (in SGD)
    luxury_tiers: Dict[str, List[Union[int, float, None]]] = field(default_factory=lambda: {
        "entry": [0, 5000],
        "mid": [5000, 20000],
        "high": [20000, 100000],
        "ultra": [100000, None]
    })
    
    # Brand categorization
    premium_brands: List[str] = field(default_factory=lambda: [
        "rolex", "patek_philippe", "audemars_piguet", "omega", "cartier"
    ])


@dataclass
class BehaviorConfig:
    """Processing behavior configuration."""

    continue_on_error: bool = True
    verbose: bool = True
    validate_output: bool = True
    overwrite_existing: bool = False


@dataclass
class SummaryConfig:
    """Summary statistics configuration."""

    include_stats: List[str] = field(default_factory=lambda: [
        "total_records", "date_range", "missing_data_percent", "avg_price",
        "price_volatility", "price_trend", "data_quality_score", "features_count"
    ])
    
    include_metadata: List[str] = field(default_factory=lambda: [
        "brand", "model", "luxury_tier", "processing_timestamp"
    ])


@dataclass
class DataPrepConfig:
    """Main configuration class for data preparation pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)