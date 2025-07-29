"""
Configuration schemas for the data preparation pipeline using Hydra.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Data input/output configuration."""
    input_dir: str = "data/final"
    output_dir: str = "data/processed"
    max_files: Optional[int] = None


@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    frequency: str = "D"
    interpolation_method: str = "backfill"
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0
    min_data_points: int = 30


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 30])
    include_temporal: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_technical: bool = True
    target_shift: int = -1


@dataclass
class LuxuryTier:
    """Configuration for a luxury tier."""
    min_price: float
    max_price: float
    volatility_factor: float


@dataclass
class WatchConfig:
    """Watch-specific configuration."""
    price_column: str = "price(SGD)"
    date_column: str = "date"
    luxury_tiers: Dict[str, LuxuryTier] = field(default_factory=dict)
    brand_tiers: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Output configuration."""
    save_individual_files: bool = False
    save_combined_file: bool = True
    include_metadata: bool = True


@dataclass
class DataPrepConfig:
    """Main configuration class for data preparation pipeline."""
    data: DataConfig = field(default_factory=DataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """Initialize default luxury tiers and brand tiers if not provided."""
        if not self.watch.luxury_tiers:
            self.watch.luxury_tiers = {
                "entry_luxury": LuxuryTier(0, 5000, 1.5),
                "mid_luxury": LuxuryTier(5000, 20000, 1.2),
                "high_luxury": LuxuryTier(20000, 100000, 1.0),
                "ultra_luxury": LuxuryTier(100000, float('inf'), 0.8)
            }
        
        if not self.watch.brand_tiers:
            self.watch.brand_tiers = {
                "ultra_luxury": ["patek_philippe", "audemars_piguet", "vacheron_constantin"],
                "high_luxury": ["rolex", "omega", "cartier"],
                "mid_luxury": ["tudor", "longines", "tissot", "seiko", "hublot"]
            }