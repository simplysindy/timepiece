"""
Simplified data preparation pipeline for watch price data.

This module provides a consolidated approach to data processing, replacing
the complex multi-asset pipeline with a focused watch-only solution.
"""

from .config import DataPrepConfig, DataConfig, ProcessingConfig, FeatureConfig, WatchConfig, OutputConfig
from .process import WatchDataProcessor

__version__ = "2.0.0"

__all__ = [
    "DataPrepConfig",
    "DataConfig", 
    "ProcessingConfig",
    "FeatureConfig",
    "WatchConfig",
    "OutputConfig",
    "WatchDataProcessor"
]