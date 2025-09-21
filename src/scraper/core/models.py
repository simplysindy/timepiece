"""
Data models for watch scraping system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class WatchTarget:
    """Represents a watch to be scraped."""
    brand: str
    model_name: str
    url: str
    watch_id: str
    model_id: Optional[str] = None
    slug: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'brand': self.brand,
            'model_name': self.model_name,
            'url': self.url,
            'watch_id': self.watch_id,
            'model_id': self.model_id,
            'slug': self.slug
        }


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    watch: WatchTarget
    success: bool
    data_points: int = 0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    filename: str
    is_valid: bool
    row_count: int
    error_message: Optional[str] = None
    has_date_issues: bool = False