"""
Main entry point for watch scraping pipeline.
Coordinates discovery, scraping, and validation phases.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from .core.models import WatchTarget
from .discovery.discovery import WatchDiscovery
from .price import PriceScraper
from .validator import WatchDataValidator
from src.utils.io import read_mixed_json_file

logger = logging.getLogger(__name__)


class WatchScrapingPipeline:
    """Orchestrates the complete watch scraping pipeline."""
    
    def __init__(self, config: DictConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.pipeline_config = config.pipeline
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for the pipeline."""
        logging_config = self.config.get("logging", {})
        level = getattr(logging, logging_config.get("level", "INFO").upper())
        format_str = logging_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        logging.basicConfig(
            level=level,
            format=format_str,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def run(self) -> None:
        """Execute the pipeline based on configuration."""
        logger.info("🎯 WATCH SCRAPING PIPELINE STARTED")
        logger.info("="*60)
        self._print_configuration()
        
        targets: List[WatchTarget] = []
        
        try:
            # Phase 1: Discovery
            if self.pipeline_config.run_discovery:
                targets = self._run_discovery()
            
            # Phase 2: Scraping
            if self.pipeline_config.run_scraping:
                if not targets:
                    targets = self._load_targets()
                
                if targets:
                    self._run_scraping(targets)
                else:
                    logger.warning("No targets available for scraping")
            
            # Phase 3: Validation
            if self.pipeline_config.run_validation:
                self._run_validation()
            
            self._print_completion_summary()
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            raise
    
    def _print_configuration(self) -> None:
        """Print pipeline configuration."""
        logger.info("Pipeline Configuration:")
        logger.info(f"  Discovery: {'✓' if self.pipeline_config.run_discovery else '✗'}")
        logger.info(f"  Scraping: {'✓' if self.pipeline_config.run_scraping else '✗'}")
        logger.info(f"  Validation: {'✓' if self.pipeline_config.run_validation else '✗'}")
        logger.info("")
    
    def _run_discovery(self) -> List[WatchTarget]:
        """Run discovery phase."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: DISCOVERY")
        logger.info("="*60)
        
        discovery = WatchDiscovery(self.config)
        targets = discovery.run()
        
        logger.info(f"Discovery completed: {len(targets)} targets found")
        return targets
    
    def _run_scraping(self, targets: List[WatchTarget]) -> Dict:
        """Run scraping phase."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: SCRAPING")
        logger.info("="*60)
        logger.info(f"Total targets to scrape: {len(targets)}")
        
        scraper = PriceScraper(self.config)
        results = scraper.scrape_batch(targets)
        
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Scraping completed: {successful}/{len(results)} successful")
        
        return results
    
    def _run_validation(self) -> Dict:
        """Run validation phase."""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: VALIDATION")
        logger.info("="*60)
        
        validator = WatchDataValidator(self.config)
        results = validator.validate_all()
        
        # Report invalid watches
        invalid_watches = validator.get_invalid_watches(results)
        if invalid_watches:
            logger.info(f"\n⚠️ Found {len(invalid_watches)} watches needing attention:")
            for watch in invalid_watches[:5]:
                logger.info(f"  - {watch['brand']} {watch['model']}: {watch['issue']}")
            if len(invalid_watches) > 5:
                logger.info(f"  ... and {len(invalid_watches) - 5} more")
        
        return results
    
    def _load_targets(self) -> List[WatchTarget]:
        """Load existing targets from file."""
        discovery_config = self.config.discovery
        output_file = Path(discovery_config.output_file)
        
        if not output_file.exists():
            logger.warning(f"No targets file found at {output_file}")
            logger.info("Run discovery first or provide targets file")
            return []
        
        logger.info(f"Loading targets from {output_file}")
        targets_data = read_mixed_json_file(str(output_file))
        
        # Convert to WatchTarget objects
        targets: List[WatchTarget] = []
        for data in targets_data:
            # Extract watch_id from URL if not present
            watch_id = data.get("watch_id")
            model_id = data.get("model_id")
            
            # If watch_id is None or empty, try multiple extraction methods
            if not watch_id:
                # Method 1: Extract from model_name if it starts with ID
                model_name = data.get("model_name", "")
                if model_name and " - " in model_name:
                    parts = model_name.split(" - ", 1)
                    if parts[0].isdigit():
                        watch_id = parts[0]
                        model_id = watch_id
                        # Update model_name to remove ID prefix
                        data["model_name"] = parts[1]
                
                # Method 2: Extract from URL
                if not watch_id and data.get("url"):
                    import re
                    match = re.search(r"/watch_model/(\d+)-", data["url"])
                    if match:
                        watch_id = match.group(1)
                        model_id = watch_id
            
            # Skip if still no watch_id after extraction attempt
            if not watch_id:
                logger.warning(f"Skipping target without watch_id: {data.get('brand')} - {data.get('model_name')}")
                continue
            
            brand_raw = data.get("brand")
            model_name_raw = data.get("model_name")
            url_raw = data.get("url")

            if not isinstance(brand_raw, str) or not brand_raw.strip():
                logger.warning(
                    "Skipping target without brand for watch_id %s", watch_id
                )
                continue

            if not isinstance(model_name_raw, str) or not model_name_raw.strip():
                logger.warning(
                    "Skipping target without model_name for watch_id %s", watch_id
                )
                continue

            if not isinstance(url_raw, str) or not url_raw.strip():
                logger.warning(
                    "Skipping target without url for watch_id %s", watch_id
                )
                continue

            brand = brand_raw.strip()
            model_name = model_name_raw.strip()
            url = url_raw.strip()
            watch_id_str = str(watch_id)
            model_id_str = str(model_id) if model_id is not None else None

            target = WatchTarget(
                brand=brand,
                model_name=model_name,
                url=url,
                watch_id=watch_id_str,
                model_id=model_id_str,
                slug=data.get("slug")
            )
            targets.append(target)
        
        logger.info(f"Loaded {len(targets)} valid targets")
        return targets
    
    def _print_completion_summary(self) -> None:
        """Print final summary."""
        logger.info("\n" + "="*60)
        logger.info("🏁 PIPELINE COMPLETE")
        logger.info("="*60)
        
        # Output directory info
        output_dir = Path(self.config.scraping.output_dir)
        if output_dir.exists():
            csv_files = list(output_dir.glob("*.csv"))
            logger.info(f"📁 Output directory: {output_dir}")
            logger.info(f"📊 Total CSV files: {len(csv_files)}")
            
            # Calculate total data size
            total_size = sum(f.stat().st_size for f in csv_files)
            size_mb = total_size / (1024 * 1024)
            logger.info(f"💾 Total data size: {size_mb:.2f} MB")


@hydra.main(version_base=None, config_path="../../conf", config_name="scraping")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the watch scraping pipeline.
    
    Args:
        cfg: Hydra configuration object
    """
    pipeline = WatchScrapingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
