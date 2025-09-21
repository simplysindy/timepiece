"""
Watch URL discovery orchestrator.
Coordinates URL extraction from brand pages.
"""

import logging
import time
import random
from typing import List, Optional
from pathlib import Path

from omegaconf import DictConfig

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

from ..core.models import WatchTarget
from ..core.browser_factory import BrowserFactory, CloudflareHandler
from src.utils.io import ensure_output_directory, write_jsonl_file
from .extractors import URLExtractorChain, WatchDataParser

logger = logging.getLogger(__name__)


class WatchDiscovery:
    """Orchestrates watch URL discovery from brand pages."""
    
    def __init__(self, config: DictConfig):
        """Initialize discovery with configuration."""
        self.config = config
        discovery_config = config.get("discovery", {})
        
        # Configuration
        target_count_raw = discovery_config.get("target_count_per_brand", 10)
        if isinstance(target_count_raw, int):
            self.target_count: int = target_count_raw
        else:
            try:
                self.target_count = int(target_count_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid target_count_per_brand value %r; defaulting to 10",
                    target_count_raw,
                )
                self.target_count = 10
        self.delay_range = tuple(discovery_config.get("delay_range", [3, 8]))
        self.headless = discovery_config.get("headless", False)
        self.output_file = discovery_config.get(
            "output_file", "data/targets/watch_targets.jsonl"
        )
        self.max_retries = config.get("scraping", {}).get("max_retries", 3)
        
        # Brand URLs from configuration
        self.brand_urls = config.get("brands", {})
        
        # Initialize components
        self.extractor_chain = URLExtractorChain()
        self.parser = WatchDataParser()
        
    def discover_brand_watches(
        self, 
        brand: str, 
        brand_url: str,
        max_watches: Optional[int] = None
    ) -> List[WatchTarget]:
        """
        Discover watches for a single brand.
        
        Args:
            brand: Brand name
            brand_url: Brand page URL
            max_watches: Maximum watches to discover
            
        Returns:
            List of WatchTarget objects
        """
        if max_watches is None:
            effective_max_watches: int = self.target_count
        else:
            effective_max_watches = max_watches

        logger.info(
            f"🔍 Discovering watches for {brand} (target: {effective_max_watches})"
        )
        
        # Extract brand slug for filtering
        brand_slug = self._extract_brand_slug(brand_url)
        model_fragment = self.parser.normalize_brand_slug(brand_slug or brand)
        
        for attempt in range(1, self.max_retries + 1):
            driver = None
            try:
                # Create browser instance
                driver = BrowserFactory.create_driver("discovery", headless=self.headless)
                
                # Navigate to brand page
                if not self._navigate_to_page(driver, brand_url):
                    logger.error(f"Failed to navigate to {brand} page")
                    continue
                
                # Extract URLs
                urls = self.extractor_chain.extract_urls(
                    driver, model_fragment, effective_max_watches
                )
                
                if not urls:
                    logger.warning(f"No URLs found for {brand} on attempt {attempt}")
                    if attempt < self.max_retries:
                        self._random_delay()
                        continue
                
                # Parse watch data from URLs
                watches = []
                for url in list(urls)[:effective_max_watches]:
                    watch_data = self.parser.parse_watch_url(url, brand)
                    
                    watch = WatchTarget(
                        brand=brand,
                        model_name=watch_data['model_name'],
                        url=url,
                        watch_id=watch_data['watch_id'] or watch_data['model_id'],
                        model_id=watch_data['model_id'],
                        slug=model_fragment
                    )
                    watches.append(watch)
                
                logger.info(f"✅ Found {len(watches)} watches for {brand}")
                
                # Save brand-specific file
                self._save_brand_watches(brand, watches)
                
                return watches
                
            except Exception as e:
                logger.error(f"Error discovering {brand} watches (attempt {attempt}): {e}")
                if attempt < self.max_retries:
                    self._random_delay()
                    
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass
                        
        logger.error(f"Failed to discover watches for {brand} after {self.max_retries} attempts")
        return []
    
    def discover_all_brands(self) -> List[WatchTarget]:
        """
        Discover watches from all configured brands.
        
        Returns:
            List of all discovered WatchTarget objects
        """
        logger.info("🚀 Starting watch discovery for all brands")
        logger.info(f"Brands to process: {', '.join(self.brand_urls.keys())}")
        
        all_watches = []
        brand_delay = self.config.get("scraping", {}).get("brand_delay", 60)
        
        for idx, (brand, brand_url) in enumerate(self.brand_urls.items(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{idx}/{len(self.brand_urls)}] Processing {brand}")
            logger.info(f"{'='*60}")
            
            watches = self.discover_brand_watches(brand, brand_url)
            all_watches.extend(watches)
            
            # Summary for this brand
            if watches:
                logger.info(f"Sample watches for {brand}:")
                for watch in watches[:3]:
                    logger.info(f"  - {watch.model_name or watch.url}")
            
            # Delay between brands
            if idx < len(self.brand_urls) and brand_delay > 0:
                logger.info(f"Waiting {brand_delay}s before next brand...")
                time.sleep(brand_delay)
        
        self._print_summary(all_watches)
        return all_watches
    
    def _navigate_to_page(self, driver, url: str) -> bool:
        """Navigate to URL with Cloudflare handling."""
        try:
            logger.info(f"Navigating to {url}")
            try:
                driver.get(url)
            except TimeoutException:
                logger.warning(
                    "Page load timeout for %s; continuing with available DOM", url
                )

            wait = WebDriverWait(driver, 15)

            try:
                wait.until(
                    lambda d: d.execute_script("return document.readyState")
                    in {"interactive", "complete"}
                )
            except TimeoutException:
                logger.debug("Ready state wait timed out; attempting to proceed")

            # Check for Cloudflare challenge after initial paint
            if CloudflareHandler.check_challenge(driver):
                logger.warning("Cloudflare challenge detected")
                if not CloudflareHandler.wait_for_challenge(driver):
                    return False

            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "a")))
            except TimeoutException:
                logger.warning("Anchor tags did not render in time; continuing")

            time.sleep(random.uniform(*self.delay_range))

            return True

        except TimeoutException:
            logger.warning(f"Page load timeout for {url}")
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    def _extract_brand_slug(self, brand_url: str) -> str:
        """Extract brand slug from URL."""
        return brand_url.rstrip("/").split("/")[-1]
    
    def _random_delay(self) -> None:
        """Add random delay between requests."""
        delay = random.uniform(*self.delay_range)
        logger.info(f"Waiting {delay:.1f}s...")
        time.sleep(delay)
    
    def _save_brand_watches(self, brand: str, watches: List[WatchTarget]) -> None:
        """Save watches for a specific brand."""
        if not watches:
            return
            
        targets_dir = ensure_output_directory("data", "targets")
        brand_filename = targets_dir / f"{brand.replace(' ', '_')}.jsonl"
        
        # Convert to dictionaries for saving
        watch_dicts = [w.to_dict() for w in watches]
        
        write_jsonl_file(watch_dicts, brand_filename)
        logger.info(f"💾 Saved {len(watches)} watches to {brand_filename}")
    
    def _print_summary(self, watches: List[WatchTarget]) -> None:
        """Print discovery summary."""
        logger.info("\n" + "="*60)
        logger.info("🎯 DISCOVERY SUMMARY")
        logger.info("="*60)
        logger.info(f"Total watches discovered: {len(watches)}")
        
        # Brand breakdown
        brand_counts = {}
        for watch in watches:
            brand_counts[watch.brand] = brand_counts.get(watch.brand, 0) + 1
        
        logger.info("\nBreakdown by brand:")
        for brand, count in brand_counts.items():
            logger.info(f"  {brand}: {count}/{self.target_count}")
        
        # Success rate
        expected_total = len(self.brand_urls) * self.target_count
        if expected_total > 0:
            success_rate = (len(watches) / expected_total) * 100
            logger.info(f"\nSuccess rate: {success_rate:.1f}%")
    
    def run(self) -> List[WatchTarget]:
        """
        Run the discovery process.
        
        Returns:
            List of discovered WatchTarget objects
        """
        watches = self.discover_all_brands()
        
        if watches:
            # Save consolidated file
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            watch_dicts = [w.to_dict() for w in watches]
            write_jsonl_file(watch_dicts, output_path)
            
            logger.info(f"\n📄 Saved all targets to: {self.output_file}")
        else:
            logger.error("⚠️ No watches discovered!")
        
        return watches
