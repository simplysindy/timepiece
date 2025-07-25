"""
Watch URL discovery from WatchCharts brand pages.
Consolidated from watch_urls.py with configuration support.
"""

import json
import logging
from typing import Dict, List

from bs4 import BeautifulSoup

from .core.base_scraper import WatchScraper
from .core.browser import BrowserManager

logger = logging.getLogger(__name__)


class WatchDiscovery(WatchScraper):
    """Discover watch URLs from WatchCharts brand pages."""

    def __init__(self, config: Dict):
        """Initialize discovery with configuration."""
        super().__init__(config)
        
        # Override delay range for discovery (lighter load)
        discovery_config = config.get('discovery', {})
        self.delay_range = tuple(discovery_config.get('delay_range', [3, 8]))
        self.target_count_per_brand = discovery_config.get('target_count_per_brand', 10)
        self.discovery_headless = discovery_config.get('headless', True)
        self.output_file = discovery_config.get('output_file', 'watch_targets_100.json')
        
        # Brand URLs from configuration
        self.brand_urls = config.get('brands', {})

    def create_browser_session(self) -> 'webdriver.Chrome':
        """Create browser session optimized for discovery."""
        return BrowserManager.create_discovery_driver(headless=self.discovery_headless)

    def discover_watches_from_brand_page(
        self, brand: str, brand_url: str, target_count: int = None
    ) -> List[Dict]:
        """Discover watches from a brand page."""
        if target_count is None:
            target_count = self.target_count_per_brand
            
        logger.info(f"🔍 Discovering watches for {brand}")
        
        driver = None
        watches = []
        
        try:
            # Create driver and navigate with retries
            driver = self.create_browser_session()
            
            # Wait for watch listings to load
            wait_elements = [
                "a[href*='/watch_model/']",  # Direct watch model links
                "a[href*='/watch/']",  # Alternative watch links
                ".watch-card a",  # Watch card links  
                ".watch-item a",  # Watch item links
            ]
            
            if not self.safe_navigate_with_retries(
                driver, brand_url, wait_for_elements=wait_elements
            ):
                logger.error(f"Failed to navigate to {brand} page")
                return watches
            
            # Parse page with BeautifulSoup for efficiency
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            
            # Find watch model links
            watch_links = soup.find_all("a", href=True)
            processed_urls = set()
            
            for link in watch_links:
                if len(watches) >= target_count:
                    break
                
                href = link.get("href", "")
                if "/watch_model/" not in href:
                    continue
                
                # Build full URL and avoid duplicates
                full_url = self.ensure_absolute_url(href)
                if full_url in processed_urls:
                    continue
                processed_urls.add(full_url)
                
                # Extract and clean model name
                title = link.get_text(strip=True)
                if not title and link.parent:
                    title = link.parent.get_text(strip=True)[:100]
                
                model_name = self.clean_model_name(title, brand)
                if len(model_name) < 3:
                    continue
                
                # Extract watch ID and combine with model name
                watch_id = self.extract_watch_id_from_url(href)
                if watch_id and watch_id != "unknown":
                    model_name = f"{watch_id} - {model_name}"
                
                watch_data = {
                    "brand": brand,
                    "model_name": model_name,
                    "url": full_url,
                    "source": "generated",
                }
                
                watches.append(watch_data)
                logger.info(f"✅ Found: {brand} - {model_name}")
            
            logger.info(f"🎯 Discovered {len(watches)} watches for {brand}")
            return watches
        
        except Exception as e:
            logger.error(f"Error discovering watches for {brand}: {str(e)}")
            return watches
        
        finally:
            BrowserManager.quit_driver_safely(driver)
            self.random_delay()

    def discover_all_watches(self) -> List[Dict]:
        """Discover watches from all configured brand pages."""
        logger.info("🚀 Starting watch discovery for all brands")
        
        all_watches = []
        
        for brand, brand_url in self.brand_urls.items():
            try:
                brand_watches = self.discover_watches_from_brand_page(
                    brand, brand_url, self.target_count_per_brand
                )
                all_watches.extend(brand_watches)
                
                logger.info(f"📊 {brand}: {len(brand_watches)}/{self.target_count_per_brand} watches discovered")
                
            except Exception as e:
                logger.error(f"Failed to process {brand}: {e}")
                continue
        
        logger.info(f"🎯 Total watches discovered: {len(all_watches)}")
        
        # Summary by brand
        brand_counts = {}
        for watch in all_watches:
            brand = watch["brand"]
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        logger.info("📊 Discovery Summary:")
        for brand, count in brand_counts.items():
            logger.info(f"  {brand}: {count}/{self.target_count_per_brand} watches")
        
        return all_watches

    def save_watch_targets(self, watches: List[Dict], filename: str = None) -> None:
        """Save discovered watches to JSON file."""
        if filename is None:
            filename = self.output_file
            
        logger.info(f"💾 Saving {len(watches)} watches to {filename}")
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(watches, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Watch targets saved to {filename}")

    def run_discovery(self) -> List[Dict]:
        """Main discovery execution method."""
        logger.info("🚀 WATCH DISCOVERY STARTED")
        logger.info(f"Target: {len(self.brand_urls)} brands, {self.target_count_per_brand} watches each")
        
        # Discover all watches
        watches = self.discover_all_watches()
        
        if watches:
            # Save to JSON file
            self.save_watch_targets(watches)
            
            logger.info("\\n" + "=" * 60)
            logger.info("🎯 WATCH DISCOVERY COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total watches discovered: {len(watches)}")
            target_total = len(self.brand_urls) * self.target_count_per_brand
            logger.info(f"Target: {target_total} watches")
            logger.info(f"Success rate: {len(watches) / target_total * 100:.1f}%")
            
            # Brand breakdown
            brand_counts = {}
            for watch in watches:
                brand = watch["brand"]
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            
            logger.info("\\nBrand Breakdown:")
            for brand, count in brand_counts.items():
                logger.info(f"  {brand}: {count}/{self.target_count_per_brand}")
            
            logger.info(f"\\nResults saved to: {self.output_file}")
        else:
            logger.error("❌ No watches discovered!")
        
        return watches