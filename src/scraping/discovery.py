"""
Watch URL discovery from WatchCharts brand pages.
Consolidated from watch_urls.py with configuration support.
"""

import json
import logging
import math
import os
from typing import Dict, List

from bs4 import BeautifulSoup
from selenium import webdriver

from .core.base_scraper import WatchScraper
from .core.browser import BrowserManager

logger = logging.getLogger(__name__)


class WatchDiscovery(WatchScraper):
    """Discover watch URLs from WatchCharts brand pages."""

    def __init__(self, config: Dict):
        """Initialize discovery with configuration."""
        super().__init__(config)

        # Override delay range for discovery (lighter load)
        discovery_config = config.get("discovery", {})
        self.delay_range = tuple(discovery_config.get("delay_range", [3, 8]))
        self.target_count_per_brand = discovery_config.get("target_count_per_brand", 10)
        self.discovery_headless = discovery_config.get("headless", True)
        self.output_file = discovery_config.get(
            "output_file", "data/targets/watch_targets.json"
        )

        # Brand URLs from configuration
        self.brand_urls = config.get("brands", {})

    def create_browser_session(self) -> "webdriver.Chrome":
        """Create browser session optimized for discovery."""
        return BrowserManager.create_discovery_driver(headless=self.discovery_headless)

    def get_total_watch_count(
        self, driver: "webdriver.Chrome", brand_url: str, brand: str
    ) -> int:
        """Extract total watch count from brand search page."""
        try:
            # Navigate to first page to get total count
            if not self.safe_navigate_with_retries(driver, brand_url):
                logger.error(f"Failed to navigate to {brand} page for count extraction")
                return 0

            # Parse page with BeautifulSoup
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")

            # Primary method: Look for the specific pagination display count element
            pagination_element = soup.find("p", {"id": "pagination-display-count"})
            if pagination_element:
                text = pagination_element.get_text().strip()
                logger.info(f"📊 Found pagination element text: '{text}'")

                # Extract number from text like "368 results"
                import re

                match = re.search(r"(\d+)\s+results", text)
                if match:
                    total = int(match.group(1))
                    logger.info(
                        f"📊 Found total count from pagination element: {total}"
                    )
                    return total

            # Fallback methods if pagination element not found
            text_content = soup.get_text().lower()
            import re

            # Pattern 1: "X results" anywhere in text
            results_match = re.search(r"(\d+)\s+results", text_content)
            if results_match:
                total = int(results_match.group(1))
                logger.info(f"📊 Found total count via 'X results' pattern: {total}")
                return total

            # Pattern 2: "Showing 1-24 of X"
            showing_match = re.search(r"showing\s+\d+-\d+\s+of\s+(\d+)", text_content)
            if showing_match:
                total = int(showing_match.group(1))
                logger.info(
                    f"📊 Found total count via 'showing X of Y' pattern: {total}"
                )
                return total

            # Pattern 3: Look for pagination indicators
            pagination_links = soup.find_all("a", href=True)
            max_page = 0
            for link in pagination_links:
                href = link.get("href", "")
                page_match = re.search(r"page=(\d+)", href)
                if page_match:
                    page_num = int(page_match.group(1))
                    max_page = max(max_page, page_num)

            if max_page > 0:
                # Estimate total: max_page * 24 watches per page (with ?page=x parameter)
                estimated_total = max_page * 24
                logger.info(
                    f"📊 Estimated total count from pagination: {estimated_total} (max page: {max_page})"
                )
                return estimated_total

            # Pattern 4: Count actual watch_model links and assume it's a single page
            watch_model_links = [
                link
                for link in soup.find_all("a", href=True)
                if "/watch_model/" in link.get("href", "")
            ]
            if watch_model_links:
                logger.info(
                    f"📊 Single page detected with {len(watch_model_links)} watch links"
                )
                return len(watch_model_links) // 3  # Account for duplicates

            logger.warning(f"Could not determine total watch count for {brand}")
            return 0

        except Exception as e:
            logger.error(f"Error extracting total watch count for {brand}: {str(e)}")
            return 0

    def discover_watches_from_brand_page(
        self, brand: str, brand_url: str, target_count: int = None
    ) -> List[Dict]:
        """Discover watches from a brand page with pagination support."""
        if target_count is None:
            target_count = self.target_count_per_brand

        logger.info(f"🔍 Discovering watches for {brand}")

        driver = None
        all_watches = []  # Store ALL discovered watches
        target_watches = []  # Store limited watches for main targets file

        # Create main log file once for this brand
        os.makedirs("logs/debug", exist_ok=True)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_log_file = (
            f"logs/debug/{brand.lower().replace(' ', '_')}_all_links_{timestamp}.log"
        )

        try:
            # Create driver and navigate with retries
            driver = self.create_browser_session()

            # Add initial human behavior simulation
            from .core.browser import simulate_human_behavior

            simulate_human_behavior(driver)

            # First, get the total count of watches available
            total_watch_count = self.get_total_watch_count(driver, brand_url, brand)
            if total_watch_count:
                logger.info(
                    f"📊 {brand} has {total_watch_count} total watches available"
                )

            # Calculate how many pages we need to scrape (24 watches per page when using ?page=x)
            watches_per_page = 24
            total_pages = 1
            if total_watch_count:
                total_pages = math.ceil(total_watch_count / watches_per_page)
                logger.info(
                    f"📄 Will scrape {total_pages} pages to get all {total_watch_count} watches (24 per page)"
                )

            # Scrape all pages
            processed_urls = set()
            for page_num in range(1, total_pages + 1):
                logger.info(f"🔍 Scraping page {page_num}/{total_pages} for {brand}")

                # Build page URL (always use ?page=x parameter for consistency)
                page_url = f"{brand_url}?page={page_num}"

                # Navigate to page with Cloudflare handling
                if not self.safe_navigate_with_retries(driver, page_url):
                    logger.error(f"Failed to navigate to {brand} page {page_num}")
                    continue

                # Wait for dynamic content to load and handle Cloudflare
                import time

                from selenium.webdriver.common.by import By
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.support.ui import WebDriverWait

                from .core.browser import check_cloudflare_challenge

                # Check for Cloudflare challenge with enhanced waiting
                if check_cloudflare_challenge(driver):
                    logger.warning(
                        f"Cloudflare challenge detected on page {page_num}, attempting to wait it out..."
                    )
                    from .core.browser import (
                        simulate_human_behavior,
                        wait_for_cloudflare_challenge,
                    )

                    # Use the enhanced Cloudflare waiting function
                    if not wait_for_cloudflare_challenge(driver, max_wait=90):
                        logger.error(
                            f"Cloudflare challenge not resolved on page {page_num}, skipping"
                        )
                        continue

                    # Simulate human behavior after challenge resolution
                    simulate_human_behavior(driver)

                # Wait for content to load
                try:
                    # Wait for pagination element which indicates page is ready
                    WebDriverWait(driver, 10).until(
                        EC.any_of(
                            EC.presence_of_element_located(
                                (By.ID, "pagination-display-count")
                            ),
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "a[href*='/watch_model/']")
                            ),
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, ".watch-grid, .watch-list")
                            ),
                        )
                    )
                except Exception:
                    logger.warning(
                        f"Timeout waiting for content on page {page_num}, proceeding anyway"
                    )

                # Additional wait for dynamic content
                time.sleep(2)

                # Scroll to trigger any lazy loading
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(1)

                # Parse page with BeautifulSoup for efficiency
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, "html.parser")

                # Find watch model links
                watch_links = soup.find_all("a", href=True)

                # Debug: Log total links found and how many contain watch_model
                total_links = len(watch_links)
                watch_model_links = [
                    link
                    for link in watch_links
                    if "/watch_model/" in link.get("href", "")
                ]
                logger.info(
                    f"📋 Page {page_num}: Found {total_links} total links, {len(watch_model_links)} with '/watch_model/'"
                )

                # Debug: Log unique watch_model URLs for inspection and save to main log file
                if len(watch_model_links) > 0:
                    # Extract and deduplicate only relative /watch_model/ paths
                    unique_watch_paths = set()
                    for link in watch_model_links:
                        href = link.get("href", "")
                        # Only keep relative paths that start with /watch_model/
                        if href.startswith("/watch_model/"):
                            unique_watch_paths.add(href)

                    unique_paths_list = sorted(list(unique_watch_paths))

                    logger.info(
                        f"🔗 Unique watch_model paths from page {page_num} ({len(unique_paths_list)} unique):"
                    )

                    # Save to main log file
                    with open(main_log_file, "a") as main_f:
                        header = f"Page {page_num} - {brand} - Found {len(unique_paths_list)} unique watch_model paths:"
                        separator = "=" * 80
                        main_f.write(f"\n{header}\n{separator}\n")

                        for i, path in enumerate(unique_paths_list):
                            logger.info(f"   {i + 1}. {path}")
                            main_f.write(f"{i + 1}. {path}\n")

                        main_f.write("\n")

                    logger.info(
                        f"💾 Unique paths appended to main log: {main_log_file}"
                    )

                page_watches = 0
                for link in watch_links:
                    href = link.get("href", "")
                    if "/watch_model/" not in href:
                        continue

                    # Build full URL and avoid duplicates
                    full_url = self.ensure_absolute_url(href)
                    if full_url in processed_urls:
                        logger.debug(f"🔄 Duplicate URL skipped: {href}")
                        continue
                    processed_urls.add(full_url)

                    # Extract watch ID and model info from URL
                    watch_id = self.extract_watch_id_from_url(href)
                    model_info = self.extract_model_info_from_url(href, brand)

                    # Debug: Log filtering details
                    if not watch_id or watch_id == "unknown" or not model_info:
                        logger.debug(
                            f"🚫 Filtered out {href}: watch_id='{watch_id}', model_info='{model_info}'"
                        )
                        continue

                    model_name = model_info

                    watch_data = {
                        "brand": brand,
                        "model_name": model_name,
                        "url": full_url,
                        "source": "generated",
                    }

                    # Add to ALL watches collection
                    all_watches.append(watch_data)
                    page_watches += 1

                    # Add to target collection only if under limit
                    if len(target_watches) < target_count:
                        target_watches.append(watch_data)
                        logger.debug(f"✅ Found: {brand} - {model_name}")

                logger.info(f"✅ Page {page_num}: Discovered {page_watches} watches")

                # Add delay between pages to avoid rate limiting
                if page_num < total_pages:
                    self.random_delay()

            logger.info(
                f"🎯 Discovered {len(all_watches)} total watches for {brand} (using {len(target_watches)} for targets)"
            )

            # Save brand-specific JSON file with ALL watches
            self.save_brand_watches(brand, all_watches)

            return target_watches

        except Exception as e:
            logger.error(f"Error discovering watches for {brand}: {str(e)}")
            return target_watches

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

                logger.info(
                    f"📊 {brand}: {len(brand_watches)}/{self.target_count_per_brand} watches discovered"
                )

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

        # Change extension to .jsonl
        if filename.endswith(".json"):
            filename = filename.replace(".json", ".jsonl")

        with open(filename, "w", encoding="utf-8") as f:
            for watch in watches:
                json.dump(watch, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"✅ Watch targets saved to {filename}")

    def save_brand_watches(self, brand: str, watches: List[Dict]) -> None:
        """Save discovered watches for a specific brand to a JSON file."""
        # Ensure targets directory exists
        targets_dir = "data/targets"
        os.makedirs(targets_dir, exist_ok=True)

        # Create brand-specific filename with .jsonl extension
        brand_filename = os.path.join(targets_dir, f"{brand.replace(' ', '_')}.jsonl")

        logger.info(f"💾 Saving {len(watches)} watches for {brand} to {brand_filename}")

        with open(brand_filename, "w", encoding="utf-8") as f:
            for watch in watches:
                json.dump(watch, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"✅ {brand} watches saved to {brand_filename}")

    def run_discovery(self) -> List[Dict]:
        """Main discovery execution method."""
        logger.info("🚀 WATCH DISCOVERY STARTED")
        logger.info(
            f"Target: {len(self.brand_urls)} brands, {self.target_count_per_brand} watches each"
        )

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
