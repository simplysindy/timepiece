"""
Consolidated watch scraping functionality combining base scraper, price scraper, and batch scraper.
This module provides the core WatchScraper class with all scraping capabilities.
"""

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .browser import (
    BrowserManager,
    check_cloudflare_challenge,
    check_page_loaded_successfully,
)

logger = logging.getLogger(__name__)


@dataclass
class WatchTarget:
    """Data class for watch scraping targets."""

    brand: str
    model_name: str
    url: str
    watch_id: str


class WatchScraper:
    """
    Consolidated watch scraper that handles discovery, price extraction, and batch processing.
    Combines functionality from BaseScraper, CloudflareBypassScraper, and MassWatchScraper.
    """

    def __init__(self, config: Dict):
        """Initialize scraper with configuration."""
        self.config = config
        self.scraping_config = config.get("scraping", {})

        # Extract configuration values
        self.delay_range = tuple(self.scraping_config.get("delay_range", [10, 20]))
        self.max_retries = self.scraping_config.get("max_retries", 3)
        self.brand_delay = self.scraping_config.get("brand_delay", 60)
        self.headless = self.scraping_config.get("headless", True)
        self.output_dir = self.scraping_config.get("output_dir", "data/watches")

        self.base_url = "https://watchcharts.com"

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def random_delay(self) -> None:
        """Add random delay between requests."""
        delay = random.uniform(*self.delay_range)
        logger.info(f"Waiting {delay:.1f} seconds...")
        time.sleep(delay)

    def create_browser_session(self) -> webdriver.Chrome:
        """Create a fresh browser session optimized for scraping."""
        return BrowserManager.create_scraping_driver(headless=self.headless)

    def safe_navigate_with_retries(
        self,
        driver: webdriver.Chrome,
        url: str,
        wait_for_elements: List[str] = None,
    ) -> bool:
        """
        Navigate to URL with retries and comprehensive error handling.

        Args:
            driver: Selenium WebDriver instance
            url: URL to navigate to
            wait_for_elements: List of CSS selectors to wait for after page load

        Returns:
            bool: True if navigation successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Navigating to {url} (attempt {attempt + 1}/{self.max_retries})"
                )

                # Navigate to URL
                driver.get(url)

                # Wait for basic page load
                WebDriverWait(driver, 30).until(
                    lambda d: d.execute_script("return document.readyState")
                    == "complete"
                )
                time.sleep(3)

                # Check if page loaded successfully
                if not check_page_loaded_successfully(driver):
                    if check_cloudflare_challenge(driver):
                        logger.warning("Cloudflare challenge detected, attempting extended wait...")
                        from .browser import wait_for_cloudflare_challenge, simulate_human_behavior
                        
                        # Use enhanced Cloudflare waiting
                        if not wait_for_cloudflare_challenge(driver, max_wait=90):
                            raise Exception("Cloudflare challenge not resolved after extended wait")
                        
                        # Simulate human behavior after resolution
                        simulate_human_behavior(driver)
                    else:
                        raise Exception("Website failed to load properly")

                # Wait for specific elements if provided
                if wait_for_elements:
                    for selector in wait_for_elements:
                        try:
                            WebDriverWait(driver, 20).until(
                                EC.presence_of_element_located(
                                    (By.CSS_SELECTOR, selector)
                                )
                            )
                            logger.info(f"Found required element: {selector}")
                            break
                        except Exception:
                            logger.debug(f"Element not found: {selector}")
                            continue

                logger.info("Navigation successful")
                return True

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to navigate after {self.max_retries} attempts: {e}"
                    )
                    return False

                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                self.random_delay()

        return False

    def extract_watch_id_from_url(self, url: str) -> str:
        """Extract watch ID from WatchCharts URL."""
        # Expected format: https://watchcharts.com/watch_model/{ID}-{slug}/overview
        match = re.search(r"/watch_model/(\d+)-([^/]+)", url)
        if match:
            return match.group(1)

        # Fallback - look for numeric ID in path
        path_parts = urlparse(url).path.split("/")
        for part in path_parts:
            if part and part.split("-")[0].isdigit():
                return part.split("-")[0]

        return "unknown"

    def extract_model_info_from_url(self, url: str, brand: str) -> str:
        """Extract model number and name from WatchCharts URL."""
        # Expected format: https://watchcharts.com/watch_model/{model_number}-{brand}-{model_name}/overview
        try:
            # Extract the path segment containing model info
            path = urlparse(url).path
            match = re.search(r"/watch_model/(\d+)-(.+?)/?(?:/overview)?$", path)

            if not match:
                return ""

            model_number = match.group(1)
            url_slug = match.group(2)

            # Remove brand from slug (normalize to lowercase for comparison)
            brand_normalized = brand.lower().replace(" ", "-").replace(".", "")
            url_parts = url_slug.split("-")

            # Find where brand ends in the URL parts
            brand_parts = brand_normalized.split("-")
            start_idx = 0

            # Look for brand parts at the beginning of url_parts
            for i, brand_part in enumerate(brand_parts):
                if i < len(url_parts) and url_parts[i].lower() == brand_part:
                    start_idx = i + 1
                else:
                    # If brand doesn't match exactly, try to find it
                    for j, url_part in enumerate(url_parts):
                        if brand_part in url_part.lower():
                            start_idx = max(start_idx, j + 1)
                            break
                    break

            # Extract model name parts after brand
            model_parts = url_parts[start_idx:]
            if not model_parts:
                return f"{model_number}"

            # Clean up and format model name
            model_name = " ".join(model_parts).replace("-", " ")

            # Capitalize important words
            model_name = " ".join(
                word.capitalize()
                if word.lower()
                not in [
                    "and",
                    "of",
                    "the",
                    "a",
                    "an",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "with",
                ]
                else word.lower()
                for word in model_name.split()
            )

            # Remove any trailing identifiers (like model codes) and /overview
            model_name = re.sub(r"\s+[A-Z0-9]{4,}\s*$", "", model_name).strip()
            model_name = model_name.replace("/overview", "").strip()

            return f"{model_number} - {model_name}" if model_name else model_number

        except Exception as e:
            logger.debug(f"Error extracting model info from URL {url}: {e}")
            return ""

    def clean_model_name(self, model_name: str, brand: str) -> str:
        """Clean and standardize model names by removing unwanted patterns."""
        if not model_name:
            return ""

        # Remove brand name from model name if present
        cleaned = model_name.replace(brand, "").strip()
        if cleaned.startswith("-"):
            cleaned = cleaned[1:].strip()

        # Comprehensive cleanup patterns
        cleanup_patterns = [
            r"In Production",
            r"Retail Price[^A-Za-z]*",
            r"Market Price[^A-Za-z]*",
            r"S\\$[\\d,]+",
            r"~S\\$[\\d,]+",
            r"\\d+mm",
            r"\\d+M ",
            r"Steel\\d+mm\\d+M",
            r"Steel\\d+mm",
            r"White gold\\d+mm\\d+M",
            r"Rose gold\\d+mm\\d+M",
            r"Yellow gold\\d+mm\\d+M",
            r"Gold/steel\\d+mm\\d+M",
            r"Titanium\\d+mm\\d+M",
            r"Ceramic\\d+mm\\d+M",
            r"[A-Z0-9]+/[A-Z0-9.-]+",  # Reference numbers like 5167A, 126300
            r"[A-Z0-9]{4,}",  # Long alphanumeric codes
        ]

        for pattern in cleanup_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Clean up whitespace and normalize
        cleaned = " ".join(cleaned.split())

        return cleaned if len(cleaned) >= 3 else model_name

    def make_filename_safe(self, text: str) -> str:
        """Convert text to filesystem-safe filename."""
        return (
            text.replace(" ", "_")
            .replace("/", "-")
            .replace("\\\\", "-")
            .replace(":", "-")
        )

    def ensure_absolute_url(self, url: str, add_overview: bool = True) -> str:
        """Ensure URL is absolute and optionally add /overview suffix."""
        # Make absolute if relative
        if url.startswith("/"):
            url = urljoin(self.base_url, url)

        # Add /overview if requested and not present
        if add_overview and not url.endswith("/overview"):
            url = url.rstrip("/") + "/overview"

        return url

    def extract_price_data(self, driver: webdriver.Chrome) -> Optional[pd.DataFrame]:
        """Extract price data using Chart.js extraction method."""
        chart_script = """
        function extractFromCharts() {
            if (!window.Chart || !window.Chart.instances) {
                return null;
            }

            const chartInstances = Object.values(window.Chart.instances);

            for (let chart of chartInstances) {
                if (!chart.data || !chart.data.datasets) continue;

                for (let dataset of chart.data.datasets) {
                    if (!dataset.data || dataset.data.length === 0) continue;

                    const samplePoint = dataset.data[0];
                    if (samplePoint &&
                        typeof samplePoint === 'object' &&
                        samplePoint.y !== undefined &&
                        samplePoint.x instanceof Date) {

                        return dataset.data.map(point => ({
                            date: point.x.toISOString().split('T')[0],
                            price: point.y
                        }));
                    }
                }
            }
            return null;
        }

        return JSON.stringify(extractFromCharts());
        """

        try:
            result = driver.execute_script(chart_script)
            if result and result != "null":
                data = json.loads(result)
                if data:
                    df = pd.DataFrame(data)
                    df.rename(columns={"price": "price(SGD)"}, inplace=True)
                    return df
        except Exception as e:
            logger.error(f"Chart.js extraction failed: {e}")

        return None

    def load_existing_data(self, output_file: str) -> Optional[pd.DataFrame]:
        """Load existing CSV data and return DataFrame with latest date info."""
        try:
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                if len(df) > 0 and "date" in df.columns:
                    # Convert date column to datetime for proper comparison
                    df["date"] = pd.to_datetime(df["date"])
                    return df
        except Exception as e:
            logger.warning(f"Error loading existing data from {output_file}: {e}")
        return None

    def merge_with_existing_data(
        self, new_df: pd.DataFrame, existing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge new data with existing data, keeping only newer data points."""
        if existing_df is None or len(existing_df) == 0:
            return new_df

        # Convert date columns to datetime
        new_df["date"] = pd.to_datetime(new_df["date"])
        existing_df["date"] = pd.to_datetime(existing_df["date"])

        # Get the latest date from existing data
        latest_existing_date = existing_df["date"].max()

        # Filter new data to only include dates after the latest existing date
        newer_data = new_df[new_df["date"] > latest_existing_date]

        if len(newer_data) > 0:
            # Combine existing data with newer data
            combined_df = pd.concat([existing_df, newer_data], ignore_index=True)
            # Sort by date and remove duplicates
            combined_df = combined_df.sort_values("date").drop_duplicates(
                subset=["date"], keep="last"
            )
            logger.info(
                f"Added {len(newer_data)} new data points after {latest_existing_date.date()}"
            )
            return combined_df
        else:
            logger.info(f"No new data points found after {latest_existing_date.date()}")
            return existing_df

    def scrape_single_watch(self, watch: WatchTarget) -> bool:
        """Scrape a single watch with full error handling and incremental updates."""
        # Create filename in format {brand}-{model_number}-{model_name}.csv
        brand_safe = self.make_filename_safe(watch.brand)

        # Extract model number and name from the model_name field
        if " - " in watch.model_name and watch.model_name.split(" - ")[0].isdigit():
            model_number = watch.model_name.split(" - ")[0]
            model_name = watch.model_name.split(" - ", 1)[1]
        else:
            # Fallback for any remaining old format data
            model_number = watch.watch_id
            model_name = watch.model_name

        model_name_safe = self.make_filename_safe(model_name)
        filename = f"{brand_safe}-{model_number}-{model_name_safe}.csv"
        output_file = os.path.join(self.output_dir, filename)

        # Load existing data to check for latest date
        existing_data = self.load_existing_data(output_file)
        if existing_data is not None:
            latest_date = existing_data["date"].max()
            logger.info(
                f"🔍 {watch.brand} {watch.model_name} - Checking for data after {latest_date.date()}"
            )
        else:
            logger.info(f"🔄 {watch.brand} {watch.model_name} - Starting fresh scrape")

        driver = None
        try:
            # Create driver and navigate with retries
            driver = self.create_browser_session()

            # Define chart elements to wait for
            chart_elements = ["canvas", "[data-chart]", ".chart"]

            if not self.safe_navigate_with_retries(
                driver, watch.url, wait_for_elements=chart_elements
            ):
                logger.error(f"Failed to navigate to {watch.brand} {watch.model_name}")
                return False

            # Extract data
            new_df = self.extract_price_data(driver)

            if new_df is not None and len(new_df) > 0:
                # Merge with existing data (if any)
                final_df = self.merge_with_existing_data(new_df, existing_data)

                # Convert dates back to string format for CSV storage
                if pd.api.types.is_datetime64_any_dtype(final_df["date"]):
                    final_df["date"] = final_df["date"].dt.strftime("%Y-%m-%d")
                elif final_df["date"].dtype == "object":
                    final_df["date"] = pd.to_datetime(final_df["date"]).dt.strftime(
                        "%Y-%m-%d"
                    )

                # Save merged data
                final_df.to_csv(output_file, index=False)

                if existing_data is not None:
                    new_points = len(final_df) - len(existing_data)
                    if new_points > 0:
                        logger.info(
                            f"✅ {watch.brand} {watch.model_name} - Updated with {new_points} new data points"
                        )
                    else:
                        logger.info(
                            f"✅ {watch.brand} {watch.model_name} - No new data points found"
                        )
                else:
                    logger.info(
                        f"✅ {watch.brand} {watch.model_name} - Saved {len(final_df)} data points (fresh)"
                    )

                return True
            else:
                logger.warning(
                    f"❌ {watch.brand} {watch.model_name} - No data extracted"
                )
                return False

        except Exception as e:
            logger.error(f"{watch.brand} {watch.model_name} - Error: {str(e)}")

            # Save error screenshot
            try:
                if driver:
                    error_dir = os.path.join(self.output_dir, "error")
                    os.makedirs(error_dir, exist_ok=True)

                    screenshot_filename = f"{watch.watch_id}_error.png"
                    screenshot_path = os.path.join(error_dir, screenshot_filename)
                    driver.save_screenshot(screenshot_path)
                    logger.info(f"Error screenshot saved: {screenshot_path}")
            except Exception as screenshot_error:
                logger.warning(f"Failed to save error screenshot: {screenshot_error}")

            return False

        finally:
            BrowserManager.quit_driver_safely(driver)
            self.random_delay()

    def scrape_watches_batch(self, watches: List[WatchTarget]) -> Dict[str, bool]:
        """Scrape multiple watches with brand-based batching and delays."""
        results = {}

        # Group watches by brand
        brand_groups = {}
        for watch in watches:
            if watch.brand not in brand_groups:
                brand_groups[watch.brand] = []
            brand_groups[watch.brand].append(watch)

        # Process each brand group
        for i, (brand, brand_watches) in enumerate(brand_groups.items()):
            logger.info(f"Processing {brand} ({len(brand_watches)} watches)")

            # Process watches in this brand
            for watch in brand_watches:
                try:
                    success = self.scrape_single_watch(watch)
                    results[watch.watch_id] = success

                    if success:
                        logger.info(
                            f"✅ Successfully processed {watch.brand} {watch.model_name}"
                        )
                    else:
                        logger.warning(
                            f"❌ Failed to process {watch.brand} {watch.model_name}"
                        )

                except Exception as e:
                    logger.error(
                        f"❌ Error processing {watch.brand} {watch.model_name}: {e}"
                    )
                    results[watch.watch_id] = False

                # Small delay between watches within same brand
                time.sleep(random.uniform(2, 5))

            # Longer delay between brands (except for last brand)
            if i < len(brand_groups) - 1:
                logger.info(f"Waiting {self.brand_delay} seconds before next brand...")
                time.sleep(self.brand_delay)

        return results
