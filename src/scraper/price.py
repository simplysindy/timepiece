"""
Price data scraping module.
Handles extraction of price history from watch pages.
"""

import json
import logging
import os
import time
import random
from typing import Optional, Dict, List, Any
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from omegaconf import DictConfig

from .core.models import WatchTarget, ScrapingResult
from .core.browser_factory import BrowserFactory, CloudflareHandler
from src.utils.io import (
    load_existing_csv_data,
    make_filename_safe,
    safe_write_csv_with_backup,
)

logger = logging.getLogger(__name__)


class PriceScraper:
    """Scrapes price data from individual watch pages."""
    
    def __init__(self, config: DictConfig):
        """Initialize price scraper with configuration."""
        self.config = config
        scraping_config = config.get("scraping", {})
        
        # Configuration
        self.delay_range = tuple(scraping_config.get("delay_range", [10, 20]))
        self.max_retries = scraping_config.get("max_retries", 3)
        self.headless = scraping_config.get("headless", True)
        self.output_dir = Path(scraping_config.get("output_dir", "data/watches"))
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def scrape_watch(self, watch: WatchTarget) -> ScrapingResult:
        """
        Scrape price data for a single watch.
        
        Args:
            watch: WatchTarget to scrape
            
        Returns:
            ScrapingResult with outcome
        """
        # Build output filename
        filename = self._build_filename(watch)
        output_file = self.output_dir / filename
        
        # Check existing data
        existing_data = self._load_existing_data(output_file)
        if existing_data is not None:
            latest_date = existing_data["date"].max()
            logger.info(f"📊 {watch.brand} {watch.model_name} - Checking for updates after {latest_date.date()}")
        else:
            logger.info(f"🆕 {watch.brand} {watch.model_name} - Starting fresh scrape")
        
        # Attempt scraping with retries
        for attempt in range(1, self.max_retries + 1):
            driver = None
            try:
                driver = BrowserFactory.create_driver("scraping", headless=self.headless)
                
                # Navigate to watch page
                if not self._navigate_with_cloudflare(driver, watch.url):
                    logger.error(f"Failed to navigate to {watch.url}")
                    continue
                
                # Extract price data
                price_data = self._extract_price_data(driver)
                
                if price_data is not None and not price_data.empty:
                    # Merge with existing data
                    final_data = self._merge_data(price_data, existing_data)
                    
                    # Save data
                    if self._save_data(final_data, output_file):
                        new_points = len(final_data) - (len(existing_data) if existing_data is not None else 0)
                        logger.info(f"✅ {watch.brand} {watch.model_name} - Saved {new_points} new points")
                        
                        return ScrapingResult(
                            watch=watch,
                            success=True,
                            data_points=len(final_data)
                        )
                    else:
                        logger.error(f"Failed to save data for {watch.brand} {watch.model_name}")
                        
                else:
                    logger.warning(f"No data extracted for {watch.brand} {watch.model_name}")
                    
                if attempt < self.max_retries:
                    self._random_delay()
                    
            except Exception as e:
                logger.error(f"Error scraping {watch.brand} {watch.model_name} (attempt {attempt}): {e}")
                self._save_error_screenshot(driver, watch)
                
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass
        
        return ScrapingResult(
            watch=watch,
            success=False,
            error_message="Failed after all retry attempts"
        )
    
    def scrape_batch(self, watches: List[WatchTarget]) -> Dict[str, ScrapingResult]:
        """
        Scrape multiple watches with intelligent batching.
        
        Args:
            watches: List of WatchTarget objects to scrape
            
        Returns:
            Dictionary mapping watch_id to ScrapingResult
        """
        results = {}
        brand_delay = self.config.get("scraping", {}).get("brand_delay", 60)
        
        # Group by brand for efficient processing
        brand_groups = {}
        for watch in watches:
            if watch.brand not in brand_groups:
                brand_groups[watch.brand] = []
            brand_groups[watch.brand].append(watch)
        
        # Process each brand group
        for idx, (brand, brand_watches) in enumerate(brand_groups.items(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {brand} ({len(brand_watches)} watches)")
            logger.info(f"{'='*60}")
            
            for watch in brand_watches:
                result = self.scrape_watch(watch)
                results[watch.watch_id] = result
                
                # Small delay between watches
                time.sleep(random.uniform(2, 5))
            
            # Longer delay between brands
            if idx < len(brand_groups) and brand_delay > 0:
                logger.info(f"Waiting {brand_delay}s before next brand...")
                time.sleep(brand_delay)
        
        self._print_summary(results)
        return results
    
    def _navigate_with_cloudflare(self, driver: webdriver.Chrome, url: str) -> bool:
        """Navigate to URL with Cloudflare handling."""
        try:
            driver.get(url)
            
            # Wait for page load
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            
            # Check for Cloudflare
            if CloudflareHandler.check_challenge(driver):
                if not CloudflareHandler.wait_for_challenge(driver, max_wait=90):
                    return False
            
            # Wait for chart elements
            try:
                WebDriverWait(driver, 10).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.TAG_NAME, "canvas")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-chart]")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".chart"))
                    )
                )
            except Exception:
                logger.warning("Chart elements not found, proceeding anyway")
            
            time.sleep(2)  # Allow charts to render
            return True
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False

    def _extract_price_data(self, driver: webdriver.Chrome) -> Optional[pd.DataFrame]:
        """Extract price history from the page."""
        
        # Try the corrected console snippet method that constructs the URL
        logger.info("Extracting price data from JSON endpoint...")
        df = self._extract_via_console_snippet(driver, debug=False)
        if df is not None and not df.empty:
            return df
        
        # If primary method failed, log the issue
        logger.warning("Failed to extract from JSON endpoint, chart data may be outdated")
        
        # Don't fall back to Chart.js data as it contains stale information
        # Log what went wrong for debugging
        debug_script = r"""
        return {
            current_url: window.location.href,
            pathname: window.location.pathname,
            watch_id: location.pathname.match(/watch_model\\/(\\d+)-/)?.[1] || 'not found',
            has_jquery: typeof $ !== 'undefined'
        };
        """
        
        try:
            debug_info = driver.execute_script(debug_script)
            logger.error(f"Extraction failed. Debug info: {debug_info}")
        except:
            pass
        
        logger.error("❌ Could not extract current price data")
        return None
    
    def _extract_via_console_snippet(self, driver: webdriver.Chrome, debug: bool = True) -> Optional[pd.DataFrame]:
        """Extract price data by constructing the correct JSON endpoint URL."""
        snippet = r"""
        var callback = arguments[arguments.length - 1];

        (function() {
            try {
                // Extract watch ID from the current page URL
                const pathMatch = location.pathname.match(/watch_model\/(\d+)-/);
                if (!pathMatch || !pathMatch[1]) {
                    callback({ error: 'Could not extract watch ID from URL: ' + location.pathname });
                    return;
                }
                
                const watchId = pathMatch[1];
                
                // Construct the correct JSON endpoint URL
                // This matches the working console code exactly
                const correctUrl = `/charts/watch/${watchId}.json?type=trend&key=0100&variation_id=0&mobile=0&_=${Date.now()}`;
                
                console.log('Fetching from constructed URL:', correctUrl);
                console.log('Watch ID:', watchId);
                
                // Use jQuery's getJSON if available (preferred method)
                if (window.$ && window.$.getJSON) {
                    window.$.getJSON(correctUrl)
                        .then(function(resp) {
                            try {
                                // Extract data following the exact pattern from console
                                if (!resp || !resp.data || !resp.data.all) {
                                    callback({ error: 'No data.all in response', url: correctUrl });
                                    return;
                                }
                                
                                const all = resp.data.all;
                                const rows = Object.entries(all).map(function([ts, v]) {
                                    return {
                                        date: new Date(Number(ts) * 1000).toISOString().slice(0, 10),
                                        value: v
                                    };
                                });
                                
                                callback({ 
                                    rows: rows, 
                                    source: correctUrl,
                                    watchId: watchId,
                                    success: true 
                                });
                            } catch (err) {
                                callback({ error: 'Error processing response: ' + err.message, url: correctUrl });
                            }
                        })
                        .catch(function(err) {
                            callback({ error: 'jQuery getJSON failed: ' + (err.message || String(err)), url: correctUrl });
                        });
                } 
                // Fallback to fetch if jQuery isn't available
                else if (window.fetch) {
                    fetch(correctUrl)
                        .then(function(resp) { 
                            if (!resp.ok) {
                                throw new Error('HTTP ' + resp.status);
                            }
                            return resp.json(); 
                        })
                        .then(function(data) {
                            try {
                                if (!data || !data.data || !data.data.all) {
                                    callback({ error: 'No data.all in response', url: correctUrl });
                                    return;
                                }
                                
                                const all = data.data.all;
                                const rows = Object.entries(all).map(function([ts, v]) {
                                    return {
                                        date: new Date(Number(ts) * 1000).toISOString().slice(0, 10),
                                        value: v
                                    };
                                });
                                
                                callback({ 
                                    rows: rows, 
                                    source: correctUrl,
                                    watchId: watchId,
                                    success: true 
                                });
                            } catch (err) {
                                callback({ error: 'Error processing fetch response: ' + err.message, url: correctUrl });
                            }
                        })
                        .catch(function(err) {
                            callback({ error: 'Fetch failed: ' + (err.message || String(err)), url: correctUrl });
                        });
                } else {
                    callback({ error: 'Neither jQuery.getJSON nor fetch available' });
                }
            } catch (err) {
                callback({ error: 'Outer error: ' + (err.message || String(err)) });
            }
        })();
        """

        try:
            # Execute the extraction script directly without waiting for global url
            result = driver.execute_async_script(snippet)
            
            if isinstance(result, dict):
                if result.get("error"):
                    logger.warning(f"Console snippet extraction error: {result['error']}")
                    if result.get("url"):
                        logger.debug(f"Attempted URL: {result['url']}")
                    return None
                    
                rows = result.get("rows")
                if rows and result.get("success"):
                    source = result.get("source", "unknown")
                    watch_id = result.get("watchId", "unknown")
                    logger.info(
                        f"Successfully extracted {len(rows)} rows for watch {watch_id}"
                    )
                    
                    # Create DataFrame
                    df = pd.DataFrame(rows)
                    
                    # Ensure we have the right columns
                    if 'date' in df.columns and 'value' in df.columns:
                        # Rename value to price(SGD)
                        df = df.rename(columns={'value': 'price(SGD)'})
                        
                        # Convert date to datetime and sort
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date').reset_index(drop=True)
                        
                        # Log sample of data for verification
                        if len(df) > 0:
                            logger.info(
                                f"✅ Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}, "
                                f"Price range: SGD {df['price(SGD)'].min():.0f} to {df['price(SGD)'].max():.0f}"
                            )
                        
                        return df
                    else:
                        logger.error(f"Unexpected columns in extracted data: {df.columns.tolist()}")
                        
            logger.debug("Console snippet extraction did not return valid data")
            return None
            
        except Exception as exc:
            logger.error(f"Console snippet extraction failed with exception: {exc}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _extract_from_chart(self, driver: webdriver.Chrome) -> Optional[pd.DataFrame]:
        """Extract price data from Chart.js datasets if available."""
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
                    df = self._normalise_price_dataframe(df)
                    if df is not None and not df.empty:
                        self._record_extraction_metadata(
                            method="chart_dataset",
                            source="Chart.js instances",
                            extra={"row_count": len(df)}
                        )
                        return df
        except Exception as e:
            logger.error(f"Chart extraction failed: {e}")

        return None

    def _extract_from_json_endpoint(self, driver: webdriver.Chrome) -> Optional[pd.DataFrame]:
        """Extract price data by mimicking the in-page JSON request."""
        json_script = r"""
        var callback = arguments[arguments.length - 1];
        try {
            function findJsonUrl() {
                const chartContainer = document.querySelector('[data-chart-url]');
                if (chartContainer && chartContainer.dataset.chartUrl) {
                    return chartContainer.dataset.chartUrl;
                }

                const scripts = Array.from(document.querySelectorAll('script'));
                for (const script of scripts) {
                    const text = script.textContent || '';
                    const match = text.match(/\$\.getJSON\((['\"])(.*?)\1/);
                    if (match) {
                        return match[2];
                    }
                }

                if (window.performance && window.performance.getEntriesByType) {
                    const entries = window.performance.getEntriesByType('resource');
                    for (const entry of entries) {
                        if (/(history|chart|price)/i.test(entry.name)) {
                            return entry.name;
                        }
                    }
                }

                return null;
            }

            function toAbsolute(url) {
                try {
                    return new URL(url, window.location.href).href;
                } catch (err) {
                    return url;
                }
            }

            const url = findJsonUrl();
            if (!url) {
                callback(null);
                return;
            }

            const targetUrl = toAbsolute(url);

            function resolveRows(data) {
                try {
                    const all = data && data.data && data.data.all;
                    if (!all) {
                        callback({ error: 'Missing data.all', source: targetUrl });
                        return;
                    }

                    const rows = Object.entries(all).map(([ts, value]) => ({
                        date: new Date(Number(ts) * 1000).toISOString().slice(0, 10),
                        value: value
                    }));

                    callback({ rows: rows, source: targetUrl });
                } catch (err) {
                    callback({ error: err && err.message ? err.message : String(err), source: targetUrl });
                }
            }

            if (window.$ && window.$.getJSON) {
                window.$.getJSON(targetUrl)
                    .then(resolveRows)
                    .catch(err => callback({ error: err && err.message ? err.message : String(err), source: targetUrl }));
            } else if (window.fetch) {
                fetch(targetUrl)
                    .then(resp => resp.json())
                    .then(resolveRows)
                    .catch(err => callback({ error: err && err.message ? err.message : String(err), source: targetUrl }));
            } else {
                callback({ error: 'No fetch or jQuery available', source: targetUrl });
            }
        } catch (err) {
            callback({ error: err && err.message ? err.message : String(err) });
        }
        """

        try:
            json_result = driver.execute_async_script(json_script)
            if isinstance(json_result, dict):
                if json_result.get("error"):
                    logger.debug(
                        "JSON extraction issue from %s: %s",
                        json_result.get("source", "unknown"),
                        json_result["error"],
                    )
                rows = json_result.get("rows")
                if rows:
                    logger.debug(
                        "Fetched %s rows from %s via JSON endpoint",
                        len(rows),
                        json_result.get("source", "unknown"),
                    )
                    df = pd.DataFrame(rows)
                    df = self._normalise_price_dataframe(df)
                    if df is not None and not df.empty:
                        self._record_extraction_metadata(
                            method="json_endpoint",
                            source=json_result.get("source", "unknown"),
                            extra={"row_count": len(df)}
                        )
                        return df
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")

        return None

    def _normalise_price_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Ensure scraped data contains date and price(SGD) columns with clean types."""
        if df is None or df.empty:
            return None

        normalised = df.copy()

        price_column = None
        for candidate in ("price", "value", "y", "price_sgd", "price(SGD)"):
            if candidate in normalised.columns:
                price_column = candidate
                break

        if price_column is None:
            logger.debug("No price-like column found in scraped data columns=%s", normalised.columns.tolist())
            return None

        column_values = normalised[price_column]
        if column_values.apply(lambda x: isinstance(x, dict)).any():
            def extract_from_dict(val: Dict[str, float]) -> Optional[float]:
                if not isinstance(val, dict):
                    return None
                for key in ("price", "value", "avg", "mean", "median"):
                    if key in val and isinstance(val[key], (int, float)):
                        return val[key]
                return next((v for v in val.values() if isinstance(v, (int, float))), None)

            normalised[price_column] = column_values.apply(extract_from_dict)

        elif column_values.apply(lambda x: isinstance(x, (list, tuple))).any():
            normalised[price_column] = column_values.apply(
                lambda val: next((item for item in val if isinstance(item, (int, float))), None)
                if isinstance(val, (list, tuple))
                else val
            )

        if price_column != "price(SGD)":
            logger.debug("Normalising price column '%s'", price_column)
        normalised.rename(columns={price_column: "price(SGD)"}, inplace=True)

        try:
            normalised["price(SGD)"] = pd.to_numeric(normalised["price(SGD)"], errors="coerce")
        except Exception as exc:
            logger.debug("Failed numeric conversion for price column: %s", exc)

        if "date" not in normalised.columns:
            logger.debug("No date column found in scraped data columns=%s", normalised.columns.tolist())
            return None

        normalised["date"] = pd.to_datetime(normalised["date"], errors="coerce")

        pre_drop_len = len(normalised)
        normalised = normalised.dropna(subset=["date", "price(SGD)"])
        if len(normalised) != pre_drop_len:
            logger.debug("Dropped %s rows with missing date/price", pre_drop_len - len(normalised))

        if normalised.empty:
            logger.debug("Normalised price dataframe became empty after dropping invalid rows")
            return None

        normalised["date"] = normalised["date"].dt.strftime("%Y-%m-%d")
        normalised = normalised.sort_values("date").reset_index(drop=True)
        logger.debug("Normalised dataframe rows=%s", len(normalised))

        return normalised

    def _record_extraction_metadata(self, method: str, source: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Store metadata about the latest extraction for logging."""
        self._last_extraction_metadata = {
            "method": method,
            "source": source,
            "extra": extra or {},
        }

    def _log_extraction_summary(self, method: str, df: pd.DataFrame) -> None:
        """Log a concise summary of the extracted price data."""
        metadata = getattr(self, "_last_extraction_metadata", {}) or {}
        source = metadata.get("source", "unknown")
        row_count = len(df)
        price_min = df["price(SGD)"].min() if "price(SGD)" in df.columns else None
        price_max = df["price(SGD)"].max() if "price(SGD)" in df.columns else None
        date_min = df["date"].min() if "date" in df.columns else None
        date_max = df["date"].max() if "date" in df.columns else None

        logger.info(
            "Extracted price data via %s from %s | rows=%s date_range=%s→%s price_range=%s→%s",
            method,
            source,
            row_count,
            date_min,
            date_max,
            price_min,
            price_max,
        )
    
    def _merge_data(
        self, 
        new_data: pd.DataFrame, 
        existing_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge new data with existing data."""
        if existing_data is None or existing_data.empty:
            return new_data

        new_data = new_data.copy()
        existing_data = existing_data.copy()

        # Convert dates to datetime to enable sorting/deduplication
        new_data["date"] = pd.to_datetime(new_data["date"])
        existing_data["date"] = pd.to_datetime(existing_data["date"])

        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        new_rows_count = len(combined) - len(existing_data)
        if new_rows_count > 0:
            logger.info(f"Merged dataset added {new_rows_count} rows")
        else:
            logger.info("Merged dataset produced no additional rows; retaining existing data")

        return combined
    
    def _build_filename(self, watch: WatchTarget) -> str:
        """Build filename for watch data."""
        brand_safe = make_filename_safe(watch.brand)
        model_safe = make_filename_safe(watch.model_name)
        watch_id = watch.watch_id or watch.model_id or "unknown"
        
        return f"{brand_safe}-{model_safe}-{watch_id}.csv"
    
    def _load_existing_data(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load existing CSV data."""
        if filepath.exists():
            return load_existing_csv_data(str(filepath))
        return None
    
    def _save_data(self, data: pd.DataFrame, filepath: Path) -> bool:
        """Save data to CSV file."""
        # Convert dates to string format
        if pd.api.types.is_datetime64_any_dtype(data["date"]):
            data["date"] = data["date"].dt.strftime("%Y-%m-%d")
        
        return safe_write_csv_with_backup(data, str(filepath), index=False)
    
    def _save_error_screenshot(self, driver: Optional[webdriver.Chrome], watch: WatchTarget) -> None:
        """Save screenshot on error."""
        if not driver:
            return
            
        try:
            error_dir = self.output_dir / "errors"
            error_dir.mkdir(exist_ok=True)
            
            screenshot_file = error_dir / f"{watch.watch_id}_error.png"
            driver.save_screenshot(str(screenshot_file))
            logger.info(f"Error screenshot saved: {screenshot_file}")
        except Exception as e:
            logger.debug(f"Failed to save screenshot: {e}")
    
    def _random_delay(self) -> None:
        """Add random delay."""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def _print_summary(self, results: Dict[str, ScrapingResult]) -> None:
        """Print scraping summary."""
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        
        logger.info("\n" + "="*60)
        logger.info("📊 SCRAPING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total watches: {len(results)}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"❌ Failed: {failed}")
        
        if len(results) > 0:
            success_rate = (successful / len(results)) * 100
            logger.info(f"📈 Success rate: {success_rate:.1f}%")
