"""
URL extraction strategies using Strategy pattern.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Set, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

logger = logging.getLogger(__name__)


class ExtractionStrategy(ABC):
    """Abstract base class for URL extraction strategies."""
    
    @abstractmethod
    def extract(self, driver: webdriver.Chrome, model_fragment: str, max_urls: int) -> Set[str]:
        """Extract URLs from the page."""
        pass


class JavaScriptExtractor(ExtractionStrategy):
    """Extract URLs using JavaScript execution."""
    
    def extract(self, driver: webdriver.Chrome, model_fragment: str, max_urls: int) -> Set[str]:
        """Extract watch URLs using JavaScript."""
        urls = set()
        
        js_script = """
        var links = [];
        var anchors = document.querySelectorAll('a[href*="/watch_model/"]');
        for (var i = 0; i < anchors.length; i++) {
            links.push(anchors[i].href);
        }
        return links;
        """
        
        try:
            js_links = driver.execute_script(js_script)
            for href in js_links:
                if not href:
                    continue
                    
                href_lower = str(href).lower()
                if model_fragment in href_lower:
                    urls.add(href)
                    if len(urls) >= max_urls:
                        break
                        
            logger.info(f"JavaScript extractor found {len(urls)} URLs")
            
        except Exception as e:
            logger.warning(f"JavaScript extraction failed: {e}")
            
        return urls


class XPathExtractor(ExtractionStrategy):
    """Extract URLs using XPath selectors."""
    
    def extract(self, driver: webdriver.Chrome, model_fragment: str, max_urls: int) -> Set[str]:
        """Extract watch URLs using XPath."""
        urls = set()
        
        try:
            # Check element count first
            count_script = """
            return document.querySelectorAll('a[href*="/watch_model/"]').length;
            """
            element_count = driver.execute_script(count_script)
            
            if element_count > 0:
                logger.debug(f"Found {element_count} candidate watch_model anchors")
                
                watch_links = driver.find_elements(
                    By.XPATH, "//a[contains(@href, '/watch_model/')]"
                )
                
                for link in watch_links:
                    try:
                        href = link.get_attribute("href")
                    except NoSuchElementException:
                        continue
                        
                    if not href:
                        continue
                        
                    href_lower = href.lower()
                    if model_fragment in href_lower:
                        urls.add(href)
                        if len(urls) >= max_urls:
                            break
                            
            logger.info(f"XPath extractor found {len(urls)} URLs")
            
        except Exception as e:
            logger.warning(f"XPath extraction failed: {e}")
            
        return urls


class BroadSearchExtractor(ExtractionStrategy):
    """Broad search fallback for finding URLs."""
    
    def extract(self, driver: webdriver.Chrome, model_fragment: str, max_urls: int) -> Set[str]:
        """Extract URLs using broad search patterns."""
        urls = set()
        
        try:
            all_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='watches']")[:50]
            
            for link in all_links:
                try:
                    href = link.get_attribute("href")
                except NoSuchElementException:
                    continue
                    
                if not href:
                    continue
                    
                href_lower = href.lower()
                if model_fragment in href_lower and "/brand/" not in href_lower:
                    urls.add(href)
                    if len(urls) >= max_urls:
                        break
                        
            logger.info(f"Broad search found {len(urls)} URLs")
            
        except Exception as e:
            logger.warning(f"Broad search failed: {e}")
            
        return urls


class URLExtractorChain:
    """Chain of responsibility for URL extraction."""
    
    def __init__(self):
        """Initialize extractor chain."""
        self.strategies = [
            JavaScriptExtractor(),
            XPathExtractor(),
            BroadSearchExtractor()
        ]
    
    def extract_urls(
        self, 
        driver: webdriver.Chrome, 
        model_fragment: str, 
        max_urls: int
    ) -> Set[str]:
        """
        Extract URLs using multiple strategies until enough are found.
        
        Args:
            driver: Selenium WebDriver instance
            model_fragment: Model/brand fragment to filter URLs
            max_urls: Maximum number of URLs to extract
            
        Returns:
            Set of extracted URLs
        """
        all_urls = set()
        
        for index, strategy in enumerate(self.strategies):
            remaining = max_urls - len(all_urls)
            if remaining <= 0:
                break

            strategy_name = strategy.__class__.__name__
            urls = strategy.extract(driver, model_fragment, remaining)
            before_count = len(all_urls)
            all_urls.update(urls)

            obtained = len(all_urls) - before_count
            if obtained:
                logger.debug(
                    "%s added %d URL(s); collected %d / %d",
                    strategy_name,
                    obtained,
                    len(all_urls),
                    max_urls,
                )

            if len(all_urls) >= max_urls:
                break

            if obtained == 0 and index < len(self.strategies) - 1:
                logger.info(
                    "%s yielded no URLs; trying fallback strategy",
                    strategy_name,
                )
            elif obtained and len(all_urls) < max_urls and index < len(self.strategies) - 1:
                logger.info(
                    "%s found partial results; attempting fallback for more coverage",
                    strategy_name,
                )

        return all_urls


class WatchDataParser:
    """Parse watch data from URLs."""
    
    @staticmethod
    def parse_watch_url(url: str, brand: str) -> dict:
        """
        Parse watch information from URL.
        
        Args:
            url: Watch URL to parse
            brand: Brand name
            
        Returns:
            Dictionary with watch information
        """
        watch_data = {
            'url': url,
            'brand': brand,
            'model_id': '',
            'model_name': '',
            'watch_id': ''
        }
        
        try:
            if "/watch_model/" in url:
                # Extract from watch_model URL pattern
                parts = url.split("/watch_model/")[1].split("/")[0]
                model_parts = parts.split("-", 2)
                
                if len(model_parts) >= 1:
                    watch_data['watch_id'] = model_parts[0]
                    watch_data['model_id'] = model_parts[0]
                
                if len(model_parts) >= 3:
                    # Format: ID-brand-model
                    model_name = model_parts[2].replace("-", " ").title()
                    watch_data['model_name'] = model_name
                elif len(model_parts) >= 2:
                    # Format: ID-model
                    model_name = "-".join(model_parts[1:]).replace("-", " ").title()
                    watch_data['model_name'] = model_name
            else:
                # Fallback parsing for non-standard URLs
                model_name = url.split("/")[-1].replace("-", " ").title()
                watch_data['model_name'] = model_name
                
        except Exception as e:
            logger.debug(f"Error parsing URL {url}: {e}")
        
        return watch_data
    
    @staticmethod
    def normalize_brand_slug(brand: str) -> str:
        """Normalize brand name to URL slug format."""
        return re.sub(r"[^a-z0-9]+", "-", brand.lower()).strip("-")
