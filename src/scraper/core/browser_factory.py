"""
Browser factory for creating configured Selenium drivers.
Implements factory pattern for browser creation.
"""

import logging
import random
import time
from typing import Optional, Dict, Any
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class BrowserFactory:
    """Factory for creating browser instances with different configurations."""
    
    # Stealth JavaScript to avoid detection
    STEALTH_JS = """
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    delete navigator.__proto__.webdriver;
    
    Object.defineProperty(navigator, 'plugins', {
        get: () => [
            {0: {type: "application/x-google-chrome-pdf", suffixes: "pdf"}},
            {1: {type: "application/pdf", suffixes: "pdf"}}
        ]
    });
    
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
    Object.defineProperty(navigator, 'platform', {get: () => 'Linux x86_64'});
    Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
    Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
    
    window.chrome = {
        runtime: {},
        app: {isInstalled: false}
    };
    """
    
    @staticmethod
    def create_driver(driver_type: str = "standard", **kwargs) -> webdriver.Chrome:
        """
        Create a browser driver based on type.
        
        Args:
            driver_type: Type of driver ("standard", "discovery", "scraping")
            **kwargs: Additional configuration options
            
        Returns:
            Configured Chrome driver
        """
        if driver_type == "discovery":
            return BrowserFactory._create_discovery_driver(**kwargs)
        elif driver_type == "scraping":
            return BrowserFactory._create_scraping_driver(**kwargs)
        else:
            return BrowserFactory._create_standard_driver(**kwargs)
    
    @staticmethod
    def _create_standard_driver(headless: bool = True) -> webdriver.Chrome:
        """Create standard Chrome driver."""
        options = BrowserFactory._get_base_options(headless)
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        BrowserFactory._configure_driver(driver)
        return driver
    
    @staticmethod
    def _create_discovery_driver(headless: bool = False) -> webdriver.Chrome:
        """Create discovery driver mirroring the working urls.py Selenium setup."""
        options = BrowserFactory._get_urls_style_options(headless)

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options,
        )

        BrowserFactory._configure_driver(driver)
        return driver
    
    @staticmethod
    def _create_scraping_driver(headless: bool = True) -> webdriver.Chrome:
        """Create driver optimized for price data scraping."""
        options = BrowserFactory._get_enhanced_options(headless)
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        BrowserFactory._inject_stealth_script(driver)
        BrowserFactory._configure_driver(driver)
        return driver
    
    @staticmethod
    def _get_base_options(headless: bool) -> Options:
        """Get basic Chrome options."""
        options = Options()
        
        if headless:
            options.add_argument("--headless=new")
        
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.page_load_strategy = "eager"
        
        return options
    
    @staticmethod
    def _get_enhanced_options(headless: bool) -> Options:
        """Get enhanced Chrome options with anti-detection features."""
        options = BrowserFactory._get_base_options(headless)

        # Anti-detection options
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        # User agent
        options.add_argument(
            "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        
        # Additional stealth options
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-gpu")
        
        # Preferences
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values": {
                "notifications": 2,
                "geolocation": 2
            }
        })
        
        return options

    @staticmethod
    def _get_urls_style_options(headless: bool) -> Options:
        """Chrome options aligned with the standalone urls scraper."""
        options = webdriver.ChromeOptions()

        if headless:
            options.add_argument("--headless=new")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        options.page_load_strategy = "eager"

        return options

    @staticmethod
    def _inject_stealth_script(driver: webdriver.Chrome) -> None:
        """Inject stealth JavaScript to avoid detection."""
        try:
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": BrowserFactory.STEALTH_JS}
            )
        except Exception as e:
            logger.debug(f"Failed to inject stealth script: {e}")
    
    @staticmethod
    def _configure_driver(driver: webdriver.Chrome) -> None:
        """Configure driver with timeouts and window size."""
        driver.set_page_load_timeout(20)  # Increased from 20 to 60 seconds
        driver.implicitly_wait(15)  # Increased from 10 to 15 seconds
        driver.set_window_size(1920, 1080)


class CloudflareHandler:
    """Handles Cloudflare challenges and detection."""
    
    @staticmethod
    def check_challenge(driver: webdriver.Chrome) -> bool:
        """Check if page has Cloudflare challenge."""
        try:
            page_source = driver.page_source.lower()
            page_title = driver.title.lower()
            
            # Check for success indicators first
            success_indicators = ["watchcharts", "chart", "price", "watch"]
            if sum(1 for ind in success_indicators if ind in page_source) >= 2:
                return False
            
            # Check for challenge indicators
            challenge_indicators = [
                "checking your browser",
                "just a moment",
                "cf-browser-verification",
                "verify you are human"
            ]
            
            for indicator in challenge_indicators:
                if indicator in page_source or indicator in page_title:
                    logger.warning(f"Cloudflare challenge detected: '{indicator}'")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Cloudflare: {e}")
            return False
    
    @staticmethod
    def wait_for_challenge(driver: webdriver.Chrome, max_wait: int = 60) -> bool:
        """Wait for Cloudflare challenge to complete."""
        start_time = time.time()
        
        logger.info(f"Waiting for Cloudflare challenge (max {max_wait}s)...")
        
        while time.time() - start_time < max_wait:
            if not CloudflareHandler.check_challenge(driver):
                logger.info("Cloudflare challenge resolved")
                time.sleep(2)
                return True
            
            CloudflareHandler.simulate_human_behavior(driver)
            time.sleep(2)
        
        logger.warning("Cloudflare challenge timeout")
        return False
    
    @staticmethod
    def simulate_human_behavior(driver: webdriver.Chrome) -> None:
        """Simulate human-like behavior."""
        try:
            # Random scroll
            scroll = random.randint(100, 300)
            driver.execute_script(f"window.scrollBy(0, {scroll});")
            time.sleep(random.uniform(0.5, 1.0))
            
        except Exception:
            pass  # Ignore errors in simulation
