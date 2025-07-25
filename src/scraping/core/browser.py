"""
Browser management and Selenium utilities for watch scraping.
Consolidated from selenium_utils.py with configuration support.
"""

import logging
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class BrowserManager:
    """Manages browser instances and configuration for scraping tasks."""

    @staticmethod
    def create_scraping_driver(headless: bool = True) -> webdriver.Chrome:
        """Create a Chrome driver optimized for price data scraping."""
        options = Options()

        if headless:
            options.add_argument("--headless")

        # Anti-detection options for bypassing Cloudflare
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # Realistic browser profile
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        options.add_argument("--disable-extensions")

        # Performance optimizations
        options.add_argument("--no-first-run")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-backgrounding-occluded-windows")

        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )

            # Execute stealth script to avoid detection
            stealth_script = """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
            """
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument", {"source": stealth_script}
            )

            logger.info("Successfully created scraping Chrome driver")
            return driver

        except Exception as e:
            logger.error(f"Failed to create Chrome driver: {e}")
            raise

    @staticmethod
    def create_discovery_driver(headless: bool = True) -> webdriver.Chrome:
        """Create driver optimized for URL discovery tasks."""
        options = Options()

        if headless:
            options.add_argument("--headless")

        # Basic anti-detection (less aggressive for discovery)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )
            logger.info("Successfully created discovery Chrome driver")
            return driver

        except Exception as e:
            logger.error(f"Failed to create discovery driver: {e}")
            raise

    @staticmethod
    def quit_driver_safely(driver: Optional[webdriver.Chrome]) -> None:
        """Safely quit a Selenium driver with error handling."""
        if driver:
            try:
                driver.quit()
                logger.debug("Driver quit successfully")
            except Exception as e:
                logger.warning(f"Error quitting driver: {e}")


def check_page_loaded_successfully(
    driver: webdriver.Chrome, expected_domain: str = "watchcharts.com"
) -> bool:
    """Check if the website loaded successfully (not a Cloudflare page or error)."""
    try:
        current_url = driver.current_url.lower()
        page_source = driver.page_source.lower()

        # Check if we're on the expected domain
        if expected_domain not in current_url:
            logger.warning(f"Not on expected domain. Current URL: {current_url}")
            return False

        # Check for successful WatchCharts content
        success_indicators = ["watchcharts", "chart", "price", "watch", "overview"]

        success_count = sum(
            1 for indicator in success_indicators if indicator in page_source
        )

        if success_count >= 2:  # At least 2 success indicators
            logger.info("Website loaded successfully")
            return True
        else:
            logger.warning(
                f"Website may not have loaded properly. Found {success_count} success indicators"
            )
            return False

    except Exception as e:
        logger.error(f"Error checking website load status: {e}")
        return False


def check_cloudflare_challenge(driver: webdriver.Chrome) -> bool:
    """Check if the current page contains a Cloudflare challenge."""
    try:
        page_source = driver.page_source.lower()
        page_title = driver.title.lower()

        cloudflare_indicators = [
            "checking your browser",
            "please wait while we check your browser",
            "cf-browser-verification",
            "cloudflare",
            "just a moment",
            "enable javascript and cookies",
            "ray id",
        ]

        # Also check title for Cloudflare indicators
        title_indicators = ["attention required", "cloudflare", "just a moment"]

        # Check page source
        for indicator in cloudflare_indicators:
            if indicator in page_source:
                logger.warning(
                    f"Cloudflare challenge detected: '{indicator}' found in page"
                )
                return True

        # Check page title
        for indicator in title_indicators:
            if indicator in page_title:
                logger.warning(
                    f"Cloudflare challenge detected: '{indicator}' found in title"
                )
                return True

        return False

    except Exception as e:
        logger.error(f"Error checking for Cloudflare challenge: {e}")
        return False
