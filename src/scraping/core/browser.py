"""
Browser management and Selenium utilities for watch scraping.
Consolidated from selenium_utils.py with configuration support.
"""

import logging
import random
import time
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

try:
    import undetected_chromedriver as uc
    UC_AVAILABLE = True
except ImportError:
    UC_AVAILABLE = False

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
        """Create driver optimized for URL discovery tasks with enhanced Cloudflare bypass."""
        if UC_AVAILABLE:
            logger.info("Using undetected-chromedriver for enhanced Cloudflare bypass")
            try:
                # Create undetected Chrome driver with minimal options
                driver = uc.Chrome(
                    headless=headless,
                    use_subprocess=False,
                    version_main=None,  # Auto-detect Chrome version
                )
                
                # Set window size
                driver.set_window_size(1920, 1080)
                
                logger.info("Successfully created undetected discovery Chrome driver")
                return driver
                
            except Exception as e:
                logger.warning(f"Failed to create undetected driver, falling back to regular driver: {e}")
        
        # Fallback to regular Chrome driver
        logger.info("Using regular Chrome driver with enhanced stealth")
        options = Options()

        if headless:
            options.add_argument("--headless=new")  # Use new headless mode

        # Enhanced anti-detection for discovery
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # More realistic browser fingerprint with updated user agent
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        
        # Enhanced stealth options for Cloudflare bypass
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-running-insecure-content")
        options.add_argument("--disable-extensions")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-ipc-flooding-protection")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-translate")
        options.add_argument("--disable-logging")
        options.add_argument("--disable-plugins-discovery")
        options.add_argument("--disable-preconnect")
        options.add_argument("--disable-component-extensions-with-background-pages")
        
        # Additional Cloudflare bypass options
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values": {
                "notifications": 2,
                "geolocation": 2,
                "media_stream": 2,
                "plugins": 2
            },
            "profile.managed_default_content_settings": {
                "images": 1
            }
        })

        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )

            # Enhanced stealth script for Cloudflare bypass
            stealth_script = """
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            delete navigator.__proto__.webdriver;
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format", filename: "internal-pdf-viewer"}},
                    {1: {type: "application/pdf", suffixes: "pdf", description: "Portable Document Format", filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai"}},
                    {2: {type: "application/x-nacl", suffixes: "", description: "Native Client Executable", filename: "internal-nacl-plugin"}},
                    {3: {type: "application/x-pnacl", suffixes: "", description: "Portable Native Client Executable", filename: "internal-nacl-plugin"}}
                ]
            });
            
            // Enhanced navigator properties
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'platform', {get: () => 'Linux x86_64'});
            Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
            Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
            Object.defineProperty(navigator, 'doNotTrack', {get: () => null});
            Object.defineProperty(navigator, 'maxTouchPoints', {get: () => 0});
            Object.defineProperty(navigator, 'vendor', {get: () => 'Google Inc.'});
            Object.defineProperty(navigator, 'vendorSub', {get: () => ''});
            
            // Screen properties
            Object.defineProperty(screen, 'width', {get: () => 1920});
            Object.defineProperty(screen, 'height', {get: () => 1080});
            Object.defineProperty(screen, 'availWidth', {get: () => 1920});
            Object.defineProperty(screen, 'availHeight', {get: () => 1040});
            Object.defineProperty(screen, 'colorDepth', {get: () => 24});
            Object.defineProperty(screen, 'pixelDepth', {get: () => 24});
            
            // Chrome object
            window.chrome = {
                runtime: {},
                app: {isInstalled: false, getDetails: function() {return null;}},
                csi: function(){},
                loadTimes: function(){return {requestTime: Date.now()/1000, startLoadTime: Date.now()/1000, commitLoadTime: Date.now()/1000, finishDocumentLoadTime: Date.now()/1000, finishLoadTime: Date.now()/1000, firstPaintTime: Date.now()/1000, firstPaintAfterLoadTime: 0, navigationType: 'Other'};},
            };
            
            // Permissions API mock
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );
            
            // Remove automation indicators
            ['__nightmare', '__webdriver_evaluate', '__selenium_evaluate', '__webdriver_script_function', '__webdriver_script_func', '__webdriver_script_fn', '__fxdriver_evaluate', '__driver_unwrapped', '__webdriver_unwrapped', '__driver_evaluate', '__selenium_unwrapped', '__fxdriver_unwrapped'].forEach(prop => {
                delete window[prop];
            });
            """
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument", {"source": stealth_script}
            )
            
            # Set realistic viewport
            driver.set_window_size(1920, 1080)

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


def simulate_human_behavior(driver: webdriver.Chrome) -> None:
    """Simulate human-like behavior to bypass detection."""
    try:
        import random
        from selenium.webdriver.common.action_chains import ActionChains
        
        # Random mouse movements
        actions = ActionChains(driver)
        for _ in range(3):
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            actions.move_by_offset(x, y)
            actions.pause(random.uniform(0.1, 0.3))
        
        # Perform the actions
        try:
            actions.perform()
        except:
            pass  # Ignore errors in mouse simulation
            
        # Random scroll
        scroll_amount = random.randint(100, 500)
        driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
        time.sleep(random.uniform(0.5, 1.5))
        driver.execute_script("window.scrollTo(0, 0);")
        
    except Exception as e:
        logger.debug(f"Human behavior simulation failed: {e}")


def wait_for_cloudflare_challenge(driver: webdriver.Chrome, max_wait: int = 60) -> bool:
    """Wait for Cloudflare challenge to complete with extended timeout."""
    import time
    
    start_time = time.time()
    check_interval = 2
    
    logger.info(f"Waiting for Cloudflare challenge to complete (max {max_wait}s)...")
    
    while time.time() - start_time < max_wait:
        try:
            # Check if challenge is still active
            if not check_cloudflare_challenge(driver):
                logger.info("Cloudflare challenge appears to be resolved")
                # Additional wait to ensure page is fully loaded
                time.sleep(3)
                return True
                
            # Simulate human behavior while waiting
            if random.randint(1, 3) == 1:  # 33% chance
                simulate_human_behavior(driver)
                
            time.sleep(check_interval)
            
        except Exception as e:
            logger.debug(f"Error during Cloudflare wait: {e}")
            time.sleep(check_interval)
    
    logger.warning(f"Cloudflare challenge did not resolve within {max_wait} seconds")
    return False


def check_cloudflare_challenge(driver: webdriver.Chrome) -> bool:
    """Check if the current page contains an active Cloudflare challenge."""
    try:
        page_source = driver.page_source.lower()
        page_title = driver.title.lower()

        # First check for positive indicators that we're on the actual site
        success_indicators = ["watchcharts", "chart", "price", "watch", "overview", "pagination"]
        success_count = sum(1 for indicator in success_indicators if indicator in page_source)
        
        # If we have multiple success indicators, we're likely past Cloudflare
        if success_count >= 2:
            return False

        # Only check for active challenge indicators, not just "cloudflare" presence
        active_challenge_indicators = [
            "checking your browser",
            "please wait while we check your browser", 
            "cf-browser-verification",
            "just a moment",
            "enable javascript and cookies",
            "challenge-running",
            "challenge-stage", 
            "cf-challenge-running",
            "verify you are human",
        ]

        # Check for active challenge indicators in page source
        for indicator in active_challenge_indicators:
            if indicator in page_source:
                logger.warning(
                    f"Cloudflare challenge detected: '{indicator}' found in page"
                )
                return True

        # Check title for active challenge indicators
        active_title_indicators = ["attention required", "just a moment", "security check"]
        for indicator in active_title_indicators:
            if indicator in page_title:
                logger.warning(
                    f"Cloudflare challenge detected: '{indicator}' found in title"
                )
                return True
                
        # Check for active challenge elements (more specific selectors)
        try:
            active_cf_elements = driver.find_elements(By.CSS_SELECTOR, 
                "[class*='cf-challenge'], [id*='cf-challenge'], [class*='challenge-running']")
            if active_cf_elements:
                logger.warning("Active Cloudflare challenge detected: challenge elements found")
                return True
        except:
            pass

        return False

    except Exception as e:
        logger.error(f"Error checking for Cloudflare challenge: {e}")
        return False
