"""
Shared Selenium browser utilities for the kite_connect package.

Consolidates the Chrome / Edge driver creation logic used by both
``download_nse_csv.py`` and ``get_request_token.py``.
"""

import logging
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.service import Service as EdgeService

# Suppress noisy Selenium driver-manager HTTP warnings
logging.getLogger("selenium.webdriver.common.selenium_manager").setLevel(logging.ERROR)

# Optional: use webdriver-manager if installed (avoids network probe entirely)
try:
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    _WDM_AVAILABLE = True
except ImportError:
    _WDM_AVAILABLE = False


def _resolve_chrome_service():
    """Return a ChromeService, preferring webdriver-manager when available."""
    if _WDM_AVAILABLE:
        try:
            return ChromeService(ChromeDriverManager().install())
        except Exception:
            pass
    return ChromeService()      # fallback to Selenium built-in manager / cache


def _resolve_edge_service():
    """Return an EdgeService, preferring webdriver-manager when available."""
    if _WDM_AVAILABLE:
        try:
            return EdgeService(EdgeChromiumDriverManager().install())
        except Exception:
            pass
    return EdgeService()


def get_driver(download_dir=None, stealth=False):
    """
    Create and return a Selenium WebDriver (Chrome first, then Edge).

    Parameters
    ----------
    download_dir : str, optional
        If provided, the browser will download files to this directory
        without prompting.  Used by the NSE CSV downloader.
    stealth : bool, optional
        If ``True``, apply anti-bot / anti-automation tweaks (custom
        user-agent, hide ``navigator.webdriver`` flag, etc.).  Needed
        for sites like NSE that block automated browsers.

    Returns
    -------
    selenium.webdriver.Chrome | selenium.webdriver.Edge
    """
    browsers = [
        ("Chrome", _chrome_options, webdriver.Chrome, _resolve_chrome_service),
        ("Edge",   _edge_options,   webdriver.Edge,   _resolve_edge_service),
    ]

    last_error = None
    for name, options_fn, driver_cls, service_fn in browsers:
        try:
            options = options_fn(download_dir, stealth)
            service = service_fn()
            driver = driver_cls(service=service, options=options)

            if stealth:
                # Hide webdriver flag from JavaScript
                driver.execute_cdp_cmd(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": "Object.defineProperty(navigator, 'webdriver', "
                               "{get: () => undefined})"},
                )

            print(f"  Using: {name}")
            return driver
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"No Selenium-compatible browser found (Chrome or Edge). Last error: {last_error}"
    )


# ── Private helpers ────────────────────────────────────────────

def _common_options(options, download_dir, stealth):
    """Apply options shared between Chrome and Edge."""
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")

    if download_dir:
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        options.add_experimental_option("prefs", prefs)

    if stealth:
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )

    return options


def _chrome_options(download_dir, stealth):
    from selenium.webdriver.chrome.options import Options
    return _common_options(Options(), download_dir, stealth)


def _edge_options(download_dir, stealth):
    from selenium.webdriver.edge.options import Options
    return _common_options(Options(), download_dir, stealth)
