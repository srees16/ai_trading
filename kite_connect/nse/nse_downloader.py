"""
Downloads live equity market data CSV from NSE India.

Strategy:
  1. Use Selenium to open NSE and build a valid cookie session
  2. Wait for the table to fully render
  3. Click the Download (.csv) button
  4. Wait and verify the file is actually downloaded
  
URL: https://www.nseindia.com/market-data/live-equity-market
"""

import os
import sys
import time
import glob
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import NSE_URL, DOWNLOAD_DIR
from core.selenium_service import get_driver


def wait_for_download(directory, prefix="MW-", timeout=60):
    """Wait until a new CSV file with the given prefix appears and finishes downloading."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        files = glob.glob(os.path.join(directory, f"{prefix}*.csv"))
        if files:
            latest = max(files, key=os.path.getmtime)
            # Make sure it's not still being written (no .crdownload companion)
            if not os.path.exists(latest + ".crdownload"):
                # Verify file has content
                if os.path.getsize(latest) > 100:
                    return latest
        time.sleep(1)
    return None


def download_nse_csv():
    """Open NSE live equity page, wait for data, click Download CSV."""
    print("=" * 60)
    print("  NSE Live Equity Market - CSV Download")
    print("=" * 60)

    # Remove old MW-*.csv files from Downloads so we can detect the new one
    old_files = glob.glob(os.path.join(DOWNLOAD_DIR, "MW-*.csv"))

    driver = get_driver(download_dir=DOWNLOAD_DIR, stealth=True)
    csv_path = None

    try:
        # Step 1: Visit NSE homepage first to establish cookies/session
        print("\n  Step 1: Establishing NSE session (visiting homepage)...")
        driver.get("https://www.nseindia.com")
        time.sleep(3)

        # Step 2: Navigate to the live equity market page
        print("  Step 2: Opening live equity market page...")
        driver.get(NSE_URL)

        wait = WebDriverWait(driver, 40)

        # Step 3: Wait for the equity table to render
        print("  Step 3: Waiting for market data table to load...")
        try:
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "table#liveEquityTable, table.common-table, #equityStockTable")
            ))
            print("  [OK] Table found on page")
        except Exception:
            print("  [!] Table selector not found, waiting extra time...")

        # Give JS more time to populate the table data
        time.sleep(8)

        # Step 4: Find and click the download button
        print("  Step 4: Looking for Download (.csv) button...")
        download_btn = None

        selectors = [
            "#dwldcsv",
            "a#dwldcsv",
            "#liveEquityMarket a#dwldcsv",
            "#cr_equity_tab_content a#dwldcsv",
            "a[id='dwldcsv']",
        ]

        for selector in selectors:
            try:
                download_btn = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if download_btn:
                    print(f"  [OK] Found button: {selector}")
                    break
            except Exception:
                continue

        if download_btn is None:
            # Try XPath for any download-related links
            try:
                download_btn = driver.find_element(
                    By.XPATH,
                    "//a[contains(@id,'dwld') or contains(@id,'csv') or contains(text(),'Download')]"
                )
                print(f"  [OK] Found button via XPath")
            except Exception:
                pass

        if download_btn is None:
            # Debug: print all links on the page
            links = driver.find_elements(By.TAG_NAME, "a")
            print(f"\n  [DEBUG] Found {len(links)} links. Searching for download-related...")
            for link in links:
                link_id = link.get_attribute("id") or ""
                link_class = link.get_attribute("class") or ""
                link_text = link.text.strip()
                if any(k in (link_id + link_class + link_text).lower() for k in ["download", "csv", "dwld"]):
                    print(f"    <a id='{link_id}' class='{link_class}'>{link_text}</a>")
                    if download_btn is None:
                        download_btn = link

        if download_btn is None:
            raise RuntimeError("Could not find the Download CSV button")

        # Scroll to the button and click
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", download_btn)
        time.sleep(1)

        print("  Clicking Download (.csv)...")
        try:
            download_btn.click()
        except Exception:
            driver.execute_script("arguments[0].click();", download_btn)

        # Step 5: Wait for file to appear
        print("  Step 5: Waiting for download to complete...")
        csv_path = wait_for_download(DOWNLOAD_DIR, prefix="MW-", timeout=45)

        if csv_path:
            print(f"\n  [OK] Downloaded: {os.path.basename(csv_path)}")
            print(f"  [OK] Location:   {csv_path}")
            print(f"  [OK] Size:       {os.path.getsize(csv_path):,} bytes")
        else:
            # Check if it went to kite_connect folder as fallback
            script_dir = os.path.abspath(os.path.dirname(__file__))
            alt = wait_for_download(script_dir, prefix="MW-", timeout=5)
            if alt:
                csv_path = alt
                print(f"\n  [OK] Downloaded: {csv_path}")
            else:
                print("\n  [!] CSV file not found after waiting.")
                print("  [!] Please check your browser's download location manually.")

    except Exception as e:
        print(f"\n  [ERROR] {e}")

    finally:
        time.sleep(1)
        driver.quit()
        print("  [OK] Browser closed.")

    print("\n  Done!")
    return csv_path


if __name__ == "__main__":
    download_nse_csv()
