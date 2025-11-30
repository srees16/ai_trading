"""
Finviz news scraper with Elite authentication support.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from scrapers import BaseNewsScraper
from models import NewsItem
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time


class FinvizScraper(BaseNewsScraper):
    """Scraper for Finviz news with Elite authentication."""
    
    # Elite credentials (optional - only needed for Elite features)
    FINVIZ_USERNAME = "s.srees@live.com"
    FINVIZ_PASSWORD = "Imbest1!"
    
    def __init__(self, use_elite: bool = False):
        super().__init__("Finviz", "https://finviz.com/quote.ashx?t={}")
        self.use_elite = use_elite
        self.driver: Optional[webdriver.Chrome] = None
        self._authenticated = False
    
    def _init_selenium_driver(self):
        """Initialize Selenium driver for Elite access."""
        if self.driver is not None:
            return
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def _authenticate_elite(self):
        """Authenticate with Finviz Elite."""
        if self._authenticated or not self.use_elite:
            return True
        
        try:
            self._init_selenium_driver()
            self.driver.get("https://elite.finviz.com/login.ashx")
            time.sleep(3)
            
            wait = WebDriverWait(self.driver, 15)
            email_input = wait.until(EC.presence_of_element_located((By.NAME, "email")))
            password_input = wait.until(EC.presence_of_element_located((By.NAME, "password")))
            
            email_input.send_keys(self.FINVIZ_USERNAME)
            password_input.send_keys(self.FINVIZ_PASSWORD)
            password_input.submit()
            
            wait.until(EC.presence_of_element_located((By.ID, "screener-content")))
            self._authenticated = True
            print("✅ Finviz Elite authenticated successfully.")
            return True
            
        except TimeoutException as e:
            print(f"❌ Finviz Elite authentication failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Finviz authentication error: {e}")
            return False
    
    def __del__(self):
        """Cleanup Selenium driver."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
    
    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """Fetch news from Finviz (with optional Elite authentication)."""
        news_items = []
        
        # Authenticate if Elite is enabled
        if self.use_elite and not self._authenticated:
            self._authenticate_elite()
        
        url = self.base_url.format(ticker)
        
        html = await self._fetch_html(url)
        if not html:
            return news_items
        
        soup = self._parse_html(html)
        news_table = soup.find('table', class_='fullview-news-outer')
        if not news_table:
            return news_items
        
        rows = news_table.find_all('tr')
        
        for row in rows[:10]:
            try:
                cells = row.find_all('td')
                if len(cells) < 2:
                    continue
                
                time_cell = self._extract_text(cells[0])
                timestamp = self._parse_timestamp(time_cell)
                
                link = cells[1].find('a')
                if not link:
                    continue
                
                title = self._extract_text(link)
                article_url = link.get('href', '')
                
                category = self._categorize_news(title)
                
                news_item = NewsItem(
                    title=title,
                    summary=title,
                    url=article_url,
                    timestamp=timestamp,
                    source=self.source_name,
                    ticker=ticker,
                    category=category
                )
                news_items.append(news_item)
            except Exception as e:
                print(f"Error parsing Finviz article: {e}")
                continue
        
        return news_items
    
    def _parse_timestamp(self, time_str: str) -> datetime:
        """Parse Finviz timestamp format."""
        try:
            now = datetime.now()
            
            if 'Today' in time_str or 'AM' in time_str or 'PM' in time_str:
                # Today's news - use current date
                return now
            else:
                # Parse date like "Nov-28-23"
                parts = time_str.split()
                if parts:
                    # Try to parse, fallback to now
                    return now - timedelta(days=1)
        except:
            pass
        
        return datetime.now()
