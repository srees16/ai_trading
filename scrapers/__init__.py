"""
News scraper modules.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
from models import NewsItem, NewsCategory
from config import Config


class BaseNewsScraper(ABC):
    """Abstract base class for news scrapers with common BeautifulSoup utilities."""
    
    def __init__(self, source_name: str, base_url: str = ""):
        self.source_name = source_name
        self.base_url = base_url
    
    @abstractmethod
    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """
        Fetch news for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of NewsItem objects
        """
        pass
    
    async def _fetch_html(self, url: str, headers: dict = None) -> Optional[str]:
        """
        Fetch HTML content from URL.
        
        Args:
            url: URL to fetch
            headers: Optional custom headers
            
        Returns:
            HTML content as string or None if failed
        """
        if headers is None:
            headers = {"User-Agent": Config.USER_AGENT}
        
        try:
            # Configure connector with increased header size limits for sites like Yahoo Finance
            connector = aiohttp.TCPConnector(limit=100)
            timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
            
            # Increase max_line_size and max_field_size to handle large headers
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                trust_env=True
            ) as session:
                async with session.get(
                    url, 
                    headers=headers,
                    max_field_size=65536,  # Increase from default 8190 to 64KB
                    max_line_size=65536    # Increase from default 8190 to 64KB
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        print(f"{self.source_name} returned status {response.status} for URL: {url}")
                        return None
        except Exception as e:
            print(f"Error fetching from {self.source_name}: {e}")
            return None
    
    def _parse_html(self, html: str, parser: str = 'lxml') -> BeautifulSoup:
        """
        Parse HTML content into BeautifulSoup object.
        
        Args:
            html: HTML content string
            parser: Parser to use (default: lxml)
            
        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html, parser)
    
    def _extract_text(self, element, strip: bool = True) -> str:
        """
        Extract text from BeautifulSoup element safely.
        
        Args:
            element: BeautifulSoup element
            strip: Whether to strip whitespace
            
        Returns:
            Extracted text or empty string
        """
        if element:
            return element.get_text(strip=strip)
        return ""
    
    def _categorize_news(self, text: str) -> NewsCategory:
        """
        Categorize news based on keywords.
        
        Args:
            text: Text to analyze
            
        Returns:
            NewsCategory enum
        """
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in Config.BREAKING_KEYWORDS):
            return NewsCategory.BREAKING
        elif any(kw in text_lower for kw in Config.DEALS_KEYWORDS):
            return NewsCategory.DEALS_MA
        elif any(kw in text_lower for kw in Config.MACRO_KEYWORDS):
            return NewsCategory.MACRO_ECONOMIC
        elif any(kw in text_lower for kw in Config.EARNINGS_KEYWORDS):
            return NewsCategory.EARNINGS
        else:
            return NewsCategory.GENERAL
