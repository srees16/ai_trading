"""
News scraper modules.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
from models import NewsItem, NewsCategory
from config import Config

try:
    from fake_useragent import UserAgent
    _ua = UserAgent(fallback=Config.USER_AGENT)
except ImportError:
    _ua = None

# Configure logger for scrapers
logger = logging.getLogger(__name__)

# Retry settings
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


def _random_ua() -> str:
    """Return a random User-Agent string, falling back to Config default."""
    if _ua is not None:
        try:
            return _ua.random
        except Exception:
            pass
    return Config.USER_AGENT


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
        Fetch HTML content from URL with automatic retries and rotating User-Agent.
        
        Retries up to ``_MAX_RETRIES`` times with exponential backoff on
        transient failures (timeouts, 429/5xx responses, connection errors).
        
        Args:
            url: URL to fetch
            headers: Optional custom headers
            
        Returns:
            HTML content as string or None if all attempts failed
        """
        if headers is None:
            headers = {}
        
        for attempt in range(_MAX_RETRIES):
            # Rotate User-Agent on every attempt
            attempt_headers = {**headers, "User-Agent": _random_ua()}
            
            try:
                connector = aiohttp.TCPConnector(limit=100, force_close=True)
                timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
                
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    trust_env=True
                ) as session:
                    async with session.get(
                        url, 
                        headers=attempt_headers,
                        ssl=False
                    ) as response:
                        if response.status == 200:
                            return await response.text()
                        if response.status in (429, 500, 502, 503, 504):
                            # Retryable server errors
                            logger.debug(
                                "%s: HTTP %d on attempt %d/%d for %s",
                                self.source_name, response.status,
                                attempt + 1, _MAX_RETRIES, url,
                            )
                        else:
                            # Non-retryable client error (4xx)
                            return None
            except (aiohttp.ClientConnectorError, aiohttp.ServerTimeoutError):
                logger.debug(
                    "%s: Connection/timeout error on attempt %d/%d for %s",
                    self.source_name, attempt + 1, _MAX_RETRIES, url,
                )
            except aiohttp.ClientResponseError:
                return None
            except Exception as e:
                if str(e):
                    logger.warning(
                        "Unexpected error fetching from %s: %s: %s",
                        self.source_name, type(e).__name__, e,
                    )
                return None
            
            # Exponential backoff before next retry
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
        
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

    @staticmethod
    def compute_relevance_score(
        item: NewsItem,
        ticker: str,
        company: str = "",
    ) -> float:
        """
        Compute a 0-1 keyword relevance score for a news item.

        Factors:
        * Ticker / company-name frequency in title (weighted 2x) and summary.
        * Category bonus: BREAKING / EARNINGS / DEALS_MA score higher.
        * Title-length penalty for very short or very long titles.

        Args:
            item: The news item to score.
            ticker: Stock ticker symbol (e.g. "AAPL" or "RELIANCE.NS").
            company: Optional human-readable company name for matching.

        Returns:
            Relevance score clamped to [0.0, 1.0].
        """
        score = 0.0
        raw_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
        company_lower = company.lower() if company else raw_ticker.lower()

        title_lower = item.title.lower()
        summary_lower = (item.summary or "").lower()

        # Ticker / company mentions (title counts double)
        title_hits = title_lower.count(raw_ticker.lower()) + title_lower.count(company_lower)
        summary_hits = summary_lower.count(raw_ticker.lower()) + summary_lower.count(company_lower)
        mention_score = min((title_hits * 2 + summary_hits) * 0.15, 0.45)
        score += mention_score

        # Category bonus
        cat_bonus = {
            NewsCategory.BREAKING: 0.25,
            NewsCategory.EARNINGS: 0.20,
            NewsCategory.DEALS_MA: 0.20,
            NewsCategory.MACRO_ECONOMIC: 0.10,
            NewsCategory.GENERAL: 0.0,
        }
        score += cat_bonus.get(item.category, 0.0)

        # Baseline for any scraped article
        score += 0.20

        # Sentiment confidence boost (if available)
        if item.sentiment_confidence is not None:
            score += item.sentiment_confidence * 0.10

        return round(min(max(score, 0.0), 1.0), 4)
