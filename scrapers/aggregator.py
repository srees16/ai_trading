"""
Aggregator for all news scrapers.
"""

import logging
import asyncio
from typing import List
from scrapers import BaseNewsScraper
from scrapers.yahoo_finance import YahooFinanceScraper
from scrapers.finviz import FinvizScraper
from scrapers.investing import InvestingScraper
from scrapers.tradingview import TradingViewScraper
from models import NewsItem

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregates news from multiple sources."""
    
    def __init__(self):
        self.scrapers: List[BaseNewsScraper] = [
            YahooFinanceScraper(),
            FinvizScraper(),
            InvestingScraper(),
            TradingViewScraper(),
        ]
    
    async def fetch_all_news(self, ticker: str) -> List[NewsItem]:
        """
        Fetch news from all sources concurrently.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Combined list of news items from all sources
        """
        logger.info(f"Fetching news for {ticker} from {len(self.scrapers)} sources...")
        tasks = [scraper.fetch_news(ticker) for scraper in self.scrapers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_news = []
        for i, result in enumerate(results):
            scraper_name = self.scrapers[i].source_name
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                # Log which scraper failed for which ticker
                logger.warning(f"{scraper_name}: Failed to fetch news for {ticker} - {type(result).__name__}")
        
        logger.info(f"Total: {len(all_news)} articles collected for {ticker}")
        return all_news
    
    async def fetch_news_for_tickers(self, tickers: List[str]) -> List[NewsItem]:
        """
        Fetch news for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Combined list of news items for all tickers
        """
        logger.info(f"Starting news fetch for {len(tickers)} ticker(s): {', '.join(tickers)}")
        tasks = [self.fetch_all_news(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_news = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Failed to fetch news for {tickers[i]}: {type(result).__name__}")
        
        logger.info(f"News fetch complete: {len(all_news)} total articles for {len(tickers)} ticker(s)")
        return all_news
