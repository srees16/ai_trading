"""
Aggregator for all news scrapers.
"""

import asyncio
from typing import List
from scrapers import BaseNewsScraper
from scrapers.yahoo_finance import YahooFinanceScraper
from scrapers.finviz import FinvizScraper
from scrapers.investing import InvestingScraper
from scrapers.tradingview import TradingViewScraper
from models import NewsItem


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
        tasks = [scraper.fetch_news(ticker) for scraper in self.scrapers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_news = []
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                print(f"Scraper error: {result}")
        
        return all_news
    
    async def fetch_news_for_tickers(self, tickers: List[str]) -> List[NewsItem]:
        """
        Fetch news for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Combined list of news items for all tickers
        """
        tasks = [self.fetch_all_news(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_news = []
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                print(f"Ticker fetch error: {result}")
        
        return all_news
