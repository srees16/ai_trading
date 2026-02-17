"""
Yahoo Finance news scraper using yfinance library.
"""

import logging
from datetime import datetime
from typing import List
import yfinance as yf
from scrapers import BaseNewsScraper
from models import NewsItem

logger = logging.getLogger(__name__)


class YahooFinanceScraper(BaseNewsScraper):
    """Scraper for Yahoo Finance news using yfinance library."""
    
    def __init__(self):
        super().__init__("Yahoo Finance", "https://finance.yahoo.com/quote/{}")
    
    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """Fetch news from Yahoo Finance using yfinance library."""
        news_items = []
        
        logger.info(f"Yahoo Finance: Fetching news for {ticker}...")
        try:
            # Use yfinance to get news - this uses Yahoo's official API
            yf_ticker = yf.Ticker(ticker)
            news_data = yf_ticker.news
            
            if not news_data:
                logger.info(f"Yahoo Finance: No news found for {ticker}")
                return news_items
            
            for article in news_data[:10]:
                try:
                    # Extract from yfinance news structure
                    content = article.get('content', article)
                    
                    # Get title
                    title = content.get('title', '') if isinstance(content, dict) else article.get('title', '')
                    if not title:
                        continue
                    
                    # Get summary/description
                    summary = ''
                    if isinstance(content, dict):
                        summary = content.get('summary', '') or content.get('description', '')
                    if not summary:
                        summary = article.get('summary', '') or article.get('description', title)
                    
                    # Get URL
                    article_url = ''
                    if isinstance(content, dict):
                        canonical = content.get('canonicalUrl', {})
                        if isinstance(canonical, dict):
                            article_url = canonical.get('url', '')
                        click_through = content.get('clickThroughUrl', {})
                        if not article_url and isinstance(click_through, dict):
                            article_url = click_through.get('url', '')
                    if not article_url:
                        article_url = article.get('link', '') or article.get('url', '')
                    
                    # Get timestamp
                    timestamp = datetime.now()
                    pub_date = None
                    if isinstance(content, dict):
                        pub_date = content.get('pubDate')
                    if not pub_date:
                        pub_date = article.get('providerPublishTime')
                    
                    if pub_date:
                        if isinstance(pub_date, str):
                            try:
                                timestamp = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            except:
                                pass
                        elif isinstance(pub_date, (int, float)):
                            try:
                                timestamp = datetime.fromtimestamp(pub_date)
                            except:
                                pass
                    
                    # Categorize
                    category = self._categorize_news(title + " " + summary)
                    
                    news_item = NewsItem(
                        title=title,
                        summary=summary[:500] if summary else title,  # Limit summary length
                        url=article_url,
                        timestamp=timestamp,
                        source=self.source_name,
                        ticker=ticker,
                        category=category
                    )
                    news_items.append(news_item)
                    
                except Exception as e:
                    # Skip articles that fail to parse
                    continue
            
            if news_items:
                logger.info(f"Yahoo Finance: Fetched {len(news_items)} articles for {ticker}")
            else:
                logger.info(f"Yahoo Finance: No articles parsed for {ticker}")
            
        except Exception as e:
            logger.warning(f"Yahoo Finance: Error fetching news for {ticker}: {type(e).__name__}")
        
        return news_items
