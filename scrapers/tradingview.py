"""
TradingView news scraper.
"""

import logging
import aiohttp
from datetime import datetime
from typing import List
from scrapers import BaseNewsScraper
from models import NewsItem

logger = logging.getLogger(__name__)


class TradingViewScraper(BaseNewsScraper):
    """Scraper for TradingView news using their API."""
    
    def __init__(self):
        super().__init__("TradingView", "https://news-headlines.tradingview.com/v2/headlines")
    
    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """Fetch news from TradingView headlines API."""
        news_items = []
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Origin": "https://www.tradingview.com",
        }
        
        logger.info(f"TradingView: Fetching news for {ticker}...")
        
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try different exchange prefixes
                for exchange in ["NASDAQ", "NYSE", "AMEX"]:
                    url = f"{self.base_url}?category=base&client=web&lang=en&limit=10&streaming=true&symbol={exchange}:{ticker}"
                    
                    async with session.get(url, headers=headers, ssl=False) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # TradingView returns {"items": [...]}
                            items = data.get('items', []) if isinstance(data, dict) else data
                            
                            if items and len(items) > 0:
                                for article in items[:10]:
                                    try:
                                        title = article.get('title', '')
                                        if not title:
                                            continue
                                        
                                        # Get article details
                                        summary = article.get('shortDescription', '') or article.get('description', '') or title
                                        article_url = article.get('link', '') or article.get('storyPath', '')
                                        provider = article.get('provider', 'TradingView')
                                        
                                        # Parse timestamp
                                        timestamp = datetime.now()
                                        pub_time = article.get('published')
                                        if pub_time:
                                            try:
                                                timestamp = datetime.fromtimestamp(pub_time)
                                            except:
                                                pass
                                        
                                        category = self._categorize_news(title + " " + summary)
                                        
                                        news_item = NewsItem(
                                            title=title,
                                            summary=summary[:500] if summary else title,
                                            url=article_url,
                                            timestamp=timestamp,
                                            source=f"{self.source_name} ({provider})",
                                            ticker=ticker,
                                            category=category
                                        )
                                        news_items.append(news_item)
                                    except Exception:
                                        continue
                                
                                if news_items:
                                    break  # Found articles, stop trying other exchanges
                
                if news_items:
                    logger.info(f"TradingView: Fetched {len(news_items)} articles for {ticker}")
                else:
                    logger.info(f"TradingView: No articles found for {ticker}")
                    
        except Exception as e:
            logger.warning(f"TradingView: Error fetching news for {ticker}: {type(e).__name__}")
        
        return news_items
