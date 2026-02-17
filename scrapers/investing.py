"""
Investing.com news scraper.
"""

import logging
from datetime import datetime
from typing import List
from scrapers import BaseNewsScraper
from models import NewsItem

logger = logging.getLogger(__name__)


class InvestingScraper(BaseNewsScraper):
    """Scraper for Investing.com news."""
    
    def __init__(self):
        super().__init__("Investing.com", "https://www.investing.com/search/?q={}")
    
    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """Fetch news from Investing.com."""
        news_items = []
        url = self.base_url.format(ticker)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        
        logger.info(f"Investing.com: Fetching news for {ticker}...")
        html = await self._fetch_html(url, headers)
        if not html:
            logger.warning(f"Investing.com: Could not fetch data for {ticker}")
            return news_items
        
        soup = self._parse_html(html)
        articles = soup.find_all('article', class_='js-article-item')
        
        for article in articles[:10]:
            try:
                title_elem = article.find('a', class_='title')
                if not title_elem:
                    continue
                
                title = self._extract_text(title_elem)
                href = title_elem.get('href', '')
                article_url = f"https://www.investing.com{href}" if href else ""
                
                timestamp = datetime.now()
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
            except Exception:
                # Skip articles that fail to parse
                continue
        
        if news_items:
            logger.info(f"Investing.com: Fetched {len(news_items)} articles for {ticker}")
        else:
            logger.info(f"Investing.com: No articles found for {ticker}")
        
        return news_items
