"""
Finviz news scraper.
"""

from datetime import datetime, timedelta
from typing import List
from scrapers import BaseNewsScraper
from models import NewsItem


class FinvizScraper(BaseNewsScraper):
    """Scraper for Finviz news."""
    
    def __init__(self):
        super().__init__("Finviz", "https://finviz.com/quote.ashx?t={}")
    
    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """Fetch news from Finviz."""
        news_items = []
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
