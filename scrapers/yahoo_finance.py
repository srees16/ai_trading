"""
Yahoo Finance news scraper.
"""

from datetime import datetime
from typing import List
from scrapers import BaseNewsScraper
from models import NewsItem


class YahooFinanceScraper(BaseNewsScraper):
    """Scraper for Yahoo Finance news."""
    
    def __init__(self):
        # Updated URL format - Yahoo Finance changed their structure
        super().__init__("Yahoo Finance", "https://finance.yahoo.com/quote/{}")
    
    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """Fetch news from Yahoo Finance."""
        news_items = []
        url = self.base_url.format(ticker)
        
        # Enhanced headers to better mimic browser behavior
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        }
        
        html = await self._fetch_html(url, headers)
        if not html:
            return news_items
        
        soup = self._parse_html(html)
        
        # Try multiple selectors as Yahoo Finance structure varies
        # Look for news sections with different possible class names
        articles = (
            soup.find_all('li', class_='js-stream-content') or
            soup.find_all('div', class_='Ov(h)') or
            soup.find_all('section', {'data-test': 'news-stream'}) or
            soup.find_all('div', class_='news-stream')
        )
        
        # If no articles found with specific classes, try finding all article/h3 combinations
        if not articles:
            # Look for any h3 tags that might contain news titles
            articles = soup.find_all('h3')[:10]
        
        for article in articles[:10]:
            try:
                # Try different title extraction methods
                if article.name == 'h3':
                    title_elem = article
                else:
                    title_elem = article.find('h3') or article.find('a') or article.find('div', class_='title')
                
                if not title_elem:
                    continue
                
                title = self._extract_text(title_elem)
                if not title or len(title) < 10:  # Skip very short titles
                    continue
                
                # Extract link
                link_elem = title_elem.find('a') if article.name != 'a' else title_elem
                article_url = ""
                if link_elem and 'href' in link_elem.attrs:
                    href = link_elem['href']
                    if href.startswith('http'):
                        article_url = href
                    elif href.startswith('/'):
                        article_url = f"https://finance.yahoo.com{href}"
                
                # Extract summary
                summary_elem = article.find('p') or article.find('div', class_='summary')
                summary = self._extract_text(summary_elem) if summary_elem else title
                
                timestamp = datetime.now()
                category = self._categorize_news(title + " " + summary)
                
                news_item = NewsItem(
                    title=title,
                    summary=summary,
                    url=article_url,
                    timestamp=timestamp,
                    source=self.source_name,
                    ticker=ticker,
                    category=category
                )
                news_items.append(news_item)
            except Exception as e:
                print(f"Error parsing Yahoo article: {e}")
                continue
        
        # If still no news found, create a fallback entry
        if not news_items:
            print(f"Yahoo Finance: No news articles found for {ticker} (website structure may have changed)")
            # Create a minimal news item to indicate attempt was made
            news_items.append(NewsItem(
                title=f"No recent news found for {ticker}",
                summary="Yahoo Finance scraping completed but no articles were parsed",
                url=url,
                timestamp=datetime.now(),
                source=self.source_name,
                ticker=ticker,
                category=self._categorize_news("")
            ))
        
        return news_items
