# Centurion Capital LLC - Business Standard Scraper
"""Scrapes stock news from Business Standard (Markets / Companies)."""

import logging
from datetime import datetime
from typing import List
from urllib.parse import quote_plus

from models import NewsItem
from scrapers import BaseNewsScraper

logger = logging.getLogger(__name__)

_SELECTORS = [
    # Primary: search results
    {
        "container": "div.listing-txt, div.main-story",
        "title": "h2 a, h3 a",
        "summary": "p",
        "link": "h2 a, h3 a",
        "time": "span.date, time",
    },
    # Fallback: article listing
    {
        "container": "li.card-list-item",
        "title": "a.story-title",
        "summary": "p",
        "link": "a.story-title",
        "time": "span.date-txt",
    },
    # Fallback 2: generic
    {
        "container": "article, div.story-card",
        "title": "a[title], h2 a, h3 a",
        "summary": "p",
        "link": "a[title], h2 a, h3 a",
        "time": "time, span.date",
    },
]


class BusinessStandardScraper(BaseNewsScraper):
    """Scraper for business-standard.com stock news."""

    def __init__(self):
        super().__init__(
            "Business Standard",
            "https://www.business-standard.com",
        )

    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        news_items: List[NewsItem] = []
        company = ticker.replace(".NS", "").replace(".BO", "")

        search_url = (
            f"{self.base_url}/search?q={quote_plus(company)}&type=news"
        )

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.business-standard.com/",
        }

        logger.info("Business Standard: Fetching news for %s…", ticker)
        html = await self._fetch_html(search_url, headers)

        if not html:
            # Fallback: company topic page
            alt_url = (
                f"{self.base_url}/topic/{quote_plus(company.lower().replace(' ', '-'))}"
            )
            html = await self._fetch_html(alt_url, headers)

        if not html:
            # Both URLs blocked or failed; try RSS fallback
            rss_items = await self._try_rss(company, ticker)
            if rss_items:
                return rss_items[:10]
            logger.warning("Business Standard: Could not fetch data for %s", ticker)
            return news_items

        soup = self._parse_html(html)

        for sel in _SELECTORS:
            articles = soup.select(sel["container"])
            if not articles:
                continue

            for article in articles[:10]:
                try:
                    title_el = article.select_one(sel["title"])
                    if not title_el:
                        continue

                    title = self._extract_text(title_el)
                    if not title or len(title) < 10:
                        continue

                    link_el = article.select_one(sel["link"])
                    url = link_el.get("href", "") if link_el else ""
                    if url and not url.startswith("http"):
                        url = self.base_url + url

                    summary_el = article.select_one(sel["summary"])
                    summary = self._extract_text(summary_el) if summary_el else title

                    time_el = article.select_one(sel["time"])
                    timestamp = self._parse_datetime(time_el)

                    news_items.append(NewsItem(
                        title=title,
                        summary=summary[:500],
                        url=url,
                        timestamp=timestamp,
                        source=self.source_name,
                        ticker=ticker,
                        category=self._categorize_news(title),
                    ))
                except Exception:
                    continue

            if news_items:
                break

        if news_items:
            logger.info("Business Standard: Fetched %d articles for %s", len(news_items), ticker)
        else:
            # HTML scraping found nothing; try RSS
            rss_items = await self._try_rss(company, ticker)
            if rss_items:
                return rss_items[:10]
            logger.warning("Business Standard: No articles found for %s", ticker)

        return news_items

    @staticmethod
    def _parse_datetime(el) -> datetime:
        if not el:
            return datetime.now()
        try:
            dt_str = el.get("datetime") or el.get_text(strip=True)
            if dt_str:
                for fmt in (
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%B %d, %Y %H:%M",
                    "%d %b %Y",
                ):
                    try:
                        return datetime.strptime(dt_str.strip()[:25], fmt)
                    except ValueError:
                        continue
            return datetime.now()
        except Exception:
            return datetime.now()

    async def _try_rss(self, company: str, ticker: str) -> List[NewsItem]:
        """Fallback: parse Business Standard RSS feed for stock news."""
        rss_url = "https://www.business-standard.com/rss/markets-106.rss"
        html = await self._fetch_html(rss_url)
        if not html:
            return []

        items: List[NewsItem] = []
        try:
            soup = self._parse_html(html, parser="lxml-xml")
            for item in soup.find_all("item")[:20]:
                title = self._extract_text(item.find("title"))
                if not title:
                    continue
                if company.lower() not in title.lower():
                    continue
                link = self._extract_text(item.find("link"))
                desc = self._extract_text(item.find("description"))
                pub = item.find("pubDate")
                ts = self._parse_rss_date(pub) if pub else datetime.now()
                items.append(NewsItem(
                    title=title,
                    summary=desc[:500] if desc else title,
                    url=link or "",
                    timestamp=ts,
                    source=self.source_name,
                    ticker=ticker,
                    category=self._categorize_news(title),
                ))
        except Exception:
            pass
        return items

    @staticmethod
    def _parse_rss_date(el) -> datetime:
        try:
            text = el.get_text(strip=True)
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(text)
        except Exception:
            return datetime.now()
