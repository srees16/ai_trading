# Centurion Capital LLC - The Hindu Business Line Scraper
"""Scrapes stock news from The Hindu Business Line."""

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
        "container": "div.story-card, div.story-box",
        "title": "a.story-card-heading, h2 a, h3 a",
        "summary": "p.story-card-text, p",
        "link": "a.story-card-heading, h2 a, h3 a",
        "time": "time, span.bl-ts",
    },
    # Fallback: listing page
    {
        "container": "div.main-story, div.other-story",
        "title": "a",
        "summary": "p",
        "link": "a",
        "time": "span.date, time",
    },
    # Fallback 2: generic articles
    {
        "container": "article",
        "title": "h2 a, h3 a, a[title]",
        "summary": "p",
        "link": "h2 a, h3 a, a[title]",
        "time": "time, span.date",
    },
]


class HinduBusinessLineScraper(BaseNewsScraper):
    """Scraper for thehindubusinessline.com stock news."""

    def __init__(self):
        super().__init__(
            "Hindu Business Line",
            "https://www.thehindubusinessline.com",
        )

    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        news_items: List[NewsItem] = []
        company = ticker.replace(".NS", "").replace(".BO", "")

        search_url = (
            f"{self.base_url}/search?q={quote_plus(company)}&type=article"
        )

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.thehindubusinessline.com/",
        }

        logger.info("Hindu Business Line: Fetching news for %s…", ticker)
        html = await self._fetch_html(search_url, headers)

        if not html:
            # Fallback: RSS feed
            rss_items = await self._try_rss(company, ticker)
            if rss_items:
                return rss_items[:10]
            logger.warning("Hindu Business Line: Could not fetch data for %s", ticker)
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
            logger.info("Hindu Business Line: Fetched %d articles for %s", len(news_items), ticker)
        else:
            # HTML scraping found nothing (JS-rendered?); try RSS fallback
            rss_items = await self._try_rss(company, ticker)
            if rss_items:
                return rss_items[:10]
            logger.warning("Hindu Business Line: No articles found for %s", ticker)

        return news_items

    async def _try_rss(self, company: str, ticker: str) -> List[NewsItem]:
        """Fallback: parse RSS feeds for market news."""
        rss_url = f"{self.base_url}/markets/stock-markets/feeder/default.htm"
        html = await self._fetch_html(rss_url)
        if not html:
            return []

        items: List[NewsItem] = []
        try:
            soup = self._parse_html(html, parser="lxml-xml")
            for item in soup.find_all("item")[:15]:
                title = self._extract_text(item.find("title"))
                if not title:
                    continue
                # Basic relevance check
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
            # RFC 822: "Tue, 04 Mar 2026 10:30:00 +0530"
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(text)
        except Exception:
            return datetime.now()

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
