# Centurion Capital LLC - Zerodha Pulse Scraper
"""Scrapes stock news from Pulse by Zerodha (pulse.zerodha.com)."""

import logging
from datetime import datetime
from typing import List
from urllib.parse import quote_plus

from models import NewsItem
from scrapers import BaseNewsScraper

logger = logging.getLogger(__name__)

_SELECTORS = [
    # Primary: pulse feed items
    {
        "container": "li.box.item",
        "title": "h2 a, a.title",
        "summary": "div.desc, div.description",
        "link": "h2 a, a.title",
        "time": "span.date",
        "source_tag": "span.feed-source",
    },
    # Fallback: alternate layout
    {
        "container": "div.item, article",
        "title": "h2 a, h3 a, a.title",
        "summary": "p, div.description",
        "link": "h2 a, h3 a, a.title",
        "time": "span.date, time",
        "source_tag": "span.source, span.feed-source",
    },
]


class ZerodhaPulseScraper(BaseNewsScraper):
    """Scraper for pulse.zerodha.com — aggregated financial news feed."""

    def __init__(self):
        super().__init__("Zerodha Pulse", "https://pulse.zerodha.com")

    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        news_items: List[NewsItem] = []
        company = ticker.replace(".NS", "").replace(".BO", "")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://pulse.zerodha.com/",
        }

        logger.info("Zerodha Pulse: Fetching news for %s…", ticker)
        # Pulse has no real search endpoint; fetch main feed and filter
        html = await self._fetch_html(self.base_url, headers)

        if not html:
            logger.warning("Zerodha Pulse: Could not fetch data for %s", ticker)
            return news_items

        soup = self._parse_html(html)

        for sel in _SELECTORS:
            articles = soup.select(sel["container"])
            if not articles:
                continue

            for article in articles[:15]:
                try:
                    title_el = article.select_one(sel["title"])
                    if not title_el:
                        continue

                    title = self._extract_text(title_el)
                    if not title or len(title) < 10:
                        continue

                    # When using the main feed, filter by relevance
                    if company.lower() not in title.lower():
                        # Also check summary
                        summary_el = article.select_one(sel["summary"])
                        summary_text = self._extract_text(summary_el) if summary_el else ""
                        if company.lower() not in summary_text.lower():
                            continue

                    link_el = article.select_one(sel["link"])
                    url = link_el.get("href", "") if link_el else ""
                    if url and not url.startswith("http"):
                        url = self.base_url + url

                    summary_el = article.select_one(sel["summary"])
                    summary = self._extract_text(summary_el) if summary_el else title

                    time_el = article.select_one(sel["time"])
                    timestamp = self._parse_datetime(time_el)

                    # Zerodha Pulse aggregates from multiple sources
                    source_name = self.source_name
                    source_tag = sel.get("source_tag")
                    if source_tag:
                        src_el = article.select_one(source_tag)
                        if src_el:
                            orig = self._extract_text(src_el)
                            if orig:
                                source_name = f"Pulse ({orig})"

                    news_items.append(NewsItem(
                        title=title,
                        summary=summary[:500],
                        url=url,
                        timestamp=timestamp,
                        source=source_name,
                        ticker=ticker,
                        category=self._categorize_news(title),
                    ))
                except Exception:
                    continue

            if news_items:
                break

        if news_items:
            logger.info("Zerodha Pulse: Fetched %d articles for %s", len(news_items), ticker)
        else:
            logger.warning("Zerodha Pulse: No articles found for %s", ticker)

        return news_items

    @staticmethod
    def _parse_datetime(el) -> datetime:
        if not el:
            return datetime.now()
        try:
            text = el.get_text(strip=True).lower()
            # Pulse often shows relative times: "2 hours ago", "1 day ago"
            if "ago" in text:
                return datetime.now()
            dt_str = el.get("datetime") or text
            for fmt in (
                "%Y-%m-%dT%H:%M:%S",
                "%d %b %Y, %I:%M %p",
                "%B %d, %Y",
            ):
                try:
                    return datetime.strptime(dt_str.strip()[:25], fmt)
                except ValueError:
                    continue
            return datetime.now()
        except Exception:
            return datetime.now()
