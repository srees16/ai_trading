# Centurion Capital LLC — Morningstar Scraper
"""
Scrapes stock news and analysis from Morningstar.com.

Uses multiple approaches:
1. Morningstar's public search/quote pages for stock-specific news
2. RSS feed fallback for broader financial news
3. Market / economic analysis for macro context

Works for both US and Indian tickers.
"""

import logging
import re
from datetime import datetime
from typing import List
from urllib.parse import quote_plus

from models import NewsItem
from scrapers import BaseNewsScraper

logger = logging.getLogger(__name__)

# CSS selectors for Morningstar article pages — multiple fallback sets
_SEARCH_SELECTORS = [
    # Search results page
    {
        "container": "div.mdc-search-results__item",
        "title": "a.mdc-search-results__heading",
        "summary": "p.mdc-search-results__description",
        "link": "a.mdc-search-results__heading",
        "time": "span.mdc-search-results__date",
    },
    # Article listing
    {
        "container": "article",
        "title": "h3 a, h2 a, a.mdc-link",
        "summary": "p",
        "link": "h3 a, h2 a, a.mdc-link",
        "time": "time, span[data-timestamp]",
    },
    # Generic fallback for richer listings
    {
        "container": "div.mdc-card, div[class*='article'], div[class*='story']",
        "title": "a[class*='title'], h3 a, h2 a",
        "summary": "p[class*='summary'], p[class*='description'], p",
        "link": "a[class*='title'], h3 a, h2 a",
        "time": "time, span[class*='date']",
    },
]

_QUOTE_SELECTORS = [
    # Quote page — Recent news section
    {
        "container": "div[class*='news'] article, section[class*='news'] li, div[class*='recent'] a",
        "title": "a, h3, h4",
        "summary": "p, span[class*='desc']",
        "link": "a",
        "time": "time, span[class*='date']",
    },
]


class MorningstarScraper(BaseNewsScraper):
    """Scraper for Morningstar.com stock news and analysis."""

    def __init__(self):
        super().__init__(
            "Morningstar",
            "https://www.morningstar.com",
        )

    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """
        Fetch news from Morningstar for the given ticker.

        Tries, in order:
        1. Morningstar search page
        2. Morningstar quote/news page
        3. RSS feed
        """
        news_items: List[NewsItem] = []
        company = ticker.replace(".NS", "").replace(".BO", "")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.morningstar.com/",
        }

        logger.info("Morningstar: Fetching news for %s…", ticker)

        # 1) Search page
        news_items = await self._try_search(company, ticker, headers)
        if news_items:
            logger.info(
                "Morningstar: Fetched %d articles for %s (search)",
                len(news_items), ticker,
            )
            return news_items[:10]

        # 2) Quote / stock page
        news_items = await self._try_quote_page(company, ticker, headers)
        if news_items:
            logger.info(
                "Morningstar: Fetched %d articles for %s (quote page)",
                len(news_items), ticker,
            )
            return news_items[:10]

        # 3) RSS fallback
        news_items = await self._try_rss(company, ticker)
        if news_items:
            logger.info(
                "Morningstar: Fetched %d articles for %s (RSS)",
                len(news_items), ticker,
            )
            return news_items[:10]

        logger.warning("Morningstar: No articles found for %s", ticker)
        return []

    # ── Approach 1: Search ───────────────────────────────────────────

    async def _try_search(
        self, company: str, ticker: str, headers: dict
    ) -> List[NewsItem]:
        search_url = (
            f"{self.base_url}/search?query={quote_plus(company)}"
            f"&type=articles"
        )
        html = await self._fetch_html(search_url, headers)
        if not html:
            return []
        return self._extract_articles(html, ticker, _SEARCH_SELECTORS)

    # ── Approach 2: Quote page ───────────────────────────────────────

    async def _try_quote_page(
        self, company: str, ticker: str, headers: dict
    ) -> List[NewsItem]:
        # Morningstar uses /stocks/{exchange}/{ticker}/news for US
        ms_ticker = company.upper()

        # Try US exchange symbols
        for exchange in ("xnas", "xnys", "xbom", "xnse"):
            quote_url = (
                f"{self.base_url}/stocks/{exchange}/{ms_ticker.lower()}/news"
            )
            html = await self._fetch_html(quote_url, headers)
            if html and len(html) > 5000:
                items = self._extract_articles(html, ticker, _QUOTE_SELECTORS)
                if items:
                    return items
        return []

    # ── Approach 3: RSS ──────────────────────────────────────────────

    async def _try_rss(self, company: str, ticker: str) -> List[NewsItem]:
        rss_urls = [
            f"https://www.morningstar.com/feeds/rss/search?q={quote_plus(company)}",
            "https://www.morningstar.com/feeds/rss/market-fair-value",
        ]
        items: List[NewsItem] = []
        for rss_url in rss_urls:
            html = await self._fetch_html(rss_url)
            if not html:
                continue
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "lxml-xml")
                for entry in soup.find_all("item")[:10]:
                    title = entry.find("title")
                    desc = entry.find("description")
                    link = entry.find("link")
                    pub = entry.find("pubDate")
                    if not title:
                        continue
                    title_text = title.get_text(strip=True)
                    if len(title_text) < 10:
                        continue
                    items.append(NewsItem(
                        title=title_text,
                        summary=(desc.get_text(strip=True)[:500] if desc else title_text),
                        url=(link.get_text(strip=True) if link else self.base_url),
                        timestamp=self._parse_rss_date(pub),
                        source=self.source_name,
                        ticker=ticker,
                        category=self._categorize_news(title_text),
                    ))
            except Exception:
                continue
        return items

    # ── Shared extraction ────────────────────────────────────────────

    def _extract_articles(
        self, html: str, ticker: str, selector_sets: list
    ) -> List[NewsItem]:
        soup = self._parse_html(html)
        items: List[NewsItem] = []

        for sel in selector_sets:
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
                    url = ""
                    if link_el:
                        url = link_el.get("href", "")
                        if url and not url.startswith("http"):
                            url = self.base_url + url

                    summary_el = article.select_one(sel["summary"])
                    summary = self._extract_text(summary_el) if summary_el else title

                    time_el = article.select_one(sel["time"])
                    timestamp = self._parse_article_date(time_el)

                    items.append(NewsItem(
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

            if items:
                break

        return items

    # ── Date helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_article_date(el) -> datetime:
        if not el:
            return datetime.now()
        raw = el.get("datetime") or el.get("data-timestamp") or el.get_text(strip=True)
        return MorningstarScraper._parse_date_str(raw)

    @staticmethod
    def _parse_rss_date(el) -> datetime:
        if not el:
            return datetime.now()
        return MorningstarScraper._parse_date_str(el.get_text(strip=True))

    @staticmethod
    def _parse_date_str(raw: str) -> datetime:
        if not raw:
            return datetime.now()
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%b %d, %Y",
            "%B %d, %Y",
            "%m/%d/%Y",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
        ):
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue

        # Try partial date match
        m = re.search(r"(\w{3,9})\s+(\d{1,2}),?\s+(\d{4})", raw)
        if m:
            try:
                return datetime.strptime(f"{m.group(1)} {m.group(2)}, {m.group(3)}", "%B %d, %Y")
            except ValueError:
                try:
                    return datetime.strptime(f"{m.group(1)} {m.group(2)}, {m.group(3)}", "%b %d, %Y")
                except ValueError:
                    pass
        return datetime.now()
