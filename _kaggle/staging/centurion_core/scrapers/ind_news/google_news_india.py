# Centurion Capital LLC - Google News India Scraper
"""
Fetches Indian stock news via Google News RSS.

Google News RSS is the most reliable source for Indian financial
news because the underlying sites (Moneycontrol, ET, Mint, etc.)
heavily use client-side rendering that defeats simple HTML scraping.

The feed returns articles from **all** major Indian financial outlets,
so this acts as a catch-all aggregator source.
"""

import logging
from datetime import datetime
from typing import List
from urllib.parse import quote_plus

from models import NewsItem
from scrapers import BaseNewsScraper

logger = logging.getLogger(__name__)

# Well-known company names for popular Indian tickers
_TICKER_COMPANY_MAP = {
    "RELIANCE": "Reliance Industries",
    "TCS": "TCS Tata Consultancy",
    "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys",
    "ICICIBANK": "ICICI Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "SBIN": "State Bank of India SBI",
    "BHARTIARTL": "Bharti Airtel",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen Toubro L&T",
    "ITC": "ITC Limited",
    "AXISBANK": "Axis Bank",
    "WIPRO": "Wipro",
    "HCLTECH": "HCL Technologies",
    "MARUTI": "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "BAJFINANCE": "Bajaj Finance",
    "SUNPHARMA": "Sun Pharma",
    "NESTLEIND": "Nestle India",
    "ULTRACEMCO": "UltraTech Cement",
    "TITAN": "Titan Company",
    "TECHM": "Tech Mahindra",
    "ONGC": "ONGC",
    "NTPC": "NTPC",
    "POWERGRID": "Power Grid",
    "JSWSTEEL": "JSW Steel",
    "COALINDIA": "Coal India",
    "BAJAJFINSV": "Bajaj Finserv",
    "DRREDDY": "Dr Reddy",
    "DIVISLAB": "Divi's Laboratories",
    "ASIANPAINT": "Asian Paints",
    "CIPLA": "Cipla",
    "GRASIM": "Grasim Industries",
    "HEROMOTOCO": "Hero MotoCorp",
    "EICHERMOT": "Eicher Motors",
    "BPCL": "BPCL Bharat Petroleum",
    "INDUSINDBK": "IndusInd Bank",
    "HINDALCO": "Hindalco",
    "APOLLOHOSP": "Apollo Hospitals",
    "SBILIFE": "SBI Life Insurance",
    "HDFCLIFE": "HDFC Life Insurance",
    "BRITANNIA": "Britannia Industries",
    "TATACONSUM": "Tata Consumer Products",
    "M&M": "Mahindra Mahindra M&M",
    "BAJAJ-AUTO": "Bajaj Auto",
    "VEDL": "Vedanta",
    "ZOMATO": "Zomato",
    "PAYTM": "Paytm One97",
    "NYKAA": "Nykaa FSN E-Commerce",
    "IRCTC": "IRCTC",
    "HAL": "HAL Hindustan Aeronautics",
    "BEL": "Bharat Electronics BEL",
}


class GoogleNewsIndiaScraper(BaseNewsScraper):
    """
    Scraper using Google News RSS for Indian stock news.

    Returns articles from multiple Indian financial outlets (Moneycontrol,
    Economic Times, Mint, Business Standard, NDTV Profit, etc.) via a
    single reliable RSS endpoint.
    """

    def __init__(self):
        super().__init__(
            "Google News India",
            "https://news.google.com",
        )

    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        news_items: List[NewsItem] = []
        bare = ticker.replace(".NS", "").replace(".BO", "")

        # Build a search query that yields relevant financial articles
        company = _TICKER_COMPANY_MAP.get(bare, bare)
        query = f"{company} stock"

        rss_url = (
            f"{self.base_url}/rss/search?"
            f"q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        )

        logger.info("Google News India: Fetching news for %s (%s)…", ticker, query)
        xml = await self._fetch_html(rss_url)

        if not xml:
            logger.warning("Google News India: Could not fetch RSS for %s", ticker)
            return news_items

        try:
            soup = self._parse_html(xml, parser="lxml-xml")
        except Exception:
            soup = self._parse_html(xml, parser="html.parser")

        for item in soup.find_all("item"):
            try:
                title_el = item.find("title")
                if not title_el:
                    continue
                title = self._extract_text(title_el)
                if not title or len(title) < 10:
                    continue

                # Get the actual article link (not Google redirect)
                link_el = item.find("link")
                url = ""
                if link_el:
                    # In RSS, <link> text is the next sibling text node
                    url = link_el.next_sibling
                    if url:
                        url = str(url).strip()
                    if not url:
                        url = self._extract_text(link_el)

                # Description / summary
                desc_el = item.find("description")
                summary = ""
                if desc_el:
                    desc_html = self._extract_text(desc_el)
                    # Google wraps description in HTML; strip tags
                    from bs4 import BeautifulSoup as _BS
                    summary = _BS(desc_html, "html.parser").get_text(strip=True)
                if not summary:
                    summary = title

                # Source (Google News provides <source> element)
                source_el = item.find("source")
                source_name = self._extract_text(source_el) if source_el else self.source_name

                # Timestamp
                pub_el = item.find("pubDate")
                timestamp = self._parse_rss_date(pub_el) if pub_el else datetime.now()

                news_items.append(NewsItem(
                    title=title,
                    summary=summary[:500],
                    url=url or "",
                    timestamp=timestamp,
                    source=source_name,
                    ticker=ticker,
                    category=self._categorize_news(title),
                ))
            except Exception:
                continue

        if news_items:
            logger.info(
                "Google News India: Fetched %d articles for %s",
                len(news_items), ticker,
            )
        else:
            logger.warning("Google News India: No articles found for %s", ticker)

        return news_items[:15]

    @staticmethod
    def _parse_rss_date(el) -> datetime:
        try:
            text = el.get_text(strip=True)
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(text)
        except Exception:
            return datetime.now()
