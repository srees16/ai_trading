# Centurion Capital LLC - Moneycontrol Scraper
"""
Scrapes stock news from Moneycontrol.com.

Strategy
--------
1.  **Primary** – Resolve the stock's internal ``sc_id`` via the autosuggestion
    JSON API, then fetch the server-rendered ``stock_news.php`` page which
    always returns ~50 articles.
2.  **Fallback** – Multiple RSS feeds (buzzingstocks, latestnews, business).
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote_plus

from models import NewsItem
from scrapers import BaseNewsScraper

logger = logging.getLogger(__name__)

# Pre-mapped sc_ids for the most common Indian tickers to skip the API call.
_TICKER_SCID: Dict[str, str] = {
    "RELIANCE": "RI",
    "TCS": "TCS",
    "HDFCBANK": "HDF01",
    "INFY": "IT",
    "ICICIBANK": "ICI02",
    "HINDUNILVR": "HUL",
    "SBIN": "SBI",
    "BHARTIARTL": "BTV",
    "KOTAKBANK": "KMB",
    "LT": "LT",
    "ITC": "ITC",
    "AXISBANK": "AB16",
    "BAJFINANCE": "BAF",
    "MARUTI": "MS24",
    "SUNPHARMA": "SU12",
    "TITAN": "TI01",
    "WIPRO": "WI",
    "HCLTECH": "HCL02",
    "ASIANPAINT": "AP31",
    "ADANIENT": "ADE01",
    "ADANIPORTS": "APL",
    "TATAMOTORS": "TM03",
    "TATASTEEL": "TIS",
    "NTPC": "NTP",
    "POWERGRID": "PGC",
    "ONGC": "ONG",
    "COALINDIA": "CI11",
    "JSWSTEEL": "JSW01",
    "ULTRACEMCO": "GCI02",
    "BAJAJFINSV": "BAF01",
    "TECHM": "TM04",
    "INDUSINDBK": "IIB",
    "HINDALCO": "HZ",
    "NESTLEIND": "NI15",
    "DIVISLAB": "DL03",
    "DRREDDY": "DRR",
    "CIPLA": "CI",
    "VEDL": "SG",
    "BPCL": "BPC",
    "GRASIM": "GR",
    "EICHERMOT": "EIM",
    "HEROMOTOCO": "HHM02",
    "M&M": "MM",
    "TATACONSUM": "TC22",
    "APOLLOHOSP": "AHS",
    "SBILIFE": "SLI02",
    "HDFCLIFE": "HLI02",
    "BRITANNIA": "BI",
}

# Ticker human-readable search keywords for relevance matching.
# E.g. articles say "HDFC Bank" or "Infosys", not "HDFCBANK" or "INFY".
_TICKER_KEYWORDS: Dict[str, List[str]] = {
    "RELIANCE": ["reliance"],
    "TCS": ["tcs", "tata consultancy"],
    "HDFCBANK": ["hdfc bank", "hdfc"],
    "INFY": ["infosys", "infy"],
    "ICICIBANK": ["icici bank", "icici"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "SBIN": ["sbi", "state bank"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "KOTAKBANK": ["kotak mahindra", "kotak bank", "kotak"],
    "LT": ["larsen", "l&t"],
    "ITC": ["itc"],
    "AXISBANK": ["axis bank", "axis"],
    "BAJFINANCE": ["bajaj finance"],
    "MARUTI": ["maruti suzuki", "maruti"],
    "SUNPHARMA": ["sun pharma", "sun pharmaceutical"],
    "TITAN": ["titan"],
    "WIPRO": ["wipro"],
    "HCLTECH": ["hcl tech", "hcl"],
    "ASIANPAINT": ["asian paints", "asian paint"],
    "ADANIENT": ["adani enterprises", "adani"],
    "ADANIPORTS": ["adani ports", "adani"],
    "TATAMOTORS": ["tata motors"],
    "TATASTEEL": ["tata steel"],
    "NTPC": ["ntpc"],
    "POWERGRID": ["power grid"],
    "ONGC": ["ongc"],
    "COALINDIA": ["coal india"],
    "JSWSTEEL": ["jsw steel", "jsw"],
    "ULTRACEMCO": ["ultratech cement", "ultratech"],
    "BAJAJFINSV": ["bajaj finserv"],
    "TECHM": ["tech mahindra"],
    "INDUSINDBK": ["indusind bank", "indusind"],
    "HINDALCO": ["hindalco"],
    "NESTLEIND": ["nestle india", "nestle"],
    "DIVISLAB": ["divi's lab", "divis lab"],
    "DRREDDY": ["dr reddy", "dr. reddy"],
    "CIPLA": ["cipla"],
    "VEDL": ["vedanta"],
    "BPCL": ["bpcl", "bharat petroleum"],
    "GRASIM": ["grasim"],
    "EICHERMOT": ["eicher motors", "eicher"],
    "HEROMOTOCO": ["hero motocorp", "hero moto"],
    "M&M": ["mahindra", "m&m"],
    "TATACONSUM": ["tata consumer"],
    "APOLLOHOSP": ["apollo hospital", "apollo"],
    "SBILIFE": ["sbi life"],
    "HDFCLIFE": ["hdfc life"],
    "BRITANNIA": ["britannia"],
}

# RSS feeds to try (order: most likely to contain stock-specific news first)
_RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/business.xml",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
]


class MoneycontrolScraper(BaseNewsScraper):
    """Scraper for Moneycontrol.com stock news."""

    def __init__(self):
        super().__init__("Moneycontrol", "https://www.moneycontrol.com")

    # ── public entry point ───────────────────────────────────────────

    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        company = ticker.replace(".NS", "").replace(".BO", "")
        # Build list of keywords for relevance matching
        keywords = _TICKER_KEYWORDS.get(company.upper(), [company.lower()])
        logger.info("Moneycontrol: Fetching news for %s …", ticker)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.moneycontrol.com/",
        }

        # ── 1. Resolve sc_id ────────────────────────────────────────
        sc_id = await self._resolve_sc_id(company, headers)

        # ── 2. Primary: stock_news.php (server-rendered) ────────────
        if sc_id:
            items = await self._fetch_stock_news_page(sc_id, company, ticker, headers, keywords)
            if items:
                logger.info(
                    "Moneycontrol: Got %d articles for %s via stock_news.php",
                    len(items), ticker,
                )
                return items

        # ── 3. Fallback: RSS feeds ──────────────────────────────────
        rss_items = await self._try_rss_feeds(company, ticker, keywords)
        if rss_items:
            logger.info(
                "Moneycontrol: Got %d articles for %s via RSS fallback",
                len(rss_items), ticker,
            )
            return rss_items

        logger.warning("Moneycontrol: No articles found for %s", ticker)
        return []

    # ── sc_id resolution ─────────────────────────────────────────────

    async def _resolve_sc_id(self, company: str, headers: dict) -> Optional[str]:
        """Return the Moneycontrol internal stock code for *company*."""
        # Fast path: static map
        upper = company.upper()
        if upper in _TICKER_SCID:
            return _TICKER_SCID[upper]

        # Slow path: autosuggestion API
        api_url = (
            f"{self.base_url}/mccode/common/autosuggestion_solr.php"
            f"?classic=true&query={quote_plus(company)}&type=1&format=json"
        )
        html = await self._fetch_html(api_url, headers)
        if not html:
            return None
        try:
            data = json.loads(html)
            if isinstance(data, list) and data:
                return data[0].get("sc_id")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        return None

    # ── stock_news.php parsing ───────────────────────────────────────

    async def _fetch_stock_news_page(
        self,
        sc_id: str,
        company: str,
        ticker: str,
        headers: dict,
        keywords: List[str] = (),
    ) -> List[NewsItem]:
        """Fetch & parse the server-rendered stock_news.php page."""
        url = (
            f"{self.base_url}/stocks/company_info/stock_news.php"
            f"?sc_id={quote_plus(sc_id)}&durtype=M&dur=3&page=1"
        )
        html = await self._fetch_html(url, headers)
        if not html:
            return []

        soup = self._parse_html(html)
        items: List[NewsItem] = []

        containers = soup.select("div.MT15.PT10.PB10")
        for container in containers:
            try:
                # Title
                title_el = container.select_one("a.g_14bl")
                if not title_el:
                    title_el = container.select_one("a.arial11_summ")
                if not title_el:
                    continue
                title = self._extract_text(title_el)
                if not title or len(title) < 10:
                    continue

                # URL (prefer the title link, fall back to image link)
                link_url = title_el.get("href", "")
                if not link_url:
                    img_link = container.select_one("a.arial11_summ")
                    link_url = img_link.get("href", "") if img_link else ""
                if link_url and not link_url.startswith("http"):
                    link_url = self.base_url + link_url

                # Summary
                summary_el = container.select_one("p.PT3")
                summary_text = self._extract_text(summary_el) if summary_el else ""
                # The p.PT3 may start with a date like "1.27 pm | 20 Feb 2026"
                timestamp = self._parse_mc_date(summary_text)
                # Strip the date prefix from the summary
                summary_text = re.sub(
                    r"^\d{1,2}\.\d{2}\s*(?:am|pm)\s*\|\s*\d{1,2}\s+\w+\s+\d{4}\s*",
                    "", summary_text, flags=re.IGNORECASE,
                ).strip()
                if not summary_text:
                    summary_text = title

                # Relevance: keep articles that mention any keyword
                combined = (title + " " + summary_text).lower()
                if not any(kw in combined for kw in keywords):
                    continue

                items.append(NewsItem(
                    title=title,
                    summary=summary_text[:500],
                    url=link_url,
                    timestamp=timestamp,
                    source=self.source_name,
                    ticker=ticker,
                    category=self._categorize_news(title),
                ))
            except Exception:
                continue

        return items[:15]

    # ── RSS fallback ─────────────────────────────────────────────────

    async def _try_rss_feeds(
        self, company: str, ticker: str, keywords: List[str] = ()
    ) -> List[NewsItem]:
        """Try multiple RSS feeds and aggregate matches."""
        kws = keywords or [company.lower()]
        items: List[NewsItem] = []
        seen_titles: set = set()

        for feed_url in _RSS_FEEDS:
            xml = await self._fetch_html(feed_url)
            if not xml:
                continue
            try:
                soup = self._parse_html(xml, parser="lxml-xml")
                for entry in soup.find_all("item"):
                    title = self._extract_text(entry.find("title"))
                    if not title or not any(kw in title.lower() for kw in kws):
                        continue
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    link = self._extract_text(entry.find("link"))
                    desc = self._extract_text(entry.find("description"))
                    pub = entry.find("pubDate")
                    ts = self._parse_rss_date(pub) if pub else datetime.utcnow()

                    items.append(NewsItem(
                        title=title,
                        summary=(desc or title)[:500],
                        url=link or "",
                        timestamp=ts,
                        source=self.source_name,
                        ticker=ticker,
                        category=self._categorize_news(title),
                    ))
            except Exception:
                continue

        return items[:15]

    # ── date helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_mc_date(text: str) -> datetime:
        """Parse Moneycontrol date like ``'1.27 pm | 20 Feb 2026'``."""
        if not text:
            return datetime.utcnow()
        m = re.search(
            r"(\d{1,2})\.(\d{2})\s*(am|pm)\s*\|\s*(\d{1,2})\s+(\w+)\s+(\d{4})",
            text, re.IGNORECASE,
        )
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            ampm = m.group(3).lower()
            day, month_str, year = int(m.group(4)), m.group(5), int(m.group(6))
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            try:
                month = datetime.strptime(month_str, "%b").month
                return datetime(year, month, day, hour, minute)
            except ValueError:
                pass
        return datetime.utcnow()

    @staticmethod
    def _parse_rss_date(el) -> datetime:
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(el.get_text(strip=True))
        except Exception:
            return datetime.utcnow()
