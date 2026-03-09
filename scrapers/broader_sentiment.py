# Centurion Capital LLC — Google Search Public Sentiment Scraper
"""
Performs a Google search for each ticker, fetches the top 5-6 organic
results, extracts readable text from each page, and runs sentiment
analysis to derive a **public sentiment** reading:
    • fearful   (strong negative bias in public discourse)
    • cautious  (mildly negative)
    • neutral
    • optimistic (mildly positive)
    • greedy    (strong positive / hype)

This provides an independent sentiment signal that captures what retail
investors and the general public are reading, separate from institutional
financial news sources.

Implementation notes
────────────────────
* Google search is performed via HTTP with rotating User-Agent headers.
  No Google API key is required—we parse the standard HTML result page.
* To respect rate limits, results are cached per ticker for 30 minutes.
* Page text extraction uses BeautifulSoup with readability heuristics:
  strip nav/header/footer/aside, keep <p> and <article> text.
* Sentiment classification uses the same DistilBERT model as the main
  pipeline (lazy-loaded singleton) so there is zero extra model cost.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import aiohttp
from bs4 import BeautifulSoup

from models import SentimentLabel

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single Google search result with extracted content."""
    title: str
    url: str
    snippet: str
    page_text: str = ""
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[SentimentLabel] = None
    sentiment_confidence: Optional[float] = None


@dataclass
class PublicSentiment:
    """Aggregated public sentiment for a stock ticker."""
    ticker: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    results_analyzed: int = 0
    avg_sentiment_score: float = 0.0
    sentiment_label: str = "neutral"          # fearful / cautious / neutral / optimistic / greedy
    positive_pct: float = 0.0
    negative_pct: float = 0.0
    neutral_pct: float = 0.0
    results: List[SearchResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "results_analyzed": self.results_analyzed,
            "avg_sentiment_score": round(self.avg_sentiment_score, 4),
            "sentiment_label": self.sentiment_label,
            "positive_pct": round(self.positive_pct, 2),
            "negative_pct": round(self.negative_pct, 2),
            "neutral_pct": round(self.neutral_pct, 2),
        }


# ── Constants ────────────────────────────────────────────────────────

_MAX_RESULTS = 6
_MAX_PAGE_CHARS = 3000          # truncate extracted page text
_SEARCH_CACHE_TTL = timedelta(minutes=30)
_REQUEST_TIMEOUT = 12           # seconds

# User-Agent pool (rotated per request)
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
]

# Domains to skip when fetching page text (social media, paywalls)
_SKIP_DOMAINS = frozenset({
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "tiktok.com", "reddit.com",
    "linkedin.com", "pinterest.com",
})


# ── Core class ───────────────────────────────────────────────────────

class GoogleSearchSentiment:
    """
    Scrapes Google search results for a stock ticker and derives
    aggregate public sentiment from the content of the top pages.

    Usage::

        gss = GoogleSearchSentiment()
        ps = await gss.analyze("AAPL")
        print(ps.sentiment_label, ps.avg_sentiment_score)
    """

    # Class-level cache: ticker (PublicSentiment, timestamp)
    _cache: Dict[str, Tuple[PublicSentiment, datetime]] = {}

    def __init__(self):
        self._ua_idx = 0

    def _next_ua(self) -> str:
        ua = _USER_AGENTS[self._ua_idx % len(_USER_AGENTS)]
        self._ua_idx += 1
        return ua

    # ── Public API ───────────────────────────────────────────────────

    async def analyze(self, ticker: str) -> PublicSentiment:
        """
        Search Google for ``<ticker> stock outlook``, fetch top pages,
        and return aggregate sentiment.

        Results are cached for 30 minutes.
        """
        now = datetime.utcnow()
        if ticker in self._cache:
            cached, ts = self._cache[ticker]
            if (now - ts) < _SEARCH_CACHE_TTL:
                logger.debug("GoogleSearchSentiment: cache hit for %s", ticker)
                return cached

        company = ticker.replace(".NS", "").replace(".BO", "")

        # Step 1: Web search (Google DuckDuckGo fallback)
        results = await self._google_search(company)
        if not results:
            results = await self._duckduckgo_search(company)
        if not results:
            ps = PublicSentiment(ticker=ticker, timestamp=now)
            self._cache[ticker] = (ps, now)
            return ps

        # Step 2: Fetch page text for each result (concurrent, limited)
        await self._fetch_page_texts(results)

        # Step 3: Run sentiment analysis on each result
        self._analyze_results(results)

        # Step 4: Aggregate
        ps = self._aggregate(ticker, results, now)
        self._cache[ticker] = (ps, now)

        logger.info(
            "GoogleSearchSentiment: %s %s (score=%.2f, %d pages)",
            ticker, ps.sentiment_label, ps.avg_sentiment_score,
            ps.results_analyzed,
        )
        return ps

    async def analyze_multiple(self, tickers: List[str]) -> Dict[str, PublicSentiment]:
        """Analyze multiple tickers sequentially (respecting rate limits)."""
        out: Dict[str, PublicSentiment] = {}
        for ticker in tickers:
            out[ticker] = await self.analyze(ticker)
            await asyncio.sleep(2.0)   # polite delay between searches
        return out

    # ── Step 1: Google search ────────────────────────────────────────

    async def _google_search(self, company: str) -> List[SearchResult]:
        """
        Perform a Google search and parse organic results.

        Query: ``<company> stock outlook sentiment analysis``
        """
        query = f"{company} stock outlook sentiment analysis"
        url = f"https://www.google.com/search?q={quote_plus(query)}&num={_MAX_RESULTS + 2}&hl=en"

        headers = {
            "User-Agent": self._next_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout, trust_env=True
            ) as session:
                async with session.get(url, headers=headers, ssl=False) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "GoogleSearchSentiment: Google returned HTTP %d",
                            resp.status,
                        )
                        return []
                    html = await resp.text()
        except Exception as exc:
            logger.warning("GoogleSearchSentiment: search request failed — %s", exc)
            return []

        return self._parse_search_results(html)

    def _parse_search_results(self, html: str) -> List[SearchResult]:
        """Extract organic results from Google's HTML."""
        soup = BeautifulSoup(html, "lxml")
        results: List[SearchResult] = []

        # Google wraps each result in a <div class="g"> or similar
        for g in soup.select("div.g, div[data-hveid]"):
            if len(results) >= _MAX_RESULTS:
                break

            link_el = g.select_one("a[href]")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                continue

            # Skip unwanted domains
            domain = urlparse(href).netloc.lower().replace("www.", "")
            if domain in _SKIP_DOMAINS:
                continue
            # Skip Google's own pages
            if "google.com" in domain:
                continue

            title_el = g.select_one("h3")
            title = title_el.get_text(strip=True) if title_el else ""
            if not title:
                continue

            # Snippet
            snippet_el = (
                g.select_one("div[data-sncf], div.VwiC3b, span.st, div[class*='snippet']")
            )
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

            results.append(SearchResult(
                title=title,
                url=href,
                snippet=snippet[:500],
            ))

        return results

    # ── Step 1b: DuckDuckGo fallback ─────────────────────────────────

    async def _duckduckgo_search(self, company: str) -> List[SearchResult]:
        """
        Fallback search engine when Google blocks or returns empty.

        Uses DuckDuckGo's HTML-lite interface which is more tolerant of
        automated requests.
        """
        query = f"{company} stock outlook sentiment"
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        headers = {
            "User-Agent": self._next_ua(),
            "Accept": "text/html",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout, trust_env=True
            ) as session:
                async with session.get(url, headers=headers, ssl=False) as resp:
                    if resp.status != 200:
                        return []
                    html = await resp.text()
        except Exception as exc:
            logger.warning("GoogleSearchSentiment: DuckDuckGo failed — %s", exc)
            return []

        soup = BeautifulSoup(html, "lxml")
        results: List[SearchResult] = []

        for item in soup.select("div.result, div.links_main"):
            if len(results) >= _MAX_RESULTS:
                break

            link_el = item.select_one("a.result__a, a.result__url, a[href]")
            if not link_el:
                continue
            href = link_el.get("href", "")
            if not href.startswith("http"):
                continue

            domain = urlparse(href).netloc.lower().replace("www.", "")
            if domain in _SKIP_DOMAINS or "duckduckgo.com" in domain:
                continue

            title = link_el.get_text(strip=True)
            if not title or len(title) < 5:
                continue

            snippet_el = item.select_one("a.result__snippet, div.result__snippet")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

            results.append(SearchResult(
                title=title,
                url=href,
                snippet=snippet[:500],
            ))

        logger.debug("DuckDuckGo: found %d results for %s", len(results), company)
        return results

    # ── Step 2: Fetch page text ──────────────────────────────────────

    async def _fetch_page_texts(self, results: List[SearchResult]) -> None:
        """Fetch and extract readable text from each result URL."""
        sem = asyncio.Semaphore(3)  # at most 3 concurrent page fetches

        async def _fetch_one(sr: SearchResult) -> None:
            async with sem:
                sr.page_text = await self._extract_page_text(sr.url)

        await asyncio.gather(*[_fetch_one(r) for r in results])

    async def _extract_page_text(self, url: str) -> str:
        """Fetch a URL and extract article-like readable text."""
        headers = {
            "User-Agent": self._next_ua(),
            "Accept": "text/html,application/xhtml+xml",
        }
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
            connector = aiohttp.TCPConnector(limit=10, force_close=True)
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout, trust_env=True
            ) as session:
                async with session.get(url, headers=headers, ssl=False) as resp:
                    if resp.status != 200:
                        return ""
                    ct = resp.headers.get("Content-Type", "")
                    if "text/html" not in ct and "application/xhtml" not in ct:
                        return ""
                    html = await resp.text()
        except Exception:
            return ""

        return self._html_to_text(html)

    @staticmethod
    def _html_to_text(html: str) -> str:
        """
        Extract readable article text from raw HTML.

        Strips navigation, headers, footers, asides, scripts, styles,
        then concatenates <p> and <article> text.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "nav", "header",
                                   "footer", "aside", "form", "iframe"]):
            tag.decompose()

        # Prefer <article> content
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        text_parts = []
        for p in paragraphs:
            txt = p.get_text(strip=True)
            if len(txt) > 30:  # skip tiny fragments
                text_parts.append(txt)

        combined = " ".join(text_parts)
        # Collapse whitespace
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined[:_MAX_PAGE_CHARS]

    # ── Step 3: Sentiment analysis ───────────────────────────────────

    def _analyze_results(self, results: List[SearchResult]) -> None:
        """
        Run DistilBERT sentiment analysis on each result's combined
        title + snippet + page_text.

        Uses the same lazy-loaded singleton as ``SentimentAnalyzer``.
        """
        from sentiment.analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()

        for sr in results:
            text = f"{sr.title}. {sr.snippet}. {sr.page_text}"
            if len(text.strip()) < 20:
                sr.sentiment_score = 0.0
                sr.sentiment_label = SentimentLabel.NEUTRAL
                sr.sentiment_confidence = 0.5
                continue

            score, label, conf = analyzer.analyze(text)
            sr.sentiment_score = score
            sr.sentiment_label = label
            sr.sentiment_confidence = conf

    # ── Step 4: Aggregate ────────────────────────────────────────────

    @staticmethod
    def _aggregate(
        ticker: str,
        results: List[SearchResult],
        ts: datetime,
    ) -> PublicSentiment:
        """
        Combine per-page sentiments into an overall public sentiment.

        Weighting: pages with actual page_text get 2x weight vs
        snippet-only results.
        """
        if not results:
            return PublicSentiment(ticker=ticker, timestamp=ts)

        total_weight = 0.0
        weighted_sum = 0.0
        pos_count = 0
        neg_count = 0
        neu_count = 0

        for sr in results:
            if sr.sentiment_score is None:
                continue
            w = 2.0 if sr.page_text else 1.0
            weighted_sum += sr.sentiment_score * w
            total_weight += w

            if sr.sentiment_label == SentimentLabel.POSITIVE:
                pos_count += 1
            elif sr.sentiment_label == SentimentLabel.NEGATIVE:
                neg_count += 1
            else:
                neu_count += 1

        n = pos_count + neg_count + neu_count
        if n == 0 or total_weight == 0:
            return PublicSentiment(ticker=ticker, timestamp=ts)

        avg_score = weighted_sum / total_weight

        # Map score to label
        if avg_score >= 0.5:
            label = "greedy"
        elif avg_score >= 0.15:
            label = "optimistic"
        elif avg_score <= -0.5:
            label = "fearful"
        elif avg_score <= -0.15:
            label = "cautious"
        else:
            label = "neutral"

        return PublicSentiment(
            ticker=ticker,
            timestamp=ts,
            results_analyzed=n,
            avg_sentiment_score=avg_score,
            sentiment_label=label,
            positive_pct=pos_count / n * 100,
            negative_pct=neg_count / n * 100,
            neutral_pct=neu_count / n * 100,
            results=results,
        )
