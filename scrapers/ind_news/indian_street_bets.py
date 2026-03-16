"""
r/IndianStreetBets (Reddit) scraper.

Scrapes r/IndianStreetBets posts across multiple flairs and filters
for mentions of user-selected Indian stock tickers.

Mirrors the architecture of ``WallStreetBetsScraper`` — uses Reddit's
public JSON API (no OAuth) with flair-based search and regex ticker
matching.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp

from models import NewsItem
from scrapers import BaseNewsScraper
from config import Config

logger = logging.getLogger(__name__)

# ── ISB-specific constants ──────────────────────────────────────────
_FLAIRS = [
    "DD",
    "Discussion",
    "News",
    "Gain",
    "Loss",
    "Meme",              # ISB is meme-heavy
    "Technical%20Analysis",
    "Fundamental%20Analysis",
    "Advice",
    "Options",
]

_BASE_SEARCH_URL = (
    "https://www.reddit.com/r/IndianStreetBets/search.json"
    "?sort=hot&restrict_sr=on&t=day&q=flair%3A{flair}&limit=50"
)

# Reddit's public JSON API needs a descriptive User-Agent
_REDDIT_UA = (
    "Centurion/1.0 (IndianStreetBets scraper; "
    "+https://github.com/centurion-capital)"
)


class IndianStreetBetsScraper(BaseNewsScraper):
    """
    Scraper for r/IndianStreetBets that filters posts by user-selected
    Indian stock tickers.

    Accepts tickers in NSE format (``RELIANCE``, ``RELIANCE.NS``,
    ``RELIANCE.BO``).  The ``.NS`` / ``.BO`` suffix is stripped before
    matching against Reddit post content.
    """

    def __init__(self):
        super().__init__(
            source_name="IndianStreetBets",
            base_url="https://www.reddit.com/r/IndianStreetBets",
        )

    # ── public interface (matches BaseNewsScraper) ──────────────────

    async def fetch_news(self, ticker: str) -> List[NewsItem]:
        """
        Fetch r/IndianStreetBets posts that mention *ticker*.

        Searches across all major flairs from the last 24 hours, then
        filters posts whose title or selftext contains the ticker
        symbol (case-insensitive word-boundary match or ``$TICKER``).
        """
        news_items: List[NewsItem] = []

        # Strip .NS / .BO suffix for Reddit matching
        ticker_clean = (
            ticker.upper()
            .replace(".NS", "")
            .replace(".BO", "")
        )

        # Pre-compile a pattern that matches the ticker as:
        #   • standalone word  (e.g. "RELIANCE" surrounded by spaces/punctuation)
        #   • cash-tag / hashtag (e.g. "$RELIANCE" or "#RELIANCE")
        pattern = re.compile(
            rf"(?<!\w)[\$#]?{re.escape(ticker_clean)}(?!\w)",
            re.IGNORECASE,
        )

        headers = {
            "User-Agent": _REDDIT_UA,
            "Accept": "application/json",
        }

        logger.info(
            "IndianStreetBets: Fetching posts mentioning %s…", ticker_clean,
        )

        timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT + 5)

        try:
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True,
            ) as session:
                for flair in _FLAIRS:
                    url = _BASE_SEARCH_URL.format(flair=flair)
                    items = await self._fetch_flair(
                        session, url, headers, ticker_clean, pattern, flair,
                    )
                    news_items.extend(items)

                    # Respect Reddit rate limits — small pause between flairs
                    if items is not None:
                        await asyncio.sleep(1.5)

        except Exception as exc:
            logger.warning(
                "IndianStreetBets: top-level error for %s — %s: %s",
                ticker_clean, type(exc).__name__, exc,
            )

        logger.info(
            "IndianStreetBets: Found %d posts mentioning %s",
            len(news_items), ticker_clean,
        )
        return news_items

    # ── internals ───────────────────────────────────────────────────

    async def _fetch_flair(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict,
        ticker: str,
        pattern: re.Pattern,
        flair: str,
    ) -> List[NewsItem]:
        """Fetch one flair page and return matching NewsItems."""
        items: List[NewsItem] = []

        try:
            async with session.get(url, headers=headers, ssl=False) as resp:
                if resp.status == 429:
                    logger.warning(
                        "IndianStreetBets: Rate-limited (429) on flair %s", flair,
                    )
                    return items
                if resp.status != 200:
                    logger.debug(
                        "IndianStreetBets: HTTP %d for flair %s",
                        resp.status, flair,
                    )
                    return items

                data = await resp.json()

        except Exception as exc:
            logger.debug(
                "IndianStreetBets: Error fetching flair %s — %s", flair, exc,
            )
            return items

        # Reddit JSON structure: data -> children -> [{ kind, data }]
        children = (
            data.get("data", {}).get("children", [])
            if isinstance(data, dict)
            else []
        )

        for child in children:
            post = child.get("data", {}) if isinstance(child, dict) else {}
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            combined = f"{title} {selftext}"

            # Only keep posts that actually mention our ticker
            if not pattern.search(combined):
                continue

            item = self._post_to_news_item(post, ticker, flair)
            if item:
                items.append(item)

        return items

    def _post_to_news_item(
        self, post: dict, ticker: str, flair: str,
    ) -> Optional[NewsItem]:
        """Convert a Reddit post dict into a ``NewsItem``."""
        title = post.get("title", "").strip()
        if not title:
            return None

        selftext = post.get("selftext", "").strip()
        # Trim very long self-posts to a reasonable summary length
        summary = selftext[:500] if selftext else title

        # Permalink full URL
        permalink = post.get("permalink", "")
        url = f"https://www.reddit.com{permalink}" if permalink else ""

        # Timestamp
        created_utc = post.get("created_utc")
        if created_utc:
            try:
                timestamp = datetime.fromtimestamp(
                    float(created_utc), tz=timezone.utc,
                )
            except (ValueError, OSError):
                timestamp = datetime.now(tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        # Enrich title with flair + score for context
        score = post.get("score", 0)
        num_comments = post.get("num_comments", 0)
        enriched_title = (
            f"[ISB/{flair}] {title} ({score}, {num_comments})"
        )

        category = self._categorize_news(f"{title} {summary}")

        return NewsItem(
            title=enriched_title,
            summary=summary,
            url=url,
            timestamp=timestamp,
            source=self.source_name,
            ticker=ticker,
            category=category,
        )
