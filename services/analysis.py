"""
Analysis Service Module for Centurion Capital LLC.

Contains the stock analysis execution logic and database persistence.

Session-cache integration
─────────────────────────
* News, sentiment results, and stock metrics are cached per ticker in
  ``SessionCache`` so that re-running analysis with overlapping tickers
  only fetches data for the *new* tickers.
* ``MetricsCalculator`` now caches per unique ticker internally, so even
  within a single run, 10 news articles for AAPL trigger only 1 yfinance
  call (instead of 10).
"""

import logging
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from config import Config
from database.service import get_database_service
from main import AlgoTradingSystem
from services.cache import get_session_cache
from scrapers.cache import get_scraper_cache

logger = logging.getLogger(__name__)

DB_AVAILABLE = Config.is_database_configured()


async def run_analysis_async(
    tickers: List[str],
    progress_callback=None,
) -> List[Any]:
    """
    Run the stock analysis asynchronously, reusing cached data
    for tickers that were already analysed in this session.
    
    Args:
        tickers: List of stock ticker symbols to analyze
    
    Returns:
        List of TradingSignal objects
    """
    market = st.session_state.get("current_market", "US")
    system = AlgoTradingSystem(tickers=tickers, market=market)
    cache = get_session_cache()
    
    # Status tracking
    status_placeholder = st.empty()
    
    signals = await _execute_analysis(
        system, tickers, cache,
        status_placeholder,
        progress_callback=progress_callback,
    )

    if not signals:
        return []

    # Persist results
    save_path = _save_results(system, signals, tickers)
    db_saved = _save_to_database(signals, tickers)

    # Show cache diagnostics
    stats = cache.stats
    scraper_stats = get_scraper_cache().stats
    logger.info("Session cache stats: %s", stats)
    logger.info("Scraper cache stats: %s", scraper_stats)

    # Clear status indicator
    status_placeholder.empty()

    if save_path:
        st.caption(f" Results saved to: {save_path}")

    # Show how many API calls were saved
    cached_news_count = len(cache.get_cached_tickers("news"))
    scraper_cached = scraper_stats.get('cached_tickers', 0)
    dedup_hashes = scraper_stats.get('content_hashes', 0)
    if cached_news_count > 0 or scraper_cached > 0:
        st.caption(
            f" Cache: {stats['hits']} hits / {stats['misses']} misses "
            f"({stats['hit_rate']} hit rate) — "
            f"{cached_news_count} ticker(s) cached, "
            f"{dedup_hashes} dedup hashes tracked"
        )

    return signals


async def _execute_analysis(
    system: AlgoTradingSystem,
    tickers: List[str],
    cache,
    status_placeholder,
    progress_callback=None,
) -> List[Any]:
    """
    Execute the main analysis pipeline with session-cache integration.
    
    Flow:
    1. **News** — only scrape tickers not yet in cache
    2. **Sentiment** — only analyse news items not yet analysed
    3. **Metrics** — MetricsCalculator has its own per-ticker cache;
       we also persist to SessionCache for cross-run reuse
    4. **Signals** — generated for *all* tickers (cached + new)
    """
    # ── Step 1: News scraping (with cache) ───────────────────────────
    # (Spinner already shows status; skip redundant info notification)
    if progress_callback:
        progress_callback(2, " Scraping news")
    
    # Build dict of cached news
    cached_news: Dict[str, list] = cache.get_all("news")
    new_tickers = cache.get_new_tickers("news", tickers)
    
    if progress_callback:
        progress_callback(5, f" Fetching news for {len(new_tickers)} new ticker(s)")

    if cached_news:
        cached_list = [t for t in tickers if t in cached_news]
        if cached_list and progress_callback:
            progress_callback(5, f" Reusing cache for {len(cached_list)} ticker(s), fetching {len(new_tickers)} new")
    
    if progress_callback:
        progress_callback(8, " Querying news sources")

    all_news = await system.news_aggregator.fetch_news_for_tickers(
        tickers, cached_news=cached_news
    )
    if not all_news:
        status_placeholder.warning("\u26a0\ufe0f No news found")
        return []
    
    if progress_callback:
        progress_callback(22, f" Collected {len(all_news)} articles — caching")

    # Cache newly fetched news by ticker
    news_ttl = timedelta(minutes=Config.NEWS_CACHE_TTL_MINUTES)
    metrics_ttl = timedelta(minutes=Config.METRICS_CACHE_TTL_MINUTES)
    
    news_by_ticker: Dict[str, list] = defaultdict(list)
    for item in all_news:
        news_by_ticker[item.ticker].append(item)
    for ticker, items in news_by_ticker.items():
        if ticker not in cached_news:
            cache.put("news", ticker, items, ttl=news_ttl)
            # Record successful scrape in DataFreshness table
            get_scraper_cache().record_fetch_to_db(
                ticker, record_count=len(items),
            )
    
    if progress_callback:
        progress_callback(28, f" {len(all_news)} news items collected")
    
    # ── Step 2: Sentiment analysis (with cache) ──────────────────────
    # (Spinner already shows status; skip redundant info notification)
    if progress_callback:
        progress_callback(30, "Analyzing sentiment")
    
    # Separate already-analysed items from new ones
    items_to_analyse = []
    already_analysed = []
    for item in all_news:
        if item.sentiment_label is not None:
            already_analysed.append(item)
        else:
            items_to_analyse.append(item)
    
    if progress_callback:
        progress_callback(35, f"{len(items_to_analyse)} items to analyse ({len(already_analysed)} cached)")

    if items_to_analyse:
        newly_analysed = system.sentiment_analyzer.analyze_news_items(items_to_analyse)
    else:
        newly_analysed = []
    
    if progress_callback:
        progress_callback(48, "Caching sentiment results")

    analyzed_news = already_analysed + newly_analysed
    
    # Update cache with sentiment-enriched items
    enriched_by_ticker: Dict[str, list] = defaultdict(list)
    for item in analyzed_news:
        enriched_by_ticker[item.ticker].append(item)
    for ticker, items in enriched_by_ticker.items():
        cache.put("sentiment", ticker, items, ttl=news_ttl)
        # Also update the news cache so next time we already have sentiments
        cache.put("news", ticker, items, ttl=news_ttl)
    
    if progress_callback:
        progress_callback(52, f" Sentiment done — {len(analyzed_news)} items")


    # ── Step 2b: Macro-economic indicators ───────────────────────────
    if progress_callback:
        progress_callback(53, " Fetching macro-economic indicators")

    market = system.market
    try:
        macro_snap = system.macro_indicators.fetch(market=market)
        system.decision_engine.set_macro_snapshot(macro_snap)
        logger.info(
            "Macro sentiment: %s (%.2f)",
            macro_snap.macro_sentiment_label or "n/a",
            macro_snap.macro_sentiment_score or 0,
        )
    except Exception as exc:
        logger.warning("Macro indicators unavailable: %s", exc)

    if progress_callback:
        progress_callback(55, "\u2705 Macro indicators loaded")

    # ── Step 2c: Google search public sentiment ──────────────────────
    if progress_callback:
        progress_callback(56, " Analyzing public sentiment (Google)")

    try:
        unique_tickers_for_gs = list({item.ticker for item in analyzed_news})
        public_sentiments = await system.broader_sentiment.analyze_multiple(
            unique_tickers_for_gs
        )
        system.decision_engine.set_public_sentiments(public_sentiments)
        for t, ps in public_sentiments.items():
            logger.info(
                "Public sentiment %s: %s (%.2f, %d pages)",
                t, ps.sentiment_label, ps.avg_sentiment_score, ps.results_analyzed,
            )
    except Exception as exc:
        logger.warning("Google public sentiment unavailable: %s", exc)

    if progress_callback:
        progress_callback(58, "\u2705 Public sentiment analyzed")
    
    # ── Step 3: Metrics + signals ────────────────────────────────────
    # (Spinner already shows status; skip redundant info notification)
    if progress_callback:
        progress_callback(55, " Calculating metrics")
    
    # Pre-populate MetricsCalculator cache with any previously cached metrics
    cached_metrics = cache.get_all("metrics")
    for ticker, m in cached_metrics.items():
        if ticker not in system.metrics_calculator._cache:
            system.metrics_calculator._cache[ticker] = m
    
    # Prefetch metrics for all unique tickers at once
    unique_tickers = list({item.ticker for item in analyzed_news})
    if progress_callback:
        progress_callback(58, f" Fetching metrics for {len(unique_tickers)} ticker(s)")

    system.metrics_calculator.prefetch_metrics(unique_tickers)
    
    if progress_callback:
        progress_callback(70, " Caching metrics & recording freshness")

    # Save back to session cache + record freshness
    sc = get_scraper_cache()
    for idx, t in enumerate(unique_tickers):
        m = system.metrics_calculator.get_stock_metrics(t)
        cache.put("metrics", t, m, ttl=metrics_ttl)
        # Record metrics freshness so we don't re-fetch after a restart
        try:
            from database.service import get_database_service
            get_database_service().record_fetch(
                t, data_type="fundamentals",
                refresh_minutes=Config.METRICS_CACHE_TTL_MINUTES,
            )
        except Exception:
            pass
        if progress_callback and unique_tickers:
            pct = 70 + int(10 * (idx + 1) / len(unique_tickers))
            progress_callback(pct, f" Metrics cached — {t}")
    
    # Generate signals
    if progress_callback:
        progress_callback(82, " Generating trading signals")

    signals = []
    total_news = len(analyzed_news)
    for i, news_item in enumerate(analyzed_news):
        metrics = system.metrics_calculator.get_stock_metrics(news_item.ticker)
        signal = system.decision_engine.generate_signal(news_item, metrics)
        signals.append(signal)
        if progress_callback and total_news:
            pct = 82 + int(10 * (i + 1) / total_news)
            if (i + 1) % max(1, total_news // 5) == 0 or i + 1 == total_news:
                progress_callback(pct, f" Signal {i + 1}/{total_news}")
        
    # Cache signals
    if progress_callback:
        progress_callback(93, " Caching signals")

    signals_by_ticker: Dict[str, list] = defaultdict(list)
    for sig in signals:
        signals_by_ticker[sig.news_item.ticker].append(sig)
    for ticker, sigs in signals_by_ticker.items():
        cache.put("signals", ticker, sigs)
    
    if progress_callback:
        progress_callback(95, f"\u2705 {len(signals)} trading signals generated")
    if progress_callback:
        progress_callback(100, "Analysis complete ")
    
    # ── Step 4: WSB email report (auto-send when SMTP configured) ────
    wsb_news = [n for n in analyzed_news if n.source == "WallStreetBets"]
    if wsb_news:
        try:
            from notifications.manager import NotificationManager
            sent = NotificationManager.send_wsb_email(analyzed_news, tickers)
            if sent:
                logger.info("WSB email report sent (%d mentions)", len(wsb_news))
            else:
                logger.info("WSB email skipped (SMTP not configured)")
        except Exception as exc:
            logger.warning("WSB email failed: %s", exc)
    
    return signals


def _save_results(system: AlgoTradingSystem, signals: List[Any], tickers: List[str]) -> Optional[str]:
    """
    Save analysis results to file.
    
    Args:
        system: AlgoTradingSystem instance
        signals: List of TradingSignal objects
        tickers: List of tickers analyzed
    
    Returns:
        Path where results were saved, or None
    """
    try:
        return system.storage_manager.save_signals(signals, append=Config.APPEND_MODE)
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None


def _save_to_database(signals: List[Any], tickers: List[str]) -> bool:
    """
    Save analysis results to database if available.
    
    Args:
        signals: List of TradingSignal objects
        tickers: List of tickers analyzed
    
    Returns:
        True if saved successfully, False otherwise
    """
    if not DB_AVAILABLE or not get_database_service:
        return False
    
    try:
        db_service = get_database_service()
        
        # Determine market from session state
        market = st.session_state.get('current_market', 'US')
        
        # Prepare data for database
        signal_data, news_data, fundamental_data = _prepare_database_data(signals)
        
        # Save to database
        run_id, counts = db_service.save_complete_analysis(
            tickers=tickers,
            signals=signal_data,
            news_items=news_data,
            fundamental_metrics=fundamental_data,
            parameters={'analysis_type': 'stock_analysis', 'market': market},
            run_type='stock_analysis',
            market=market,
        )
        
        if run_id:
            logger.info(f"Saved to database: {counts}")
            return True
        
    except Exception as e:
        logger.error(f"Database save failed: {e}")
    
    return False


def _prepare_database_data(signals: List[Any]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Prepare signal data for database storage.
    
    Args:
        signals: List of TradingSignal objects
    
    Returns:
        Tuple of (signal_data, news_data, fundamental_data)
    """
    signal_data = []
    news_data = []
    fundamental_data = []
    seen_tickers = set()
    
    for signal in signals:
        # Extract signal info
        signal_data.append({
            'ticker': signal.news_item.ticker,
            'decision': signal.decision.value,
            'confidence': abs(signal.decision_score) * 100,
            'strategy_name': 'ensemble_analysis',
            'reasons': signal.reasoning,
            'price_target': (
                getattr(signal.metrics, 'current_price', None)
                if signal.metrics else None
            ),
            'current_price': (
                getattr(signal.metrics, 'current_price', None)
                if signal.metrics else None
            ),
            'metadata': {
                'sentiment_score': signal.news_item.sentiment_score,
                'sentiment_label': (
                    signal.news_item.sentiment_label.value
                    if signal.news_item.sentiment_label else None
                ),
            }
        })
        
        # Extract news info
        news_data.append({
            'ticker': signal.news_item.ticker,
            'headline': signal.news_item.title,
            'summary': signal.news_item.summary or '',
            'source': signal.news_item.source,
            'url': signal.news_item.url or '',
            'published_at': signal.news_item.timestamp,
            'sentiment_score': signal.news_item.sentiment_score,
            'sentiment_label': (
                signal.news_item.sentiment_label.value
                if signal.news_item.sentiment_label else None
            ),
        })
        
        # Extract fundamental metrics (once per ticker)
        ticker = signal.news_item.ticker
        if ticker not in seen_tickers and signal.metrics:
            seen_tickers.add(ticker)
            fundamental_data.append(
                _extract_fundamental_metrics(ticker, signal.metrics)
            )
    
    return signal_data, news_data, fundamental_data


def _extract_fundamental_metrics(ticker: str, metrics: Any) -> Dict:
    """
    Extract fundamental metrics for database storage.
    
    Args:
        ticker: Stock ticker symbol
        metrics: StockMetrics object
    
    Returns:
        Dictionary of fundamental metrics
    """
    m = metrics
    
    # Calculate health score from fundamental scores
    health_score = None
    score_count = 0
    score_sum = 0
    
    if m.altman_z_score is not None:
        # Normalize Z-score: >2.99 = 100, <1.81 = 0
        z_normalized = min(100, max(0, (m.altman_z_score - 1.81) / (2.99 - 1.81) * 100))
        score_sum += z_normalized
        score_count += 1
    
    if m.piotroski_f_score is not None:
        # F-score is 0-9, normalize to 0-100
        f_normalized = (m.piotroski_f_score / 9) * 100
        score_sum += f_normalized
        score_count += 1
    
    if m.beneish_m_score is not None:
        # M-score: <-2.22 is good (100), >-2.22 is bad (0)
        m_normalized = 100 if m.beneish_m_score < -2.22 else 0
        score_sum += m_normalized
        score_count += 1
    
    if score_count > 0:
        health_score = score_sum / score_count
    
    return {
        'ticker': ticker,
        'pe_ratio': getattr(m, 'pe_ratio', None),
        'pb_ratio': getattr(m, 'pb_ratio', None),
        'market_cap': getattr(m, 'market_cap', None),
        'health_score': health_score,
        'data_source': 'yahoo_finance'
    }
