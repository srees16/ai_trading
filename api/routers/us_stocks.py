"""
US Stocks API router.

Endpoints for stock analysis, news scraping, sentiment analysis,
metrics calculation, decision engine, strategy listing & backtesting.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_config, get_db_service, get_trading_system
from api.schemas.common import ErrorResponse, SuccessResponse
from api.schemas.us_stocks import (
    AnalysisHistoryResponse,
    AnalysisRequest,
    AnalysisResponse,
    BacktestRequest,
    BacktestResponse,
    DecisionRequest,
    DecisionResponse,
    MetricsRequest,
    MetricsResponse,
    ScrapeNewsRequest,
    ScrapeNewsResponse,
    SentimentRequest,
    SentimentResponse,
    SentimentResult,
    StockMetricsResponse,
    StrategyInfo,
    StrategyListResponse,
    NewsItemResponse,
    TradingSignalResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/us-stocks", tags=["US Stocks"])


# -----------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------

@router.post(
    "/analysis",
    response_model=AnalysisResponse,
    summary="Run full stock analysis pipeline",
    description="Scrapes news, analyses sentiment, calculates metrics, and generates trading signals.",
)
async def run_analysis(request: AnalysisRequest):
    """Execute the complete analysis pipeline for the given tickers."""
    try:
        from main import AlgoTradingSystem
        from scrapers.us_aggregator import USNewsAggregator
        from services.sentiment import SentimentAnalyzer
        from services.metrics import MetricsCalculator
        from services.decision_engine import DecisionEngine
        from models import TradingSignal

        system = AlgoTradingSystem(tickers=request.tickers)

        # Step 1: Scrape news
        all_news = await system.news_aggregator.fetch_news_for_tickers(request.tickers)
        if not all_news:
            return AnalysisResponse(
                success=True,
                ticker_count=len(request.tickers),
                signal_count=0,
                signals=[],
            )

        # Step 2: Sentiment
        analyzed_news = system.sentiment_analyzer.analyze_news_items(all_news)

        # Step 3: Metrics + signals
        system.metrics_calculator.prefetch_metrics(
            list({n.ticker for n in analyzed_news})
        )

        signals: list[TradingSignal] = []
        for news_item in analyzed_news:
            metrics = system.metrics_calculator.get_stock_metrics(news_item.ticker)
            signal = system.decision_engine.generate_signal(news_item, metrics)
            signals.append(signal)

        # Step 4: Persist to DB (best-effort)
        try:
            from database.service import get_database_service
            db = get_database_service()
            if db and db.is_available:
                signal_data = [s.to_dict() for s in signals]
                db.save_complete_analysis(
                    tickers=request.tickers,
                    signals=signal_data,
                    news_items=[],
                    fundamental_metrics=[],
                    parameters={"analysis_type": "api_stock_analysis"},
                    run_type="api_stock_analysis",
                )
        except Exception as exc:
            logger.warning("DB save failed: %s", exc)

        # Build response
        signal_responses = [
            TradingSignalResponse(**s.to_dict()) for s in signals
        ]

        # Carver enrichment: attach vol-targeted trade plans when enabled
        carver_plans = []
        try:
            from config import Config
            if getattr(Config, "CARVER_US_ENABLED", False):
                from services.us_carver_pipeline import run_us_carver_pipeline
                carver_result = run_us_carver_pipeline(request.tickers)
                carver_plans = carver_result.trade_plans
                logger.info("Carver US enrichment: %d trade plans for %d tickers",
                            len(carver_plans), len(request.tickers))
        except Exception as exc:
            logger.debug("Carver US enrichment skipped: %s", exc)

        resp = AnalysisResponse(
            success=True,
            ticker_count=len(request.tickers),
            signal_count=len(signals),
            signals=signal_responses,
        )
        # Attach Carver plans as extra field (won't break schema)
        resp.__dict__["carver_trade_plans"] = carver_plans
        return resp

    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# News Scraping
# -----------------------------------------------------------------------

@router.post(
    "/news",
    response_model=ScrapeNewsResponse,
    summary="Scrape news for tickers",
)
async def scrape_news(request: ScrapeNewsRequest):
    """Scrape news articles from multiple sources for the given tickers."""
    try:
        from scrapers.us_aggregator import USNewsAggregator

        aggregator = USNewsAggregator()
        items = await aggregator.fetch_news_for_tickers(request.tickers)

        return ScrapeNewsResponse(
            success=True,
            total=len(items),
            items=[
                NewsItemResponse(
                    title=n.title,
                    summary=n.summary,
                    url=n.url,
                    timestamp=n.timestamp,
                    source=n.source,
                    ticker=n.ticker,
                    category=n.category.value if hasattr(n.category, "value") else str(n.category),
                    sentiment_score=n.sentiment_score,
                    sentiment_label=n.sentiment_label.value if n.sentiment_label and hasattr(n.sentiment_label, "value") else None,
                    sentiment_confidence=n.sentiment_confidence,
                )
                for n in items
            ],
        )
    except Exception as exc:
        logger.exception("News scraping failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Sentiment Analysis
# -----------------------------------------------------------------------

@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Analyse sentiment of text snippets",
)
async def analyse_sentiment(request: SentimentRequest):
    """Run sentiment analysis on arbitrary text snippets."""
    try:
        from services.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        results = []
        for text in request.texts:
            result = analyzer.analyze_text(text)
            results.append(
                SentimentResult(
                    text=text,
                    label=result.get("label", "neutral"),
                    score=result.get("score", 0.0),
                    confidence=result.get("confidence", 0.0),
                )
            )

        return SentimentResponse(success=True, results=results)
    except Exception as exc:
        logger.exception("Sentiment analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

@router.post(
    "/metrics",
    response_model=MetricsResponse,
    summary="Calculate stock metrics",
)
async def calculate_metrics(request: MetricsRequest):
    """Calculate fundamental and technical metrics for the given tickers."""
    try:
        from services.metrics import MetricsCalculator

        calc = MetricsCalculator()
        calc.prefetch_metrics(request.tickers)

        metrics_list = []
        for ticker in request.tickers:
            m = calc.get_stock_metrics(ticker)
            if m:
                metrics_list.append(
                    StockMetricsResponse(
                        ticker=m.ticker,
                        timestamp=m.timestamp,
                        peg_ratio=m.peg_ratio,
                        roe=m.roe,
                        eps=m.eps,
                        free_cash_flow=m.free_cash_flow,
                        dcf_value=m.dcf_value,
                        intrinsic_value=m.intrinsic_value,
                        altman_z_score=m.altman_z_score,
                        beneish_m_score=m.beneish_m_score,
                        piotroski_f_score=m.piotroski_f_score,
                        rsi=m.rsi,
                        macd=m.macd,
                        macd_signal=m.macd_signal,
                        macd_histogram=m.macd_histogram,
                        bollinger_upper=m.bollinger_upper,
                        bollinger_middle=m.bollinger_middle,
                        bollinger_lower=m.bollinger_lower,
                        max_drawdown=m.max_drawdown,
                        current_price=m.current_price,
                    )
                )

        return MetricsResponse(success=True, metrics=metrics_list)
    except Exception as exc:
        logger.exception("Metrics calculation failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Decision Engine
# -----------------------------------------------------------------------

@router.post(
    "/decision",
    response_model=DecisionResponse,
    summary="Generate a trading decision",
)
async def generate_decision(request: DecisionRequest):
    """Generate a BUY/SELL/HOLD decision from news + metrics."""
    try:
        from services.decision_engine import DecisionEngine
        from services.metrics import MetricsCalculator
        from models import NewsItem, SentimentLabel, NewsCategory

        news = NewsItem(
            title=request.news_title,
            summary=request.news_summary,
            url=request.news_url,
            timestamp=datetime.utcnow(),
            source=request.news_source,
            ticker=request.ticker,
            category=NewsCategory.GENERAL,
            sentiment_score=request.sentiment_score,
            sentiment_label=(
                SentimentLabel(request.sentiment_label)
                if request.sentiment_label
                else None
            ),
            sentiment_confidence=request.sentiment_confidence,
        )

        calc = MetricsCalculator()
        metrics = calc.get_stock_metrics(request.ticker)

        engine = DecisionEngine()
        signal = engine.generate_signal(news, metrics)

        return DecisionResponse(
            success=True,
            ticker=request.ticker,
            decision=signal.decision.value,
            decision_score=signal.decision_score,
            reasoning=signal.reasoning,
            sentiment_score=engine._calculate_sentiment_score(news),
            fundamental_score=engine._calculate_fundamental_score(metrics),
            technical_score=engine._calculate_technical_score(metrics),
        )
    except Exception as exc:
        logger.exception("Decision generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------

@router.get(
    "/strategies",
    response_model=StrategyListResponse,
    summary="List available trading strategies",
)
async def list_strategies():
    """Return metadata for all registered backtesting strategies."""
    try:
        from trading_strategies import list_strategies as _list_strategies

        meta = _list_strategies()
        items = [
            StrategyInfo(
                id=s["id"],
                name=s["name"],
                description=s["description"],
                category=s["category"],
                requires_sentiment=s.get("requires_sentiment", False),
                min_tickers=s.get("min_tickers", 1),
            )
            for s in meta
        ]
        return StrategyListResponse(
            success=True, count=len(items), strategies=items
        )
    except Exception as exc:
        logger.exception("Strategy listing failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/strategies/{strategy_id}",
    response_model=StrategyInfo,
    summary="Get strategy details",
)
async def get_strategy_info(strategy_id: str):
    """Return metadata for a single strategy by ID."""
    from trading_strategies import list_strategies as _list_strategies

    for s in _list_strategies():
        if s["id"] == strategy_id:
            return StrategyInfo(
                id=s["id"],
                name=s["name"],
                description=s["description"],
                category=s["category"],
                requires_sentiment=s.get("requires_sentiment", False),
                min_tickers=s.get("min_tickers", 1),
            )
    raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")


# -----------------------------------------------------------------------
# Backtesting
# -----------------------------------------------------------------------

@router.post(
    "/backtest",
    response_model=BacktestResponse,
    summary="Run a strategy backtest",
)
async def run_backtest(request: BacktestRequest):
    """Execute a backtest for the given strategy and tickers."""
    try:
        from trading_strategies import get_strategy

        strategy_cls = get_strategy(request.strategy_id)
        if strategy_cls is None:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy '{request.strategy_id}' not found",
            )

        strategy = strategy_cls()

        # Build run kwargs
        run_kwargs: Dict[str, Any] = {
            "tickers": request.tickers,
        }
        if request.start_date:
            run_kwargs["start_date"] = request.start_date
        if request.end_date:
            run_kwargs["end_date"] = request.end_date
        if request.parameters:
            run_kwargs["parameters"] = request.parameters
        if request.initial_capital:
            run_kwargs["initial_capital"] = request.initial_capital

        result = strategy.run(**run_kwargs)

        result_dict = result.to_dict() if hasattr(result, "to_dict") else {}

        return BacktestResponse(
            success=result_dict.get("success", True),
            strategy_id=request.strategy_id,
            tickers=request.tickers,
            metrics=result_dict.get("metrics", {}),
            charts=result_dict.get("charts", []),
            tables=result_dict.get("tables", []),
            execution_time=result_dict.get("execution_time", 0.0),
            error_message=result_dict.get("error_message", ""),
            metadata=result_dict.get("metadata", {}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Backtest failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Analysis History (DB)
# -----------------------------------------------------------------------

@router.get(
    "/history",
    response_model=AnalysisHistoryResponse,
    summary="Get past analysis runs",
)
async def get_analysis_history(
    limit: int = 20,
    offset: int = 0,
):
    """Retrieve historical analysis runs from the database."""
    try:
        from api.dependencies import get_db_service

        db = get_db_service()
        if db is None:
            return AnalysisHistoryResponse(success=True, runs=[], total=0)

        runs = db.get_analysis_runs(limit=limit, offset=offset)
        total = db.count_analysis_runs() if hasattr(db, "count_analysis_runs") else len(runs)

        return AnalysisHistoryResponse(
            success=True,
            runs=[r if isinstance(r, dict) else r.__dict__ for r in runs],
            total=total,
        )
    except Exception as exc:
        logger.exception("History retrieval failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Carver Systematic Trading Framework — US Stocks
# -----------------------------------------------------------------------

@router.get(
    "/carver/status",
    summary="Carver framework status for US stocks",
)
async def us_carver_status():
    """Return Carver framework configuration and module availability for US stocks."""
    try:
        from config import Config
        enabled = getattr(Config, "CARVER_US_ENABLED", False)
        return {
            "carver_us_enabled": enabled,
            "annual_vol_target": getattr(Config, "CARVER_US_ANNUAL_VOL_TARGET", 0.20),
            "initial_capital_usd": getattr(Config, "CARVER_US_INITIAL_CAPITAL", 10_000.0),
            "default_idm": getattr(Config, "CARVER_US_DEFAULT_IDM", 1.5),
            "max_leverage": getattr(Config, "CARVER_US_MAX_LEVERAGE", 1.0),
            "cost_round_trip_pct": getattr(Config, "CARVER_US_COST_ROUND_TRIP_PCT", 0.0010),
            "spread_slippage_pct": getattr(Config, "CARVER_US_SPREAD_SLIPPAGE_PCT", 0.0005),
            "trade_horizon": getattr(Config, "CARVER_TRADE_HORIZON", "swing"),
            "modules_available": _check_us_carver_modules(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/carver/pipeline",
    summary="Run Carver pipeline on US stocks",
)
async def us_carver_pipeline(tickers: Optional[List[str]] = None):
    """Execute the full Carver systematic trading pipeline for US stocks.

    Returns vol-targeted trade plans with EWMAC forecasts, cost filtering,
    and position sizing.
    """
    try:
        from config import Config
        if not getattr(Config, "CARVER_US_ENABLED", False):
            raise HTTPException(status_code=400, detail="Carver US is not enabled in config")

        from services.us_carver_pipeline import run_us_carver_pipeline, DEFAULT_US_CARVER_TICKERS

        syms = tickers or DEFAULT_US_CARVER_TICKERS
        result = await asyncio.to_thread(run_us_carver_pipeline, syms)

        return {
            "success": True,
            "trade_plans": result.trade_plans,
            "symbols_processed": result.symbols_processed,
            "symbols_filtered_by_cost": result.symbols_filtered_by_cost,
            "combined_forecasts": result.combined_forecasts,
            "idm": result.idm,
            "pipeline_log": result.pipeline_log,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("US Carver pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/carver/efficiency",
    summary="Run Carver calibration backtest on US stocks",
)
async def us_carver_efficiency(tickers: Optional[List[str]] = None):
    """Run an expanding-window backtest and return before vs after metrics.

    Computes Sharpe, Sortino, MaxDD, and annual return using the Carver
    vol-targeted framework on US historical data.
    """
    try:
        from services.us_carver_pipeline import run_us_carver_backtest

        report = await asyncio.to_thread(run_us_carver_backtest, tickers)
        if "error" in report:
            raise HTTPException(status_code=400, detail=report["error"])
        return report
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("US Carver efficiency report failed")
        raise HTTPException(status_code=500, detail=str(exc))


def _check_us_carver_modules() -> Dict[str, bool]:
    """Check which Carver modules are importable for US stocks."""
    modules = {
        "us_carver_pipeline": "services.us_carver_pipeline",
        "instrument_volatility": "services.instrument_volatility",
        "volatility_target": "services.volatility_target",
        "forecast_scalar": "services.forecast_scalar",
        "forecast_combiner": "services.forecast_combiner",
        "position_sizer": "services.position_sizer",
        "instrument_weights": "services.instrument_weights",
        "ewmac": "strategies.ewmac",
        "cost_speed_limit": "services.cost_speed_limit",
        "carver_calibration": "services.carver_calibration",
    }
    status = {}
    for name, mod_path in modules.items():
        try:
            __import__(mod_path)
            status[name] = True
        except Exception:
            status[name] = False
    return status
