"""
Stock Analysis Pipeline — Structured multi-stage evaluation.

Implements the institutional-grade pipeline:

    Raw Data → Clean → Feature Engineering → Alpha Signals →
    Combination → Portfolio Optimization → Execution → Post-Trade Analysis

Each stage is a standalone callable that can be tested, measured, and
replaced independently.  The pipeline emits events at each stage
boundary for monitoring and replay.

Usage::

    from infrastructure.analysis_pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline(market="IND")
    result = pipeline.run(tickers=["RELIANCE.NS", "TCS.NS"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageResult:
    """Output of a single pipeline stage."""
    stage: str
    data: Any
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Aggregated output across all pipeline stages."""
    tickers: List[str]
    market: str
    stages: Dict[str, PipelineStageResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_success(self) -> bool:
        return all(s.success for s in self.stages.values())


class AnalysisPipeline:
    """
    Orchestrates the multi-stage stock analysis pipeline.

    Stages (executed in order):
        1. raw_data      — Fetch OHLCV, fundamentals, news
        2. clean          — Fill gaps, handle holidays/circuits
        3. features       — Technical indicators, ML features
        4. alpha_signals  — Generate buy/sell/hold signals
        5. combination    — Multi-layer signal aggregation
        6. portfolio_opt  — Position sizing, risk allocation
        7. execution      — Order generation (live or paper)
        8. post_trade     — Performance attribution, logging
    """

    STAGES = [
        "raw_data",
        "clean",
        "features",
        "alpha_signals",
        "combination",
        "portfolio_opt",
        "execution",
        "post_trade",
    ]

    def __init__(
        self,
        market: str = "US",
        *,
        skip_stages: Optional[List[str]] = None,
    ):
        self.market = market
        self.skip_stages = set(skip_stages or [])

    def run(
        self,
        tickers: List[str],
        *,
        date_range: Optional[tuple] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full pipeline, emitting events at each stage."""
        from infrastructure.event_bus import event_bus

        result = PipelineResult(tickers=tickers, market=self.market)
        ctx = context or {}
        ctx["tickers"] = tickers
        ctx["market"] = self.market
        ctx["date_range"] = date_range

        pipeline_start = time.monotonic()

        for stage_name in self.STAGES:
            if stage_name in self.skip_stages:
                continue

            t0 = time.monotonic()
            try:
                handler = getattr(self, f"_stage_{stage_name}", None)
                if handler is None:
                    continue
                stage_data = handler(ctx)
                latency = (time.monotonic() - t0) * 1000
                sr = PipelineStageResult(
                    stage=stage_name, data=stage_data, latency_ms=latency,
                )
                ctx[stage_name] = stage_data
            except Exception as exc:
                latency = (time.monotonic() - t0) * 1000
                sr = PipelineStageResult(
                    stage=stage_name, data=None, latency_ms=latency,
                    success=False, error=str(exc),
                )
                logger.error("Pipeline stage '%s' failed: %s", stage_name, exc)

            result.stages[stage_name] = sr

            # Emit event for monitoring
            event_bus.emit(
                f"pipeline.stage.{stage_name}",
                payload={
                    "market": self.market,
                    "tickers": tickers,
                    "latency_ms": sr.latency_ms,
                    "success": sr.success,
                },
                source="analysis_pipeline",
            )

        result.total_latency_ms = (time.monotonic() - pipeline_start) * 1000
        event_bus.emit(
            "pipeline.complete",
            payload={
                "market": self.market,
                "tickers": tickers,
                "total_latency_ms": result.total_latency_ms,
                "success": result.is_success,
            },
            source="analysis_pipeline",
        )
        return result

    # ── Stage implementations ────────────────────────────────────
    # Each stage receives the running context dict and returns data
    # that gets stored in ctx[stage_name].

    def _stage_raw_data(self, ctx: dict) -> dict:
        """Fetch raw OHLCV + fundamental + news data."""
        import yfinance as yf

        tickers = ctx["tickers"]
        date_range = ctx.get("date_range")

        start = date_range[0] if date_range else None
        end = date_range[1] if date_range and len(date_range) > 1 else None

        ohlcv = {}
        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(start=start, end=end, period="1y")
                if not hist.empty:
                    ohlcv[ticker] = hist
            except Exception as exc:
                logger.warning("Raw data fetch failed for %s: %s", ticker, exc)

        return {"ohlcv": ohlcv}

    def _stage_clean(self, ctx: dict) -> dict:
        """Clean data — fill gaps, remove holiday artifacts."""
        raw = ctx.get("raw_data", {})
        ohlcv = raw.get("ohlcv", {})

        cleaned = {}
        for ticker, df in ohlcv.items():
            clean_df = df.copy()
            # Forward-fill small gaps (weekends/holidays already handled by yfinance)
            clean_df = clean_df.ffill().bfill()
            # Remove zero-volume rows (likely data errors)
            if "Volume" in clean_df.columns:
                clean_df = clean_df[clean_df["Volume"] > 0]
            cleaned[ticker] = clean_df

        return {"ohlcv": cleaned}

    def _stage_features(self, ctx: dict) -> dict:
        """Compute technical indicators and ML features."""
        import numpy as np
        import pandas as pd

        clean = ctx.get("clean", {})
        ohlcv = clean.get("ohlcv", {})

        features = {}
        for ticker, df in ohlcv.items():
            if df.empty or len(df) < 30:
                continue
            feat = {}
            close = df["Close"]

            # RSI (14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            feat["rsi"] = float((100 - 100 / (1 + rs)).iloc[-1])

            # MACD (12/26/9)
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            feat["macd"] = float(macd.iloc[-1])
            feat["macd_signal"] = float(signal.iloc[-1])
            feat["macd_histogram"] = float((macd - signal).iloc[-1])

            # Bollinger Bands (20, 2)
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            feat["bb_upper"] = float((sma20 + 2 * std20).iloc[-1])
            feat["bb_lower"] = float((sma20 - 2 * std20).iloc[-1])
            feat["bb_position"] = float(
                (close.iloc[-1] - feat["bb_lower"])
                / max(feat["bb_upper"] - feat["bb_lower"], 0.01)
            )

            # SMAs
            feat["sma_50"] = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            feat["sma_200"] = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

            # Volatility (20-day)
            feat["volatility_20d"] = float(close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5))

            # Max drawdown
            cummax = close.cummax()
            drawdown = (close - cummax) / cummax
            feat["max_drawdown"] = float(drawdown.min())

            feat["current_price"] = float(close.iloc[-1])
            feat["daily_return"] = float(close.pct_change().iloc[-1])

            features[ticker] = feat

        return features

    def _stage_alpha_signals(self, ctx: dict) -> dict:
        """Generate per-ticker alpha signals from features."""
        features = ctx.get("features", {})

        signals = {}
        for ticker, feat in features.items():
            score = 0.0
            reasons = []

            # RSI
            rsi = feat.get("rsi")
            if rsi is not None:
                if rsi < 30:
                    score += 0.4
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    score -= 0.4
                    reasons.append(f"RSI overbought ({rsi:.1f})")

            # MACD histogram
            hist = feat.get("macd_histogram")
            if hist is not None:
                if hist > 0:
                    score += 0.2
                    reasons.append("MACD bullish")
                else:
                    score -= 0.2
                    reasons.append("MACD bearish")

            # Bollinger position
            bb_pos = feat.get("bb_position")
            if bb_pos is not None:
                if bb_pos < 0.2:
                    score += 0.3
                    reasons.append("Near BB lower band")
                elif bb_pos > 0.8:
                    score -= 0.3
                    reasons.append("Near BB upper band")

            # Trend (price vs SMA200)
            price = feat.get("current_price", 0)
            sma200 = feat.get("sma_200")
            if sma200 and price > sma200:
                score += 0.1
                reasons.append("Above SMA-200")
            elif sma200 and price < sma200:
                score -= 0.1
                reasons.append("Below SMA-200")

            score = max(-1.0, min(1.0, score))
            signals[ticker] = {
                "technical_score": score,
                "reasons": reasons,
                "features": feat,
            }

        return signals

    def _stage_combination(self, ctx: dict) -> dict:
        """Combine technical signals with any available fundamental/sentiment data."""
        alpha = ctx.get("alpha_signals", {})
        # For now, pass through technical alpha as primary signal
        # When integrated_scorer layers are available, they feed in here
        return alpha

    def _stage_portfolio_opt(self, ctx: dict) -> dict:
        """Basic position sizing — equal-weight for now."""
        combined = ctx.get("combination", {})
        tickers = [t for t, s in combined.items() if s.get("technical_score", 0) > 0.2]
        n = max(len(tickers), 1)
        return {
            "allocations": {t: 1.0 / n for t in tickers},
            "selected_tickers": tickers,
        }

    def _stage_execution(self, ctx: dict) -> dict:
        """Generate order intents (does not place real orders)."""
        from infrastructure.execution_context import execution_ctx

        portfolio = ctx.get("portfolio_opt", {})
        allocations = portfolio.get("allocations", {})
        combined = ctx.get("combination", {})

        orders = []
        for ticker, weight in allocations.items():
            signal = combined.get(ticker, {})
            score = signal.get("technical_score", 0)
            side = "BUY" if score > 0 else "SELL"
            orders.append({
                "ticker": ticker,
                "side": side,
                "weight": weight,
                "score": score,
                "mode": execution_ctx.mode,
            })

        return {"orders": orders}

    def _stage_post_trade(self, ctx: dict) -> dict:
        """Log pipeline execution for audit."""
        execution = ctx.get("execution", {})
        return {
            "orders_generated": len(execution.get("orders", [])),
            "pipeline_success": True,
        }
