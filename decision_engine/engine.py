"""
Decision engine that combines sentiment, fundamentals, technicals,
macro-economic indicators, and public (Google search) sentiment.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from config import Config
from models import NewsItem, StockMetrics, TradingSignal, DecisionTag, SentimentLabel

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Combines fundamental, technical, and macro-economic
    analysis to generate trading decisions.

    Score composition (when all layers are available):
        fundamentals     40 %
        technicals       40 %
        macro-economic   20 %

    Regime-adaptive: Thresholds and position scaling adjust
    automatically based on the current market regime.
    """
    
    def __init__(self):
        """Initialize the decision engine."""
        self._macro_snapshot = None
        self._regime_snapshot = None

    def set_macro_snapshot(self, snapshot) -> None:
        """Inject a ``MacroSnapshot`` for the current analysis cycle."""
        self._macro_snapshot = snapshot

    def _get_regime(self):
        """Lazily fetch the current market regime."""
        if self._regime_snapshot is None:
            try:
                from services.regime_detector import regime_detector
                self._regime_snapshot = regime_detector.detect()
            except Exception:
                pass
        return self._regime_snapshot

    def generate_signal(
        self, 
        news_item: NewsItem, 
        metrics: Optional[StockMetrics]
    ) -> TradingSignal:
        """
        Generate a trading signal based on news and metrics.
        
        Args:
            news_item: NewsItem with sentiment analysis
            metrics: StockMetrics with fundamentals and technicals
            
        Returns:
            TradingSignal with decision and reasoning
        """
        # Calculate component scores
        fundamental_score = self._calculate_fundamental_score(metrics)
        technical_score = self._calculate_technical_score(metrics)
        macro_score = self._calculate_macro_score()

        # ── S3: Signal freshness gate ────────────────────────
        # Demote stale data: if metrics timestamp is more than N hours
        # old, decay confidence by pushing score toward HOLD.
        staleness_penalty = 0.0
        if metrics and metrics.timestamp:
            age = datetime.now() - metrics.timestamp
            max_hours = Config.SIGNAL_FRESHNESS_MAX_HOURS
            if age > timedelta(hours=max_hours):
                hours_stale = age.total_seconds() / 3600
                # Linear decay: 10% penalty per hour beyond threshold
                staleness_penalty = min(0.5, (hours_stale - max_hours) * 0.10)
                logger.info(
                    "%s: data is %.1fh old — applying %.0f%% staleness penalty",
                    news_item.ticker, hours_stale, staleness_penalty * 100,
                )

        # ── S7: Earnings blackout guard ──────────────────────
        # Suppress BUY signals near earnings announcements:
        # detected via news keyword matching on recent articles.
        earnings_blackout = False
        if news_item.category and news_item.category.value == "earnings":
            earnings_blackout = True
            logger.info(
                "%s: earnings-related news detected — blackout active",
                news_item.ticker,
            )

        # Dynamic weighting — redistribute if macro unavailable
        w_fund = Config.FUNDAMENTAL_WEIGHT
        w_tech = Config.TECHNICAL_WEIGHT
        w_macro = Config.MACRO_WEIGHT

        if macro_score is None:
            w_macro = 0.0

        total_w = w_fund + w_tech + w_macro
        if total_w > 0:
            w_fund /= total_w
            w_tech /= total_w
            w_macro /= total_w

        combined_score = (
            fundamental_score * w_fund
            + technical_score * w_tech
            + (macro_score or 0) * w_macro
        )

        # Apply staleness penalty (decays toward zero)
        if staleness_penalty > 0:
            combined_score *= (1.0 - staleness_penalty)

        # Earnings blackout: clamp positive scores to HOLD zone
        if earnings_blackout and combined_score > 0:
            combined_score = min(combined_score, Config.BUY_THRESHOLD - 0.01)

        # ── VIX regime gate (regime-adaptive thresholds) ────────
        regime = self._get_regime()
        vix_panic = regime.vix_panic if regime else Config.VIX_PANIC_THRESHOLD
        vix_caution = regime.vix_caution if regime else Config.VIX_CAUTION_THRESHOLD
        vix_scale = regime.position_scale if regime else Config.VIX_POSITION_SCALE
        buy_thresh = regime.buy_threshold if regime else Config.BUY_THRESHOLD

        if self._macro_snapshot is not None:
            vix_val = getattr(self._macro_snapshot, 'india_vix', None)
            if vix_val is None:
                vix_val = getattr(self._macro_snapshot, 'vix', None)
            if vix_val is not None:
                if vix_val >= vix_panic and combined_score > 0:
                    combined_score = min(combined_score, buy_thresh - 0.01)
                    logger.info(
                        "%s: VIX=%.1f (panic, threshold=%.1f) — suppressing BUY signal",
                        news_item.ticker, vix_val, vix_panic,
                    )
                elif vix_val >= vix_caution and combined_score > 0:
                    combined_score *= vix_scale
                    logger.info(
                        "%s: VIX=%.1f (caution, threshold=%.1f) — scaling BUY signal by %.0f%%",
                        news_item.ticker, vix_val, vix_caution, vix_scale * 100,
                    )

        # ── FII/DII gating (#10) ──────────────────────────────
        try:
            from scrapers.macro.fii_dii_tracker import compute_fii_dii_signal
            fii_signal = compute_fii_dii_signal()
            if fii_signal.is_heavy_fii_selling and combined_score > 0:
                combined_score = min(combined_score, buy_thresh - 0.01)
                logger.info(
                    "%s: Heavy FII selling (%d consecutive days) — suppressing BUY",
                    news_item.ticker, fii_signal.consecutive_fii_selling_days,
                )
            elif fii_signal.is_fii_selling_pressure and combined_score > 0:
                combined_score *= 0.7
                logger.info(
                    "%s: FII selling pressure — scaling BUY by 70%%",
                    news_item.ticker,
                )
        except Exception:
            pass  # FII data unavailable — degrade gracefully
        
        # Determine decision
        decision = self._score_to_decision(combined_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            news_item,
            metrics,
            fundamental_score,
            technical_score,
            combined_score,
            macro_score=macro_score,
        )
        
        # Create signal
        signal = TradingSignal(
            news_item=news_item,
            metrics=metrics,
            decision=decision,
            decision_score=combined_score,
            reasoning=reasoning
        )
        
        return signal
    
    def _calculate_sentiment_score(self, news_item: NewsItem) -> float:
        """
        Calculate normalized sentiment score (-1 to 1).
        
        Returns:
            Score from -1 (very negative) to 1 (very positive)
        """
        if news_item.sentiment_score is None:
            return 0.0
        
        # Gate low-confidence sentiment — treat as neutral
        if (news_item.sentiment_confidence is not None
                and news_item.sentiment_confidence < Config.SENTIMENT_CONFIDENCE_FLOOR):
            return 0.0

        # Boost score if high confidence
        score = news_item.sentiment_score
        if news_item.sentiment_confidence and news_item.sentiment_confidence > Config.SENTIMENT_HIGH_CONFIDENCE_THRESHOLD:
            score *= 1.2  # 20% boost for high confidence
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _calculate_fundamental_score(self, metrics: Optional[StockMetrics]) -> float:
        """
        Calculate fundamental score (-1 to 1).
        
        Considers: PEG ratio, ROE, EPS, intrinsic value vs current price
        """
        if not metrics:
            return 0.0
        
        score = 0.0
        count = 0
        
        # PEG Ratio (lower is better, < 1 is good)
        if metrics.peg_ratio is not None:
            if metrics.peg_ratio < 1:
                score += 0.5
            elif metrics.peg_ratio < 2:
                score += 0.2
            elif metrics.peg_ratio > 3:
                score -= 0.3
            count += 1
        
        # ROE (higher is better, > 15% is good)
        if metrics.roe is not None:
            if metrics.roe > 20:
                score += 0.4
            elif metrics.roe > 15:
                score += 0.2
            elif metrics.roe < 10:
                score -= 0.2
            count += 1
        
        # EPS (positive is good)
        if metrics.eps is not None:
            if metrics.eps > 5:
                score += 0.3
            elif metrics.eps > 0:
                score += 0.1
            else:
                score -= 0.3
            count += 1
        
        # Intrinsic Value vs Current Price
        if (
            metrics.intrinsic_value is not None 
            and metrics.current_price is not None 
            and metrics.current_price > 0
        ):
            value_ratio = metrics.intrinsic_value / metrics.current_price
            if value_ratio > 1.2:  # Undervalued by 20%+
                score += 0.5
            elif value_ratio > 1.0:  # Undervalued
                score += 0.3
            elif value_ratio < 0.8:  # Overvalued by 20%+
                score -= 0.5
            elif value_ratio < 1.0:  # Overvalued
                score -= 0.3
            count += 1

        # Piotroski F-Score (0–9; higher = healthier)
        if metrics.piotroski_f_score is not None:
            if metrics.piotroski_f_score >= 7:
                score += 0.4   # Strong financial health
            elif metrics.piotroski_f_score >= 5:
                score += 0.1
            elif metrics.piotroski_f_score <= 2:
                score -= 0.4   # Weak / distressed
            count += 1

        # Beneish M-Score (< -1.78 = unlikely fraud; > -1.78 = red flag)
        if metrics.beneish_m_score is not None:
            if metrics.beneish_m_score > -1.78:
                score -= 0.5   # Likely earnings manipulation
            elif metrics.beneish_m_score < -2.5:
                score += 0.2   # Clean financials
            count += 1

        # Altman Z-Score (> 2.99 = safe; 1.81–2.99 = grey; < 1.81 = distress)
        if metrics.altman_z_score is not None:
            if metrics.altman_z_score > 2.99:
                score += 0.3   # Minimal bankruptcy risk
            elif metrics.altman_z_score < 1.81:
                score -= 0.5   # High distress / bankruptcy risk
            count += 1
        
        # Average the score
        if count > 0:
            score = score / count
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _calculate_technical_score(self, metrics: Optional[StockMetrics]) -> float:
        """
        Calculate technical score (-1 to 1).
        
        Considers: RSI, MACD, Bollinger Bands, drawdown
        """
        if not metrics:
            return 0.0
        
        score = 0.0
        count = 0
        
        # RSI (< 30 oversold, > 70 overbought)
        if metrics.rsi is not None:
            if metrics.rsi < 30:
                score += 0.5  # Oversold - potential buy
            elif metrics.rsi < 40:
                score += 0.2
            elif metrics.rsi > 70:
                score -= 0.5  # Overbought - potential sell
            elif metrics.rsi > 60:
                score -= 0.2
            count += 1
        
        # MACD (histogram magnitude → proportional score)
        if metrics.macd_histogram is not None:
            if metrics.current_price and metrics.current_price > 0:
                # Normalise histogram by price so large-cap and small-cap
                # stocks get comparable scores.
                norm_hist = metrics.macd_histogram / metrics.current_price * 100
                macd_score = max(-0.5, min(0.5, norm_hist * 0.15))
            else:
                macd_score = 0.3 if metrics.macd_histogram > 0 else -0.3
            score += macd_score
            count += 1
        
        # Bollinger Bands (price near lower band = buy, upper band = sell)
        if (
            metrics.current_price is not None 
            and metrics.bollinger_upper is not None 
            and metrics.bollinger_lower is not None
            and metrics.bollinger_middle is not None
        ):
            band_range = metrics.bollinger_upper - metrics.bollinger_lower
            if band_range > 0:
                position = (metrics.current_price - metrics.bollinger_lower) / band_range
                
                if position < 0.2:  # Near lower band
                    score += 0.4
                elif position < 0.4:
                    score += 0.2
                elif position > 0.8:  # Near upper band
                    score -= 0.4
                elif position > 0.6:
                    score -= 0.2
                count += 1
        
        # Maximum Drawdown (large drawdown = risky)
        if metrics.max_drawdown is not None:
            if metrics.max_drawdown < -30:  # > 30% drawdown
                score -= 0.3
            elif metrics.max_drawdown < -20:
                score -= 0.1
            count += 1

        # ── S1: ADX regime detection ─────────────────────────
        # If ADX is below the trend threshold, the market is choppy
        # and momentum signals (RSI, MACD) are less reliable → dampen.
        adx_dampening = 1.0
        if metrics.adx is not None:
            if metrics.adx < Config.ADX_TREND_THRESHOLD:
                # In range-bound markets, halve the technical signal
                adx_dampening = 0.5
            count += 1

        # ── S5: OBV volume confirmation ──────────────────────
        # Bullish divergence (price up, OBV down) weakens BUY signal.
        # Bearish divergence (price down, OBV up) weakens SELL signal.
        obv_adjustment = 0.0
        if metrics.obv is not None and metrics.obv_sma is not None:
            obv_rising = metrics.obv > metrics.obv_sma
            price_bullish = score > 0
            if price_bullish and obv_rising:
                obv_adjustment = 0.15    # volume confirms bullish move
            elif price_bullish and not obv_rising:
                obv_adjustment = -0.15   # bearish divergence warning
            elif not price_bullish and not obv_rising:
                obv_adjustment = -0.10   # volume confirms bearish move
            elif not price_bullish and obv_rising:
                obv_adjustment = 0.10    # bullish divergence hint
            count += 1
        
        # Average the score
        if count > 0:
            score = score / count

        # Apply ADX regime dampening (choppy market → weaker signal)
        score *= adx_dampening
        # Apply OBV volume confirmation/divergence
        score += obv_adjustment
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _score_to_decision(self, score: float) -> DecisionTag:
        """Convert combined score to decision tag.

        Uses regime-adaptive thresholds when available.
        """
        regime = self._get_regime()
        if regime:
            sb = regime.strong_buy_threshold
            b = regime.buy_threshold
            s = regime.sell_threshold
            ss = regime.strong_sell_threshold
        else:
            sb = Config.STRONG_BUY_THRESHOLD
            b = Config.BUY_THRESHOLD
            s = Config.SELL_THRESHOLD
            ss = Config.STRONG_SELL_THRESHOLD

        if score >= sb:
            return DecisionTag.STRONG_BUY
        elif score >= b:
            return DecisionTag.BUY
        elif score <= ss:
            return DecisionTag.STRONG_SELL
        elif score <= s:
            return DecisionTag.SELL
        else:
            return DecisionTag.HOLD

    # ── Macro-economic scoring ───────────────────────────────────────

    def _calculate_macro_score(self) -> Optional[float]:
        """
        Return the pre-computed macro sentiment score from the snapshot,
        or ``None`` if unavailable.
        """
        snap = self._macro_snapshot
        if snap is None:
            return None
        if snap.macro_sentiment_score is not None:
            return max(-1.0, min(1.0, snap.macro_sentiment_score))
        return None


    # ── Reasoning ────────────────────────────────────────────────────
    
    def _generate_reasoning(
        self,
        news_item: NewsItem,
        metrics: Optional[StockMetrics],
        fundamental_score: float,
        technical_score: float,
        combined_score: float,
        *,
        macro_score: Optional[float] = None,
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        reasons = []
        
        # Fundamentals
        if metrics:
            if metrics.peg_ratio and metrics.peg_ratio < 1:
                reasons.append(f"Strong PEG ratio ({metrics.peg_ratio:.2f})")
            if metrics.roe and metrics.roe > 15:
                reasons.append(f"Good ROE ({metrics.roe:.1f}%)")
            if metrics.intrinsic_value and metrics.current_price:
                ratio = metrics.intrinsic_value / metrics.current_price
                if ratio > 1.2:
                    reasons.append(f"Undervalued ({ratio:.1%} of intrinsic value)")
                elif ratio < 0.8:
                    reasons.append(f"Overvalued ({ratio:.1%} of intrinsic value)")
        
        # Technicals
        if metrics:
            if metrics.rsi:
                if metrics.rsi < 30:
                    reasons.append(f"Oversold RSI ({metrics.rsi:.1f})")
                elif metrics.rsi > 70:
                    reasons.append(f"Overbought RSI ({metrics.rsi:.1f})")
            if metrics.macd_histogram:
                if metrics.macd_histogram > 0:
                    reasons.append("Bullish MACD")
                else:
                    reasons.append("Bearish MACD")

        # Macro
        if macro_score is not None:
            snap = self._macro_snapshot
            label = getattr(snap, "macro_sentiment_label", None) or "n/a"
            reasons.append(f"Macro: {label} ({macro_score:+.2f})")
            if snap and snap.vix is not None:
                reasons.append(f"VIX={snap.vix:.1f}")
        
        # Combine
        reasoning = "; ".join(reasons) if reasons else "Based on available data"
        reasoning += f" | Combined score: {combined_score:.2f}"
        
        return reasoning
