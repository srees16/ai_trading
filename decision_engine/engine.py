"""
Decision engine that combines sentiment, fundamentals, and technicals.
"""

from typing import Optional
from models import NewsItem, StockMetrics, TradingSignal, DecisionTag, SentimentLabel
from config import Config


class DecisionEngine:
    """
    Combines sentiment, fundamental, and technical analysis to generate trading decisions.
    """
    
    def __init__(self):
        """Initialize the decision engine."""
        pass
    
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
        sentiment_score = self._calculate_sentiment_score(news_item)
        fundamental_score = self._calculate_fundamental_score(metrics)
        technical_score = self._calculate_technical_score(metrics)
        
        # Combine scores with weights
        combined_score = (
            sentiment_score * Config.SENTIMENT_WEIGHT +
            fundamental_score * Config.FUNDAMENTAL_WEIGHT +
            technical_score * Config.TECHNICAL_WEIGHT
        )
        
        # Determine decision
        decision = self._score_to_decision(combined_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            news_item,
            metrics,
            sentiment_score,
            fundamental_score,
            technical_score,
            combined_score
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
        
        # MACD (histogram positive = bullish)
        if metrics.macd_histogram is not None:
            if metrics.macd_histogram > 0:
                score += 0.3
            else:
                score -= 0.3
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
        
        # Average the score
        if count > 0:
            score = score / count
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _score_to_decision(self, score: float) -> DecisionTag:
        """Convert combined score to decision tag."""
        if score >= Config.STRONG_BUY_THRESHOLD:
            return DecisionTag.STRONG_BUY
        elif score >= Config.BUY_THRESHOLD:
            return DecisionTag.BUY
        elif score <= Config.STRONG_SELL_THRESHOLD:
            return DecisionTag.STRONG_SELL
        elif score <= Config.SELL_THRESHOLD:
            return DecisionTag.SELL
        else:
            return DecisionTag.HOLD
    
    def _generate_reasoning(
        self,
        news_item: NewsItem,
        metrics: Optional[StockMetrics],
        sentiment_score: float,
        fundamental_score: float,
        technical_score: float,
        combined_score: float
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        reasons = []
        
        # Sentiment
        if news_item.sentiment_label == SentimentLabel.POSITIVE:
            conf = news_item.sentiment_confidence or 0
            reasons.append(f"Positive news sentiment ({conf:.2%} confidence)")
        elif news_item.sentiment_label == SentimentLabel.NEGATIVE:
            conf = news_item.sentiment_confidence or 0
            reasons.append(f"Negative news sentiment ({conf:.2%} confidence)")
        
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
        
        # Combine
        reasoning = "; ".join(reasons) if reasons else "Based on available data"
        reasoning += f" | Combined score: {combined_score:.2f}"
        
        return reasoning
