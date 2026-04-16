"""
Technical Analysis Layer.

Hybrid approach combining:
- Local advanced indicators via ``ta`` library on OHLCV data
- TradingView multi-timeframe consensus (26 oscillators + moving averages)
- Unified aggregator that fuses both into a single TA score
"""

from services.technical_analysis.aggregator import TechnicalAnalysisAggregator

__all__ = ["TechnicalAnalysisAggregator"]
