"""
Alpha Research Layer — Signal generation.

Coordinates all alpha-producing components:
  - Technical indicators (metrics/calculator)
  - Strategy backtests (strategies/ registry)
  - ML features (financial_ML/applied/)
  - Sentiment (news + DistilBERT/FinBERT)
  - Macro indicators
  - Broader public sentiment

Emits ``alpha.signal`` events for each ticker evaluated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlphaResearchService:
    """
    Orchestrates signal generation across all alpha sources.

    Re-uses existing modules (DecisionEngine, IntegratedScorer)
    but wraps them in a clean interface.
    """

    def __init__(self, market: str = "US"):
        self.market = market

    def generate_signals(
        self,
        tickers: List[str],
        *,
        date_range: Optional[tuple] = None,
        skip_layers: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """
        Run multi-layer analysis and return signal dicts.

        Delegates to IntegratedScorer for the heavy lifting.
        """
        from infrastructure.event_bus import event_bus
        from infrastructure.latency_tracker import latency_tracker
        from services.integrated_scorer import IntegratedScorer

        default_weights = weights or {
            "core": 0.35,
            "strategy": 0.25,
            "ml_features": 0.15,
            "robustness": 0.25,
            "rag": 0.00,
        }

        scorer = IntegratedScorer(weights=default_weights)

        with latency_tracker.measure("alpha.generate_signals"):
            verdicts = scorer.evaluate(
                tickers=tickers,
                market=self.market,
                date_range=date_range,
                skip_layers=skip_layers or ["rag"],
            )

        signals = []
        for v in verdicts:
            sig = {
                "ticker": v.ticker if hasattr(v, "ticker") else str(v),
                "verdict": v,
            }
            signals.append(sig)
            event_bus.emit(
                "alpha.signal",
                payload=sig,
                source="alpha_research",
            )

        return signals

    def quick_technical_scan(self, tickers: List[str]) -> Dict[str, dict]:
        """
        Lightweight technical-only scan (no ML, no sentiment).
        Returns per-ticker feature dicts.
        """
        from infrastructure.analysis_pipeline import AnalysisPipeline

        pipeline = AnalysisPipeline(
            market=self.market,
            skip_stages=["portfolio_opt", "execution", "post_trade"],
        )
        result = pipeline.run(tickers)
        return result.stages.get("features", {}).data or {}
