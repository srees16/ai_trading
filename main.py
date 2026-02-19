"""
Main Orchestration Module for Centurion Capital LLC.

Coordinates all trading system components to:
1. Scrape news from multiple sources
2. Analyze sentiment using NLP
3. Calculate fundamental and technical stock metrics
4. Generate trading decisions
5. Send notifications for actionable signals
6. Persist results to file/database
"""

import asyncio
import logging
from typing import List
from datetime import datetime

from config import Config
from models import TradingSignal
from scrapers.aggregator import NewsAggregator
from sentiment import SentimentAnalyzer
from metrics import MetricsCalculator
from decision_engine import DecisionEngine
from notifications import NotificationManager
from storage import StorageManager

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class AlgoTradingSystem:
    """Main orchestration class for the algo-trading alert system."""
    
    def __init__(self, tickers: List[str] = None):
        """
        Initialize the trading system.
        
        Args:
            tickers: List of stock tickers to monitor
        """
        self.tickers = tickers or Config.DEFAULT_TICKERS
        
        logger.info("Initializing Algo Trading Alert System...")
        logger.info(f"Monitoring tickers: {', '.join(self.tickers)}")
        
        # Initialize components
        self.news_aggregator = NewsAggregator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.decision_engine = DecisionEngine()
        self.notification_manager = NotificationManager()
        self.storage_manager = StorageManager()
        
        logger.info("System initialized successfully!")
    
    async def run(self):
        """Run the complete trading system pipeline."""
        logger.info("=" * 70)
        logger.info(f"Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        # Step 1: Scrape news
        logger.info("Step 1: Scraping news from multiple sources...")
        all_news = await self.news_aggregator.fetch_news_for_tickers(self.tickers)
        logger.info(f"Collected {len(all_news)} news items")
        
        if not all_news:
            logger.warning("No news found. Exiting.")
            return
        
        # Step 2: Analyze sentiment
        logger.info("Step 2: Analyzing sentiment...")
        analyzed_news = self.sentiment_analyzer.analyze_news_items(all_news)
        logger.info(f"Analyzed sentiment for {len(analyzed_news)} items")
        
        # Step 3: Send notifications for high-confidence news
        logger.info("Step 3: Checking for high-confidence alerts...")
        self.notification_manager.notify_multiple_news(analyzed_news)
        
        # Step 4: Calculate metrics and generate signals
        logger.info("Step 4: Calculating metrics and generating trading signals...")
        signals: List[TradingSignal] = []
        
        for news_item in analyzed_news:
            logger.debug(f"  Analyzing {news_item.ticker}...")
            
            # Calculate metrics
            metrics = self.metrics_calculator.get_stock_metrics(news_item.ticker)
            
            # Generate signal
            signal = self.decision_engine.generate_signal(news_item, metrics)
            signals.append(signal)
            
            logger.info(f"  {news_item.ticker}: Decision = {signal.decision.value}")
            
            # Send notification for strong signals
            if signal.decision.value in ['STRONG_BUY', 'STRONG_SELL']:
                self.notification_manager.notify_trading_signal(signal)
        
        logger.info(f"Generated {len(signals)} trading signals")
        
        # Step 5: Save results
        logger.info("Step 5: Saving results to file...")
        self.storage_manager.save_signals(signals, append=Config.APPEND_MODE)
        
        # Step 6: Display summary
        self._display_summary(signals)
        
        logger.info("=" * 70)
        logger.info(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
    
    def _display_summary(self, signals: List[TradingSignal]):
        """Display summary of trading signals."""
        logger.info("Summary of Trading Signals:")
        logger.info("-" * 70)
        
        # Count decisions
        decision_counts = {}
        for signal in signals:
            decision = signal.decision.value
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        # Display counts
        for decision, count in sorted(decision_counts.items()):
            logger.info(f"  {decision}: {count}")
        
        logger.info("-" * 70)
        
        # Display top signals
        logger.info("🔝 Top 5 Strongest Buy Signals:")
        buy_signals = [s for s in signals if s.decision.value in ['STRONG_BUY', 'BUY']]
        buy_signals.sort(key=lambda x: x.decision_score, reverse=True)
        
        for i, signal in enumerate(buy_signals[:5], 1):
            logger.info(f"  {i}. {signal.news_item.ticker} - {signal.decision.value} "
                  f"(Score: {signal.decision_score:.2f})")
            logger.debug(f"     {signal.news_item.title[:70]}...")
        
        logger.info("⚠️  Top 5 Strongest Sell Signals:")
        sell_signals = [s for s in signals if s.decision.value in ['STRONG_SELL', 'SELL']]
        sell_signals.sort(key=lambda x: x.decision_score)
        
        for i, signal in enumerate(sell_signals[:5], 1):
            logger.info(f"  {i}. {signal.news_item.ticker} - {signal.decision.value} "
                  f"(Score: {signal.decision_score:.2f})")
            logger.debug(f"     {signal.news_item.title[:70]}...")


async def main():
    """Main entry point."""
    # You can customize tickers here
    # tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    # system = AlgoTradingSystem(tickers=tickers)
    
    # Or use default tickers from config
    system = AlgoTradingSystem()
    
    await system.run()


if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System interrupted by user. Exiting gracefully...")
    except Exception as e:
        logger.error(f"Error running system: {e}", exc_info=True)
