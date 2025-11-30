"""
Main orchestration script for the algo-trading alert system.

This script coordinates all components to:
1. Scrape news from multiple sources
2. Analyze sentiment
3. Calculate stock metrics
4. Generate trading decisions
5. Send notifications
6. Save results to file
"""

import asyncio
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


class AlgoTradingSystem:
    """Main orchestration class for the algo-trading alert system."""
    
    def __init__(self, tickers: List[str] = None):
        """
        Initialize the trading system.
        
        Args:
            tickers: List of stock tickers to monitor
        """
        self.tickers = tickers or Config.DEFAULT_TICKERS
        
        print("Initializing Algo Trading Alert System...")
        print(f"Monitoring tickers: {', '.join(self.tickers)}\n")
        
        # Initialize components
        self.news_aggregator = NewsAggregator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.decision_engine = DecisionEngine()
        self.notification_manager = NotificationManager()
        self.storage_manager = StorageManager()
        
        print("System initialized successfully!\n")
    
    async def run(self):
        """Run the complete trading system pipeline."""
        print(f"{'='*70}")
        print(f"Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Step 1: Scrape news
        print("📰 Step 1: Scraping news from multiple sources...")
        all_news = await self.news_aggregator.fetch_news_for_tickers(self.tickers)
        print(f"✓ Collected {len(all_news)} news items\n")
        
        if not all_news:
            print("No news found. Exiting.")
            return
        
        # Step 2: Analyze sentiment
        print("🧠 Step 2: Analyzing sentiment...")
        analyzed_news = self.sentiment_analyzer.analyze_news_items(all_news)
        print(f"✓ Analyzed sentiment for {len(analyzed_news)} items\n")
        
        # Step 3: Send notifications for high-confidence news
        print("🔔 Step 3: Checking for high-confidence alerts...")
        self.notification_manager.notify_multiple_news(analyzed_news)
        
        # Step 4: Calculate metrics and generate signals
        print("\n📊 Step 4: Calculating metrics and generating trading signals...")
        signals: List[TradingSignal] = []
        
        for news_item in analyzed_news:
            print(f"  Analyzing {news_item.ticker}...", end=" ")
            
            # Calculate metrics
            metrics = self.metrics_calculator.get_stock_metrics(news_item.ticker)
            
            # Generate signal
            signal = self.decision_engine.generate_signal(news_item, metrics)
            signals.append(signal)
            
            print(f"✓ Decision: {signal.decision.value}")
            
            # Send notification for strong signals
            if signal.decision.value in ['STRONG_BUY', 'STRONG_SELL']:
                self.notification_manager.notify_trading_signal(signal)
        
        print(f"\n✓ Generated {len(signals)} trading signals\n")
        
        # Step 5: Save results
        print("💾 Step 5: Saving results to file...")
        self.storage_manager.save_signals(signals, append=Config.APPEND_MODE)
        print()
        
        # Step 6: Display summary
        self._display_summary(signals)
        
        print(f"\n{'='*70}")
        print(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
    
    def _display_summary(self, signals: List[TradingSignal]):
        """Display summary of trading signals."""
        print("📈 Summary of Trading Signals:")
        print("-" * 70)
        
        # Count decisions
        decision_counts = {}
        for signal in signals:
            decision = signal.decision.value
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        # Display counts
        for decision, count in sorted(decision_counts.items()):
            print(f"  {decision}: {count}")
        
        print("-" * 70)
        
        # Display top signals
        print("\n🔝 Top 5 Strongest Buy Signals:")
        buy_signals = [s for s in signals if s.decision.value in ['STRONG_BUY', 'BUY']]
        buy_signals.sort(key=lambda x: x.decision_score, reverse=True)
        
        for i, signal in enumerate(buy_signals[:5], 1):
            print(f"  {i}. {signal.news_item.ticker} - {signal.decision.value} "
                  f"(Score: {signal.decision_score:.2f})")
            print(f"     {signal.news_item.title[:70]}...")
        
        print("\n⚠️  Top 5 Strongest Sell Signals:")
        sell_signals = [s for s in signals if s.decision.value in ['STRONG_SELL', 'SELL']]
        sell_signals.sort(key=lambda x: x.decision_score)
        
        for i, signal in enumerate(sell_signals[:5], 1):
            print(f"  {i}. {signal.news_item.ticker} - {signal.decision.value} "
                  f"(Score: {signal.decision_score:.2f})")
            print(f"     {signal.news_item.title[:70]}...")
        
        print()


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
        print("\n\nSystem interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\nError running system: {e}")
        import traceback
        traceback.print_exc()
