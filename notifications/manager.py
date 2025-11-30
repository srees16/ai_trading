"""
Notification system for popup alerts.
"""

from typing import List
from models import NewsItem
from config import Config

try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    print("Warning: plyer not available. Notifications will be printed to console.")


class NotificationManager:
    """Manages popup notifications for significant news."""
    
    def __init__(self):
        """Initialize the notification manager."""
        self.enabled = PLYER_AVAILABLE
    
    def send_notification(
        self, 
        title: str, 
        message: str, 
        duration: int = Config.NOTIFICATION_DURATION
    ):
        """
        Send a popup notification.
        
        Args:
            title: Notification title
            message: Notification message
            duration: Duration in seconds
        """
        if self.enabled:
            try:
                notification.notify(
                    title=title,
                    message=message,
                    app_name="Algo Trading Alert",
                    timeout=duration
                )
            except Exception as e:
                print(f"Error sending notification: {e}")
                self._console_notification(title, message)
        else:
            self._console_notification(title, message)
    
    def _console_notification(self, title: str, message: str):
        """Print notification to console as fallback."""
        print("\n" + "="*60)
        print(f"🔔 ALERT: {title}")
        print("-"*60)
        print(message)
        print("="*60 + "\n")
    
    def notify_high_sentiment_news(self, news_item: NewsItem):
        """
        Send notification for highly positive or negative news.
        
        Args:
            news_item: NewsItem with high sentiment confidence
        """
        if news_item.is_highly_positive():
            title = f"🚀 STRONG BUY SIGNAL: {news_item.ticker}"
            message = (
                f"Highly positive news detected!\n\n"
                f"Title: {news_item.title[:100]}...\n"
                f"Sentiment: {news_item.sentiment_confidence:.1%} confidence\n"
                f"Source: {news_item.source}\n"
                f"URL: {news_item.url}"
            )
            self.send_notification(title, message)
        
        elif news_item.is_highly_negative():
            title = f"⚠️ STRONG SELL SIGNAL: {news_item.ticker}"
            message = (
                f"Highly negative news detected!\n\n"
                f"Title: {news_item.title[:100]}...\n"
                f"Sentiment: {news_item.sentiment_confidence:.1%} confidence\n"
                f"Source: {news_item.source}\n"
                f"URL: {news_item.url}"
            )
            self.send_notification(title, message)
    
    def notify_multiple_news(self, news_items: List[NewsItem]):
        """
        Send notifications for multiple high-sentiment news items.
        
        Args:
            news_items: List of NewsItem objects
        """
        high_sentiment_items = [
            item for item in news_items 
            if item.is_highly_positive() or item.is_highly_negative()
        ]
        
        if high_sentiment_items:
            print(f"\nFound {len(high_sentiment_items)} high-confidence news items.")
            for item in high_sentiment_items:
                self.notify_high_sentiment_news(item)
    
    def notify_trading_signal(self, signal):
        """
        Send notification for a trading signal.
        
        Args:
            signal: TradingSignal object
        """
        if signal.decision.value in ['STRONG_BUY', 'STRONG_SELL']:
            emoji = "🚀" if signal.decision.value == 'STRONG_BUY' else "⚠️"
            title = f"{emoji} {signal.decision.value}: {signal.news_item.ticker}"
            message = (
                f"Decision: {signal.decision.value}\n"
                f"Score: {signal.decision_score:.2f}\n"
                f"News: {signal.news_item.title[:80]}...\n"
                f"Reasoning: {signal.reasoning[:150]}..."
            )
            self.send_notification(title, message)
