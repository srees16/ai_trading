"""
Notification system for popup alerts and email reports.
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from config import Config
from models import NewsItem

logger = logging.getLogger(__name__)

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

    # ── Email helpers ────────────────────────────────────────────────

    @staticmethod
    def send_wsb_email(
        news_items: List[NewsItem],
        tickers: List[str],
        recipients: Optional[List[str]] = None,
    ) -> bool:
        """
        Send an email summary of WallStreetBets mentions for the
        analysed tickers.

        Uses SMTP with credentials from environment variables:
            CENTURION_EMAIL_HOST   (default: smtp-mail.outlook.com)
            CENTURION_EMAIL_PORT   (default: 587)
            CENTURION_EMAIL_USER   (sender address)
            CENTURION_EMAIL_PASS   (sender password / app-password)

        Args:
            news_items: List of all NewsItem objects from analysis
            tickers: Tickers that were analysed
            recipients: Override list; defaults to ["s.srees@live.com"]

        Returns:
            True on success, False on failure or missing config
        """
        if recipients is None:
            recipients = ["s.srees@live.com"]

        smtp_host = os.getenv("CENTURION_EMAIL_HOST", "smtp-mail.outlook.com")
        smtp_port = int(os.getenv("CENTURION_EMAIL_PORT", "587"))
        smtp_user = os.getenv("CENTURION_EMAIL_USER", "")
        smtp_pass = os.getenv("CENTURION_EMAIL_PASS", "")

        if not smtp_user or not smtp_pass:
            logger.warning(
                "Email not configured (set CENTURION_EMAIL_USER / "
                "CENTURION_EMAIL_PASS). Skipping WSB email."
            )
            return False

        # ── Build HTML body ──────────────────────────────────────────
        wsb_items = [n for n in news_items if n.source == "WallStreetBets"]
        by_ticker: dict[str, list[NewsItem]] = {}
        for item in wsb_items:
            by_ticker.setdefault(item.ticker, []).append(item)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        rows = ""
        for t in tickers:
            items = by_ticker.get(t, [])
            if not items:
                rows += (
                    f"<tr><td style='padding:6px 12px;border:1px solid #ddd;'>"
                    f"<b>{t}</b></td>"
                    f"<td style='padding:6px 12px;border:1px solid #ddd;' "
                    f"colspan='3'><em>No WSB mentions</em></td></tr>\n"
                )
                continue
            for item in items:
                sentiment = (
                    item.sentiment_label.value.title()
                    if item.sentiment_label
                    else "N/A"
                )
                link = (
                    f"<a href='{item.url}'>{item.title[:80]}</a>"
                    if item.url
                    else item.title[:80]
                )
                rows += (
                    f"<tr>"
                    f"<td style='padding:6px 12px;border:1px solid #ddd;'><b>{t}</b></td>"
                    f"<td style='padding:6px 12px;border:1px solid #ddd;'>{link}</td>"
                    f"<td style='padding:6px 12px;border:1px solid #ddd;'>{sentiment}</td>"
                    f"<td style='padding:6px 12px;border:1px solid #ddd;'>"
                    f"{item.timestamp.strftime('%H:%M') if item.timestamp else ''}</td>"
                    f"</tr>\n"
                )

        html = f"""\
<html><body style="font-family:Segoe UI,Arial,sans-serif;">
<h2 style="color:#1a1a2e;">Centurion Capital &mdash; WallStreetBets Report</h2>
<p>Generated: {now} &nbsp;|&nbsp; Tickers analysed: {', '.join(tickers)}</p>
<table style="border-collapse:collapse;width:100%;">
<thead>
<tr style="background:#1a1a2e;color:#fff;">
  <th style="padding:8px 12px;text-align:left;">Ticker</th>
  <th style="padding:8px 12px;text-align:left;">Post</th>
  <th style="padding:8px 12px;text-align:left;">Sentiment</th>
  <th style="padding:8px 12px;text-align:left;">Time</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
<br>
<p style="font-size:0.85rem;color:#999;">&copy; 2026 Centurion Capital LLC</p>
</body></html>"""

        # ── Send ─────────────────────────────────────────────────────
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"WSB Mentions — {', '.join(tickers)} — {now}"
        msg["From"] = smtp_user
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())
            logger.info("WSB email sent to %s", ", ".join(recipients))
            return True
        except Exception as exc:
            logger.error("Failed to send WSB email: %s", exc)
            return False