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
                # Windows NOTIFYICONDATAW limits: title 64 chars, message 256 chars
                safe_title = title[:63] if len(title) > 63 else title
                safe_message = message[:255] if len(message) > 255 else message
                notification.notify(
                    title=safe_title,
                    message=safe_message,
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
        print(f"ALERT: {title}")
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
            title = f" STRONG BUY SIGNAL: {news_item.ticker}"
            message = (
                f"Highly positive news detected!\n\n"
                f"Title: {news_item.title[:100]}...\n"
                f"Sentiment: {news_item.sentiment_confidence:.1%} confidence\n"
                f"Source: {news_item.source}\n"
                f"URL: {news_item.url}"
            )
            self.send_notification(title, message)
        
        elif news_item.is_highly_negative():
            title = f" STRONG SELL SIGNAL: {news_item.ticker}"
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
            emoji = "" if signal.decision.value == 'STRONG_BUY' else ""
            title = f"{emoji} {signal.decision.value}: {signal.news_item.ticker}"
            message = (
                f"Decision: {signal.decision.value}\n"
                f"Score: {signal.decision_score:.2f}\n"
                f"News: {signal.news_item.title[:80]}...\n"
                f"Reasoning: {signal.reasoning[:150]}..."
            )
            self.send_notification(title, message)

    # ── Pipeline / Order event notifications ─────────────────────────

    def notify_pipeline_signals(self, buy_verdicts: list, sell_verdicts: list):
        """Notify when the screener/scorer pipeline finds actionable signals."""
        parts = []
        if buy_verdicts:
            syms = ", ".join(
                getattr(v, "ticker", str(v)).replace(".NS", "")
                for v in buy_verdicts[:5]
            )
            parts.append(f"{len(buy_verdicts)} BUY: {syms}")
        if sell_verdicts:
            syms = ", ".join(
                getattr(v, "ticker", str(v)).replace(".NS", "")
                for v in sell_verdicts[:5]
            )
            parts.append(f"{len(sell_verdicts)} SELL: {syms}")
        if parts:
            self.send_notification(
                "Centurion — Signals Detected",
                " | ".join(parts),
                duration=15,
            )

    def notify_order_placed(self, symbol: str, side: str, qty: int,
                            price: float, order_id: str):
        """Notify after a Kite order is successfully placed."""
        self.send_notification(
            f"{side} Order Placed: {symbol}",
            f"{side} {symbol} × {qty} @ ₹{price:.2f}\nOrder ID: {order_id}",
        )
        # Also send email
        self.email_order_confirmation(
            symbol=symbol, side=side, quantity=qty,
            entry_price=price, fill_price=price,
            order_id=str(order_id), status="PLACED",
        )

    def notify_order_failed(self, symbol: str, side: str, error: str):
        """Notify when an order fails."""
        self.send_notification(
            f"Order FAILED: {symbol}",
            f"{side} {symbol} — {error}",
        )
        self.email_order_confirmation(
            symbol=symbol, side=side, quantity=0,
            entry_price=0, fill_price=0,
            order_id="-", status="FAILED", error=error,
        )

    def notify_sl_tp_event(self, event_type: str, symbol: str,
                           exit_price: float = 0):
        """Notify on SL trigger, TP fill, or trailing SL update."""
        labels = {
            "SL_TRIGGERED": "Stop-Loss Hit",
            "TP_FILLED": "Target Reached",
            "TRAILING_SL_UPDATED": "Trailing SL Moved Up",
        }
        label = labels.get(event_type, event_type)
        msg = f"{label}: {symbol}"
        if exit_price > 0:
            msg += f" @ ₹{exit_price:.2f}"
        self.send_notification(f"Trade {label}", msg)

    def notify_session_expired(self):
        """Notify when Kite session has expired."""
        self.send_notification(
            "Kite Session Expired",
            "Re-authenticate to continue placing orders.",
            duration=30,
        )

    # ── Email helpers ────────────────────────────────────────────────

    def email_order_confirmation(
        self,
        symbol: str, side: str, quantity: int,
        entry_price: float, fill_price: float,
        order_id: str, status: str,
        exchange: str = "NSE", error: str = None,
        recipients: Optional[List[str]] = None,
    ) -> bool:
        """Send an HTML email with order details after placement.

        Uses the same SMTP credentials as ``send_wsb_email``.
        Defaults to s.srees@live.com if no recipients specified.
        """
        if recipients is None:
            recipients = ["s.srees@live.com"]

        smtp_host = os.getenv("CENTURION_EMAIL_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("CENTURION_EMAIL_PORT", "587"))
        smtp_user = os.getenv("CENTURION_EMAIL_USER", "")
        smtp_pass = os.getenv("CENTURION_EMAIL_PASS", "")

        if not smtp_user or not smtp_pass:
            logger.debug("Email not configured — skipping order email")
            return False

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        side_color = "#15803d" if side == "BUY" else "#dc2626"
        status_color = "#15803d" if status in ("PLACED", "FILLED", "COMPLETE") else "#dc2626"
        error_row = ""
        if error:
            error_row = (
                f'<tr><td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">'
                f'Error</td><td style="padding:8px 14px;border:1px solid #e5e7eb;color:#dc2626;">'
                f'{error}</td></tr>'
            )

        html = f"""\
<html><body style="font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;">
<div style="max-width:520px;margin:0 auto;background:#fff;border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">
  <div style="background:#1a1a2e;padding:16px 24px;">
    <h2 style="margin:0;color:#fff;font-size:18px;">
      Centurion &mdash; Order {status}
    </h2>
  </div>
  <div style="padding:20px 24px;">
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <tr>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">Symbol</td>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;font-weight:bold;">
          {exchange}:{symbol}</td>
      </tr>
      <tr>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">Side</td>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;font-weight:bold;color:{side_color};">
          {side}</td>
      </tr>
      <tr>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">Quantity</td>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;">{quantity}</td>
      </tr>
      <tr>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">Entry Price</td>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;">
          &#8377; {entry_price:,.2f}</td>
      </tr>
      <tr>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">Fill Price</td>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;">
          &#8377; {fill_price:,.2f}</td>
      </tr>
      <tr>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">Order ID</td>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;font-family:monospace;">
          {order_id}</td>
      </tr>
      <tr>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">Status</td>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;font-weight:bold;color:{status_color};">
          {status}</td>
      </tr>
      <tr>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;color:#666;">Placed At</td>
        <td style="padding:8px 14px;border:1px solid #e5e7eb;">{now}</td>
      </tr>
      {error_row}
    </table>
  </div>
  <div style="padding:12px 24px;background:#f3f4f6;font-size:12px;color:#9ca3af;text-align:center;">
    &copy; 2026 Centurion Capital LLC &mdash; Automated Trading System
  </div>
</div>
</body></html>"""

        side_emoji = "🟢" if side == "BUY" else "🔴"
        subject = f"{side_emoji} {side} {symbol} x{quantity} @ ₹{fill_price:,.2f} — {status}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())
            logger.info("Order email sent to %s for %s %s", ", ".join(recipients), side, symbol)
            return True
        except Exception as exc:
            logger.error("Failed to send order email: %s", exc)
            return False

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
            CENTURION_EMAIL_HOST   (default: smtp.gmail.com)
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

        smtp_host = os.getenv("CENTURION_EMAIL_HOST", "smtp.gmail.com")
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
        msg["Subject"] = f" WSB Mentions — {', '.join(tickers)} — {now}"
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

    # ── Generic HTML email sender ────────────────────────────────────

    @staticmethod
    def _send_html_email(
        subject: str,
        html_body: str,
        recipients: Optional[List[str]] = None,
    ) -> bool:
        """Send a generic HTML email via configured SMTP.

        Returns True on success, False on failure or missing config.
        """
        if recipients is None:
            recipients = ["s.srees@live.com"]

        smtp_host = os.getenv("CENTURION_EMAIL_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("CENTURION_EMAIL_PORT", "587"))
        smtp_user = os.getenv("CENTURION_EMAIL_USER", "")
        smtp_pass = os.getenv("CENTURION_EMAIL_PASS", "")

        if not smtp_user or not smtp_pass:
            logger.debug("Email not configured — skipping")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())
            logger.info("Email sent: %s → %s", subject[:60], ", ".join(recipients))
            return True
        except Exception as exc:
            logger.error("Failed to send email '%s': %s", subject[:60], exc)
            return False

    # ── Daily Pipeline Report Email ──────────────────────────────────

    def email_daily_pipeline_report(self, summary: dict) -> bool:
        """Send daily pipeline run summary + paper dashboard via email."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        buy_count = summary.get("buy_signals", 0)
        sell_count = summary.get("sell_signals", 0)
        universe = summary.get("universe_size", 0)
        screened = summary.get("screened_count", 0)
        status = summary.get("status", "unknown")

        # Paper trading dashboard
        dash_html = ""
        try:
            from kite_connect.trading.paper_trader import PaperTrader
            pt = PaperTrader()
            dash = pt.dashboard()
            pnl_color = "#15803d" if dash.total_pnl >= 0 else "#dc2626"
            dash_html = f"""
            <h3 style="margin-top:24px;color:#1a1a2e;">Paper Trading Dashboard</h3>
            <table style="border-collapse:collapse;width:100%;font-size:14px;">
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Capital</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">
                    ₹{dash.initial_capital:,.0f} → ₹{dash.current_capital:,.0f}</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Total P&amp;L</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;font-weight:bold;color:{pnl_color};">
                    ₹{dash.total_pnl:,.0f} ({dash.total_pnl_pct:+.1f}%)</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Win Rate</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.win_rate:.0%}</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Sharpe Ratio</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.sharpe_ratio:.2f}</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Max Drawdown</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.max_drawdown_pct:.1f}%</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Sortino / Calmar</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.sortino_ratio:.2f} / {dash.calmar_ratio:.2f}</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Open / Closed</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.open_positions} open, {dash.closed_trades} closed</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Profit Factor</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.profit_factor:.2f}</td></tr>
            </table>"""
        except Exception as exc:
            dash_html = f"<p><em>Paper dashboard unavailable: {exc}</em></p>"

        # Verdict table
        verdicts = summary.get("verdicts", [])
        v_rows = ""
        for v in verdicts[:20]:
            cls = v.get("classification", "")
            color = "#15803d" if "BUY" in cls else "#dc2626" if "SELL" in cls else "#666"
            v_rows += (
                f"<tr><td style='padding:4px 10px;border:1px solid #e5e7eb;'>{v.get('ticker','')}</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;font-weight:bold;color:{color};'>{cls}</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;'>{v.get('score',0):.3f}</td>"
                f"<td style='padding:4px 10px;border:1px solid #e5e7eb;'>{v.get('confidence',0):.0%}</td></tr>"
            )

        html = f"""\
<html><body style="font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;">
<div style="max-width:640px;margin:0 auto;background:#fff;border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">
  <div style="background:#1a1a2e;padding:16px 24px;">
    <h2 style="margin:0;color:#fff;font-size:18px;">
      Centurion &mdash; Daily Pipeline Report
    </h2>
    <p style="margin:4px 0 0;color:#9ca3af;font-size:13px;">{now}</p>
  </div>
  <div style="padding:20px 24px;">
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Status</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;font-weight:bold;">{status}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Universe → Screened</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{universe} → {screened}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">BUY Signals</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;color:#15803d;font-weight:bold;">{buy_count}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">SELL Signals</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;color:#dc2626;font-weight:bold;">{sell_count}</td></tr>
    </table>

    {dash_html}

    {"<h3 style='margin-top:24px;color:#1a1a2e;'>Top Signals</h3>" if v_rows else ""}
    {"<table style='border-collapse:collapse;width:100%;font-size:13px;'>" if v_rows else ""}
    {"<tr style='background:#f3f4f6;'><th style='padding:6px 10px;text-align:left;'>Ticker</th><th style='padding:6px 10px;text-align:left;'>Signal</th><th style='padding:6px 10px;text-align:left;'>Score</th><th style='padding:6px 10px;text-align:left;'>Conf</th></tr>" if v_rows else ""}
    {v_rows}
    {"</table>" if v_rows else ""}
  </div>
  <div style="padding:12px 24px;background:#f3f4f6;font-size:12px;color:#9ca3af;text-align:center;">
    &copy; 2026 Centurion Capital LLC &mdash; Paper Trading Mode
  </div>
</div></body></html>"""

        emoji = "🟢" if buy_count > 0 else "⚪"
        subject = f"{emoji} Centurion Daily — {buy_count} BUY, {sell_count} SELL — {now}"
        return self._send_html_email(subject, html)

    # ── Weekly Reconciliation Email ──────────────────────────────────

    def email_reconciliation_report(self, report: dict) -> bool:
        """Send weekly paper↔backtest reconciliation comparison via email."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        def _leg_rows(leg_data: dict, label: str) -> str:
            discs = leg_data.get("discrepancies", [])
            if not discs:
                return (
                    f"<tr><td style='padding:6px 12px;border:1px solid #e5e7eb;'>{label}</td>"
                    f"<td style='padding:6px 12px;border:1px solid #e5e7eb;color:#15803d;' colspan='2'>"
                    f"✅ No discrepancies</td></tr>"
                )
            rows = ""
            for d in discs[:10]:
                rows += (
                    f"<tr><td style='padding:6px 12px;border:1px solid #e5e7eb;'>{label}</td>"
                    f"<td style='padding:6px 12px;border:1px solid #e5e7eb;'>{d}</td></tr>"
                )
            return rows

        leg1 = report.get("paper_vs_live", {})
        leg2 = report.get("backtest_vs_live", {})
        leg3 = report.get("backtest_vs_paper", {})

        total_issues = sum(
            len(leg.get("discrepancies", []))
            for leg in [leg1, leg2, leg3]
        )

        # Paper dashboard
        dash_html = ""
        try:
            from kite_connect.trading.paper_trader import PaperTrader
            pt = PaperTrader()
            dash = pt.dashboard()
            pnl_color = "#15803d" if dash.total_pnl >= 0 else "#dc2626"
            dash_html = f"""
            <h3 style="margin-top:24px;color:#1a1a2e;">Paper Trading — Week Summary</h3>
            <table style="border-collapse:collapse;width:100%;font-size:14px;">
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Capital</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">
                    ₹{dash.initial_capital:,.0f} → ₹{dash.current_capital:,.0f}</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Total P&amp;L</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;font-weight:bold;color:{pnl_color};">
                    ₹{dash.total_pnl:,.0f} ({dash.total_pnl_pct:+.1f}%)</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Win Rate</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.win_rate:.0%}</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Sharpe</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.sharpe_ratio:.2f}</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Max DD</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.max_drawdown_pct:.1f}%</td></tr>
              <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Closed Trades</td>
                  <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.closed_trades}</td></tr>
            </table>"""
        except Exception:
            pass

        html = f"""\
<html><body style="font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;">
<div style="max-width:640px;margin:0 auto;background:#fff;border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">
  <div style="background:#1a1a2e;padding:16px 24px;">
    <h2 style="margin:0;color:#fff;font-size:18px;">
      Centurion &mdash; Weekly Reconciliation
    </h2>
    <p style="margin:4px 0 0;color:#9ca3af;font-size:13px;">{now}</p>
  </div>
  <div style="padding:20px 24px;">
    <p style="font-size:15px;">
      <strong>Total discrepancies:</strong>
      <span style="color:{'#dc2626' if total_issues > 0 else '#15803d'};font-weight:bold;">
        {total_issues}</span>
    </p>
    <table style="border-collapse:collapse;width:100%;font-size:13px;">
      <tr style="background:#f3f4f6;">
        <th style="padding:6px 10px;text-align:left;">Leg</th>
        <th style="padding:6px 10px;text-align:left;">Detail</th>
      </tr>
      {_leg_rows(leg1, "Paper ↔ Live")}
      {_leg_rows(leg2, "Backtest ↔ Live")}
      {_leg_rows(leg3, "Backtest ↔ Paper")}
    </table>
    {dash_html}
  </div>
  <div style="padding:12px 24px;background:#f3f4f6;font-size:12px;color:#9ca3af;text-align:center;">
    &copy; 2026 Centurion Capital LLC
  </div>
</div></body></html>"""

        status_emoji = "🔴" if total_issues > 0 else "🟢"
        subject = f"{status_emoji} Centurion Weekly Recon — {total_issues} issues — {now}"
        return self._send_html_email(subject, html)

    # ── Go-Live Confidence Email ─────────────────────────────────────

    def email_go_live_recommendation(self, dash, weeks_active: int) -> bool:
        """Send email when paper trading metrics meet go-live thresholds.

        Thresholds: Win rate > 55%, Sharpe > 1.0, Max DD < 15%,
                    Profit factor > 1.5, at least 4 weeks of trading.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        checks = [
            ("Win Rate > 55%", dash.win_rate > 0.55, f"{dash.win_rate:.0%}"),
            ("Sharpe Ratio > 1.0", dash.sharpe_ratio > 1.0, f"{dash.sharpe_ratio:.2f}"),
            ("Max Drawdown < 15%", dash.max_drawdown_pct < 15.0, f"{dash.max_drawdown_pct:.1f}%"),
            ("Profit Factor > 1.5", dash.profit_factor > 1.5, f"{dash.profit_factor:.2f}"),
            (f"Weeks Active ≥ 4", weeks_active >= 4, f"{weeks_active}"),
            ("Closed Trades ≥ 20", dash.closed_trades >= 20, f"{dash.closed_trades}"),
        ]
        all_pass = all(ok for _, ok, _ in checks)

        check_rows = ""
        for label, passed, value in checks:
            icon = "✅" if passed else "❌"
            color = "#15803d" if passed else "#dc2626"
            check_rows += (
                f"<tr><td style='padding:6px 12px;border:1px solid #e5e7eb;'>{icon} {label}</td>"
                f"<td style='padding:6px 12px;border:1px solid #e5e7eb;color:{color};font-weight:bold;'>"
                f"{value}</td></tr>"
            )

        verdict = (
            "<div style='background:#dcfce7;border:2px solid #15803d;border-radius:8px;"
            "padding:16px;margin:16px 0;text-align:center;'>"
            "<h3 style='margin:0;color:#15803d;'>🚀 READY FOR LIVE TRADING</h3>"
            "<p style='margin:8px 0 0;color:#166534;'>All criteria met. Set "
            "<code>PAPER_TRADE_MODE = False</code> in config.py to go live.</p></div>"
        ) if all_pass else (
            "<div style='background:#fef2f2;border:2px solid #dc2626;border-radius:8px;"
            "padding:16px;margin:16px 0;text-align:center;'>"
            "<h3 style='margin:0;color:#dc2626;'>⏳ NOT YET READY</h3>"
            "<p style='margin:8px 0 0;color:#991b1b;'>Some criteria not met. "
            "Continue paper trading.</p></div>"
        )

        pnl_color = "#15803d" if dash.total_pnl >= 0 else "#dc2626"

        html = f"""\
<html><body style="font-family:Segoe UI,Arial,sans-serif;background:#f9fafb;padding:20px;">
<div style="max-width:640px;margin:0 auto;background:#fff;border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);overflow:hidden;">
  <div style="background:#1a1a2e;padding:16px 24px;">
    <h2 style="margin:0;color:#fff;font-size:18px;">
      Centurion &mdash; Go-Live Assessment
    </h2>
    <p style="margin:4px 0 0;color:#9ca3af;font-size:13px;">{now}</p>
  </div>
  <div style="padding:20px 24px;">
    {verdict}
    <h3 style="color:#1a1a2e;">Criteria Check</h3>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <tr style="background:#f3f4f6;">
        <th style="padding:6px 10px;text-align:left;">Criterion</th>
        <th style="padding:6px 10px;text-align:left;">Value</th>
      </tr>
      {check_rows}
    </table>
    <h3 style="margin-top:20px;color:#1a1a2e;">Performance Summary</h3>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Capital</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">
            ₹{dash.initial_capital:,.0f} → ₹{dash.current_capital:,.0f}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Total P&amp;L</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;font-weight:bold;color:{pnl_color};">
            ₹{dash.total_pnl:,.0f} ({dash.total_pnl_pct:+.1f}%)</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">Sortino / Calmar</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.sortino_ratio:.2f} / {dash.calmar_ratio:.2f}</td></tr>
      <tr><td style="padding:6px 12px;border:1px solid #e5e7eb;color:#666;">CVaR 95%</td>
          <td style="padding:6px 12px;border:1px solid #e5e7eb;">{dash.cvar_95:.2%}</td></tr>
    </table>
  </div>
  <div style="padding:12px 24px;background:#f3f4f6;font-size:12px;color:#9ca3af;text-align:center;">
    &copy; 2026 Centurion Capital LLC
  </div>
</div></body></html>"""

        emoji = "🚀" if all_pass else "📊"
        status = "READY" if all_pass else "NOT YET"
        subject = f"{emoji} Centurion Go-Live: {status} — Sharpe {dash.sharpe_ratio:.2f}, WR {dash.win_rate:.0%}"
        return self._send_html_email(subject, html)