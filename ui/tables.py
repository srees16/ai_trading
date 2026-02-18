"""
Data Tables Module for Centurion Capital LLC.

Contains table rendering functions for signals and analysis results.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

from ui.styles import get_decision_style, get_signal_style
from ui.components import get_decision_emoji


def render_simple_summary_table(signals: List[Any]):
    """
    Render a summary table with signals aggregated by stock and source.
    
    Args:
        signals: List of TradingSignal objects
    """
    if not signals:
        return
    
    st.subheader("📰 News by Source")
    
    # Group signals by (stock, source)
    grouped: Dict[tuple, List] = defaultdict(list)
    stock_prices: Dict[str, float] = {}
    
    for signal in signals:
        key = (signal.news_item.ticker, signal.news_item.source)
        grouped[key].append(signal)
        if signal.metrics and signal.metrics.current_price:
            stock_prices[signal.news_item.ticker] = signal.metrics.current_price
    
    summary_data = []
    for (ticker, source), group_signals in grouped.items():
        avg_score = sum(s.decision_score for s in group_signals) / len(group_signals)
        
        decisions = [s.decision.value for s in group_signals]
        most_common_decision = max(set(decisions), key=decisions.count)
        
        sentiments = [
            s.news_item.sentiment_label.value if s.news_item.sentiment_label else 'neutral'
            for s in group_signals
        ]
        most_common_sentiment = max(set(sentiments), key=sentiments.count)
        
        signal_emoji = get_decision_emoji(most_common_decision)
        price = stock_prices.get(ticker)
        
        summary_data.append({
            'Stock': ticker,
            'Source': source,
            'News Count': len(group_signals),
            'Avg Score': round(avg_score, 2),
            'Signal': f"{signal_emoji} {most_common_decision.replace('_', ' ')}",
            'Sentiment': most_common_sentiment.title(),
            'Price': f"${price:.2f}" if price else 'N/A'
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values(['Stock', 'Avg Score'], ascending=[True, False])
    
    st.dataframe(df.astype(str), use_container_width=True, hide_index=True)
    
    # Also show aggregated view per stock
    _render_overall_stock_signals(signals)


def _render_overall_stock_signals(signals: List[Any]):
    """Render overall stock signals summary."""
    st.subheader("📈 Overall Stock Signals")
    
    stock_summary: Dict[str, Dict] = defaultdict(lambda: {
        'scores': [], 'decisions': [], 'sentiments': [], 'price': None
    })
    
    for signal in signals:
        ticker = signal.news_item.ticker
        stock_summary[ticker]['scores'].append(signal.decision_score)
        stock_summary[ticker]['decisions'].append(signal.decision.value)
        if signal.news_item.sentiment_label:
            stock_summary[ticker]['sentiments'].append(signal.news_item.sentiment_label.value)
        if signal.metrics and signal.metrics.current_price:
            stock_summary[ticker]['price'] = signal.metrics.current_price
    
    stock_data = []
    for ticker, data in stock_summary.items():
        avg_score = sum(data['scores']) / len(data['scores'])
        most_common_decision = max(set(data['decisions']), key=data['decisions'].count)
        most_common_sentiment = (
            max(set(data['sentiments']), key=data['sentiments'].count)
            if data['sentiments'] else 'neutral'
        )
        
        signal_emoji = get_decision_emoji(most_common_decision)
        
        stock_data.append({
            'Stock': ticker,
            'Total News': len(data['scores']),
            'Avg Score': round(avg_score, 2),
            'Overall Signal': f"{signal_emoji} {most_common_decision.replace('_', ' ')}",
            'Sentiment': most_common_sentiment.title(),
            'Price': f"${data['price']:.2f}" if data['price'] else 'N/A'
        })
    
    stock_df = pd.DataFrame(stock_data)
    stock_df = stock_df.sort_values('Avg Score', ascending=False)
    
    st.dataframe(stock_df.astype(str), use_container_width=True, hide_index=True)


def render_signals_table(signals: List[Any]):
    """
    Render detailed signals table with styling.
    
    Args:
        signals: List of TradingSignal objects
    """
    if not signals:
        return
    
    st.subheader("📋 Detailed Analysis Results")
    
    data = []
    for signal in signals:
        data.append({
            'Ticker': signal.news_item.ticker,
            'Decision': signal.decision.value,
            'Score': f"{signal.decision_score:.2f}",
            'Sentiment': (
                signal.news_item.sentiment_label.value
                if signal.news_item.sentiment_label else 'N/A'
            ),
            'Confidence': (
                f"{signal.news_item.sentiment_confidence:.1%}"
                if signal.news_item.sentiment_confidence else 'N/A'
            ),
            'Price': (
                f"${signal.metrics.current_price:.2f}"
                if signal.metrics and signal.metrics.current_price else 'N/A'
            ),
            'RSI': (
                f"{signal.metrics.rsi:.1f}"
                if signal.metrics and signal.metrics.rsi else 'N/A'
            ),
            'Z-Score': (
                f"{signal.metrics.altman_z_score:.2f}"
                if signal.metrics and signal.metrics.altman_z_score else 'N/A'
            ),
            'F-Score': (
                str(signal.metrics.piotroski_f_score)
                if signal.metrics and signal.metrics.piotroski_f_score is not None else 'N/A'
            ),
            'Source': signal.news_item.source,
            'Title': (
                signal.news_item.title[:50] + '...'
                if len(signal.news_item.title) > 50 else signal.news_item.title
            )
        })
    
    df = pd.DataFrame(data)
    
    # Force all columns to string to prevent Arrow serialization issues
    df = df.astype(str)
    
    def highlight_decision(val):
        return get_decision_style(val)
    
    styled_df = df.style.map(highlight_decision, subset=['Decision'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name=f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_top_signals(signals: List[Any]):
    """
    Render top buy and sell signals.
    
    Args:
        signals: List of TradingSignal objects
    """
    if not signals:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔝 Top Buy Signals")
        buy_signals = [s for s in signals if s.decision.value in ['STRONG_BUY', 'BUY']]
        buy_signals.sort(key=lambda x: x.decision_score, reverse=True)
        
        if buy_signals:
            for i, signal in enumerate(buy_signals[:5], 1):
                with st.expander(
                    f"{i}. {signal.news_item.ticker} - {signal.decision.value} ({signal.decision_score:.2f})"
                ):
                    st.write(f"**Title:** {signal.news_item.title}")
                    st.write(f"**Source:** {signal.news_item.source}")
                    sentiment = (
                        signal.news_item.sentiment_label.value
                        if signal.news_item.sentiment_label else 'N/A'
                    )
                    st.write(f"**Sentiment:** {sentiment}")
                    st.write(f"**Reasoning:** {signal.reasoning}")
                    if signal.news_item.url:
                        st.write(f"**Link:** [{signal.news_item.url}]({signal.news_item.url})")
        else:
            st.info("No buy signals found")
    
    with col2:
        st.subheader("⚠️ Top Sell Signals")
        sell_signals = [s for s in signals if s.decision.value in ['STRONG_SELL', 'SELL']]
        sell_signals.sort(key=lambda x: x.decision_score)
        
        if sell_signals:
            for i, signal in enumerate(sell_signals[:5], 1):
                with st.expander(
                    f"{i}. {signal.news_item.ticker} - {signal.decision.value} ({signal.decision_score:.2f})"
                ):
                    st.write(f"**Title:** {signal.news_item.title}")
                    st.write(f"**Source:** {signal.news_item.source}")
                    sentiment = (
                        signal.news_item.sentiment_label.value
                        if signal.news_item.sentiment_label else 'N/A'
                    )
                    st.write(f"**Sentiment:** {sentiment}")
                    st.write(f"**Reasoning:** {signal.reasoning}")
                    if signal.news_item.url:
                        st.write(f"**Link:** [{signal.news_item.url}]({signal.news_item.url})")
        else:
            st.info("No sell signals found")


def render_fundamental_table(stock_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Render fundamental analysis metrics table.
    
    Args:
        stock_metrics: Dictionary mapping ticker to metrics object
    
    Returns:
        DataFrame with fundamental metrics
    """
    health_data = []
    
    for ticker, metrics in stock_metrics.items():
        # Altman Z interpretation
        z_score = metrics.altman_z_score
        if z_score is not None:
            z_status = (
                "🟢 Safe" if z_score > 2.99
                else ("🟡 Grey Zone" if z_score > 1.81 else "🔴 Distress")
            )
            z_display = f"{z_score:.2f}"
        else:
            z_display, z_status = "N/A", "N/A"
        
        # Beneish M interpretation
        m_score = metrics.beneish_m_score
        if m_score is not None:
            m_status = "🔴 Likely Manipulator" if m_score > -2.22 else "🟢 Unlikely"
            m_display = f"{m_score:.2f}"
        else:
            m_display, m_status = "N/A", "N/A"
        
        # Piotroski F interpretation
        f_score = metrics.piotroski_f_score
        if f_score is not None:
            f_status = (
                "🟢 Strong" if f_score >= 8
                else ("🟡 Moderate" if f_score >= 5 else "🔴 Weak")
            )
            f_display = f"{f_score}/9"
        else:
            f_display, f_status = "N/A", "N/A"
        
        health_data.append({
            'Stock': ticker,
            'Price': f"${metrics.current_price:.2f}" if metrics.current_price else 'N/A',
            'Altman Z': z_display,
            'Z Status': z_status,
            'Beneish M': m_display,
            'M Status': m_status,
            'Piotroski F': f_display,
            'F Status': f_status
        })
    
    df = pd.DataFrame(health_data)
    st.dataframe(df.astype(str), use_container_width=True, hide_index=True)
    
    return df


def render_backtest_signals_table(signals: Any):
    """
    Render backtest signals table with highlighting.
    
    Args:
        signals: DataFrame or list of backtest signals
    """
    signals_df = signals if isinstance(signals, pd.DataFrame) else pd.DataFrame(signals)
    
    def highlight_signals(val):
        return get_signal_style(val)
    
    if 'signal' in signals_df.columns:
        st.dataframe(
            signals_df.style.map(highlight_signals, subset=['signal']),
            use_container_width=True
        )
    else:
        st.dataframe(signals_df, use_container_width=True)
