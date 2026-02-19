"""
Chart Rendering Module for Centurion Capital LLC.

Contains all visualization and chart rendering functions for Plotly charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
from collections import defaultdict

from ui.styles import DECISION_COLORS, SENTIMENT_COLORS, HEALTH_COLORS
from ui.components import get_decision_emoji


def render_decision_chart(signals: List[Any]):
    """
    Render decision distribution pie chart with stock breakdown.
    
    Args:
        signals: List of TradingSignal objects
    """
    if not signals:
        return
    
    st.subheader("📊 Decisions")
    
    # Group stocks by decision
    decision_stocks: Dict[str, set] = defaultdict(set)
    
    for signal in signals:
        decision = signal.decision.value
        decision_stocks[decision].add(signal.news_item.ticker)
    
    # Create DataFrame with stock lists
    chart_data = []
    for decision, stocks in decision_stocks.items():
        chart_data.append({
            'Decision': decision,
            'Count': len(stocks),
            'Stocks': ', '.join(sorted(stocks))
        })
    
    df = pd.DataFrame(chart_data)
    
    # Create pie chart
    fig = px.pie(
        df,
        values='Count',
        names='Decision',
        title='Trading Decision Distribution',
        color='Decision',
        color_discrete_map=DECISION_COLORS,
        hover_data=['Stocks']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show breakdown below the chart
    _render_decision_breakdown(decision_stocks)


def _render_decision_breakdown(decision_stocks: Dict[str, set]):
    """Render stocks grouped by decision type."""
    st.markdown("**Stocks by Decision:**")
    
    col1, col2, col3 = st.columns(3)
    
    # Buy signals
    buy_stocks = decision_stocks.get('STRONG_BUY', set()) | decision_stocks.get('BUY', set())
    with col1:
        st.markdown("🟢 **BUY**")
        if buy_stocks:
            for stock in sorted(buy_stocks):
                if stock in decision_stocks.get('STRONG_BUY', set()):
                    st.markdown(f"- `{stock}` 🟢🟢")
                else:
                    st.markdown(f"- `{stock}` 🟢")
        else:
            st.caption("None")
    
    # Hold signals
    hold_stocks = decision_stocks.get('HOLD', set())
    with col2:
        st.markdown("🟡 **HOLD**")
        if hold_stocks:
            for stock in sorted(hold_stocks):
                st.markdown(f"- `{stock}`")
        else:
            st.caption("None")
    
    # Sell signals
    sell_stocks = decision_stocks.get('STRONG_SELL', set()) | decision_stocks.get('SELL', set())
    with col3:
        st.markdown("🔴 **SELL**")
        if sell_stocks:
            for stock in sorted(sell_stocks):
                if stock in decision_stocks.get('STRONG_SELL', set()):
                    st.markdown(f"- `{stock}` 🔴🔴")
                else:
                    st.markdown(f"- `{stock}` 🔴")
        else:
            st.caption("None")


def render_sentiment_chart(signals: List[Any]):
    """
    Render sentiment analysis bar chart.
    
    Args:
        signals: List of TradingSignal objects
    """
    if not signals:
        return
    
    st.subheader("🧠 Sentiment Analysis")
    
    # Extract sentiment data
    sentiment_data = []
    for signal in signals:
        if signal.news_item.sentiment_label:
            sentiment_data.append({
                'Ticker': signal.news_item.ticker,
                'Sentiment': signal.news_item.sentiment_label.value,
                'Confidence': signal.news_item.sentiment_confidence or 0
            })
    
    if sentiment_data:
        df = pd.DataFrame(sentiment_data)
        
        # Create bar chart
        fig = px.bar(
            df,
            x='Ticker',
            y='Confidence',
            color='Sentiment',
            title='Sentiment Confidence by Stock',
            color_discrete_map=SENTIMENT_COLORS,
            labels={'Confidence': 'Confidence Score'}
        )
        
        fig.update_layout(showlegend=True, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data available")


def render_score_distribution(signals: List[Any]):
    """
    Render decision score scatter plot.
    
    Args:
        signals: List of TradingSignal objects
    """
    if not signals:
        return
    
    st.subheader("📈 Score Distribution")
    
    # Extract scores
    score_data = []
    for signal in signals:
        score_data.append({
            'Ticker': signal.news_item.ticker,
            'Score': signal.decision_score,
            'Decision': signal.decision.value
        })
    
    df = pd.DataFrame(score_data)
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='Ticker',
        y='Score',
        color='Decision',
        size=[10] * len(df),
        title='Decision Scores by Stock',
        color_discrete_map=DECISION_COLORS
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


def render_fundamental_charts(stock_metrics: Dict[str, Any]):
    """
    Render fundamental analysis charts (Z-Score, M-Score, F-Score).
    
    Args:
        stock_metrics: Dictionary mapping ticker to metrics object
    """
    col1, col2, col3 = st.columns(3)
    
    # Altman Z-Score Chart
    with col1:
        _render_z_score_chart(stock_metrics)
    
    # Beneish M-Score Chart
    with col2:
        _render_m_score_chart(stock_metrics)
    
    # Piotroski F-Score Chart
    with col3:
        _render_f_score_chart(stock_metrics)


def _render_z_score_chart(stock_metrics: Dict[str, Any]):
    """Render Altman Z-Score bar chart."""
    st.subheader("📉 Altman Z-Score")
    
    z_data = []
    for ticker, metrics in stock_metrics.items():
        if metrics.altman_z_score is not None:
            z_score = metrics.altman_z_score
            risk = 'Safe' if z_score > 2.99 else ('Grey Zone' if z_score > 1.81 else 'Distress')
            z_data.append({
                'Stock': ticker,
                'Z-Score': z_score,
                'Risk': risk
            })
    
    if z_data:
        z_df = pd.DataFrame(z_data)
        fig = px.bar(
            z_df,
            x='Stock',
            y='Z-Score',
            color='Risk',
            title='Bankruptcy Risk',
            color_discrete_map=HEALTH_COLORS
        )
        fig.add_hline(y=2.99, line_dash="dash", line_color="green", annotation_text="Safe")
        fig.add_hline(y=1.81, line_dash="dash", line_color="red", annotation_text="Distress")
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Z-Score data")


def _render_m_score_chart(stock_metrics: Dict[str, Any]):
    """Render Beneish M-Score bar chart."""
    st.subheader("🔍 Beneish M-Score")
    
    m_data = []
    for ticker, metrics in stock_metrics.items():
        if metrics.beneish_m_score is not None:
            m_score = metrics.beneish_m_score
            risk = 'Likely Manipulator' if m_score > -2.22 else 'Unlikely'
            m_data.append({
                'Stock': ticker,
                'M-Score': m_score,
                'Risk': risk
            })
    
    if m_data:
        m_df = pd.DataFrame(m_data)
        fig = px.bar(
            m_df,
            x='Stock',
            y='M-Score',
            color='Risk',
            title='Manipulation Detection',
            color_discrete_map=HEALTH_COLORS
        )
        fig.add_hline(y=-2.22, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No M-Score data")


def _render_f_score_chart(stock_metrics: Dict[str, Any]):
    """Render Piotroski F-Score bar chart."""
    st.subheader("💪 Piotroski F-Score")
    
    f_data = []
    for ticker, metrics in stock_metrics.items():
        if metrics.piotroski_f_score is not None:
            f_score = metrics.piotroski_f_score
            health = 'Strong' if f_score >= 8 else ('Moderate' if f_score >= 5 else 'Weak')
            f_data.append({
                'Stock': ticker,
                'F-Score': f_score,
                'Health': health
            })
    
    if f_data:
        f_df = pd.DataFrame(f_data)
        fig = px.bar(
            f_df,
            x='Stock',
            y='F-Score',
            color='Health',
            title='Financial Health',
            color_discrete_map=HEALTH_COLORS
        )
        fig.add_hline(y=8, line_dash="dash", line_color="green", annotation_text="Strong")
        fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Moderate")
        fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 10], height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No F-Score data")


def render_fundamental_summary_metrics(stock_metrics: Dict[str, Any]):
    """
    Render summary metrics for fundamental analysis.
    
    Args:
        stock_metrics: Dictionary mapping ticker to metrics object
    """
    st.subheader("📊 Summary")
    
    total = len(stock_metrics)
    safe_count = sum(1 for m in stock_metrics.values() if m.altman_z_score and m.altman_z_score > 2.99)
    strong_count = sum(1 for m in stock_metrics.values() if m.piotroski_f_score and m.piotroski_f_score >= 8)
    clean_count = sum(1 for m in stock_metrics.values() if m.beneish_m_score and m.beneish_m_score <= -2.22)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Safe (Z-Score)", f"{safe_count}/{total}")
    with col2:
        st.metric("💪 Strong (F-Score)", f"{strong_count}/{total}")
    with col3:
        st.metric("✅ Clean (M-Score)", f"{clean_count}/{total}")
