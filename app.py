# Centurion Capital LLC

"""
Streamlit UI for the Algorithmic Trading.

run the file with: streamlit run app.py
manual stock tickers: RGTI, QUBT, QBTS, IONQ
"""

import streamlit as st
import asyncio
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.StreamHandler()]
    )
logger = logging.getLogger(__name__)

from config import Config

# Import trading strategies
from trading_strategies import STRATEGY_MAP, list_strategies, get_strategy
from main import AlgoTradingSystem
from models import TradingSignal, DecisionTag
from storage import StorageManager
from utils import parse_ticker_csv, validate_tickers, create_sample_csv


# Page configuration
st.set_page_config(
    page_title="Algo Trading Alert System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #000000;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    /* Green Run Analysis button */
    [data-testid="stSidebar"] button[kind="primary"],
    .stButton > button[kind="primary"] {
        background-color: #00cc44 !important;
        border-color: #00cc44 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background-color: #00aa33 !important;
        border-color: #00aa33 !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'signals' not in st.session_state:
        st.session_state.signals = []
    if 'tickers' not in st.session_state:
        st.session_state.tickers = Config.DEFAULT_TICKERS
    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = []
    if 'ticker_mode' not in st.session_state:
        st.session_state.ticker_mode = "Default Tickers"
    # Backtesting state
    if 'backtest_result' not in st.session_state:
        st.session_state.backtest_result = None
    if 'selected_strategy' not in st.session_state:
        st.session_state.selected_strategy = None
    # Page navigation (button-based, not sidebar)
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'main'


def render_sidebar():
    """Render the sidebar with options and controls."""
    with st.sidebar:
        # Load centurion logo for sidebar
        logo_path = Path(__file__).parent / "centurion_logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=40)
        st.title("Command Center")
        
        st.markdown("---")
        
        # Ticker Selection Mode
        st.subheader("📊 Select Stocks")
        ticker_mode = st.radio(
            "Choose input method:",
            ["Default Tickers", "Manual Entry", "Upload CSV"],
            help="Select how you want to specify the stocks to analyze"
        )
        
        tickers = []
        
        if ticker_mode == "Default Tickers":
            st.info(f"Using {len(Config.DEFAULT_TICKERS)} default tickers")
            with st.expander("View default tickers"):
                st.write(", ".join(Config.DEFAULT_TICKERS))
            tickers = Config.DEFAULT_TICKERS
        
        elif ticker_mode == "Manual Entry":
            ticker_input = st.text_area(
                "Enter tickers (comma-separated):",
                value="GOOGL, TSLA",
                height=100,
                help="Enter stock ticker symbols separated by commas"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
            st.success(f"✓ {len(tickers)} ticker(s) entered")
        
        elif ticker_mode == "Upload CSV":
            st.info("Upload a CSV file with ticker symbols")
            
            # Show sample format
            with st.expander("📄 View CSV format example"):
                st.code(create_sample_csv(), language="csv")
                
                # Download sample CSV
                st.download_button(
                    label="⬇️ Download Sample CSV",
                    data=create_sample_csv(),
                    file_name="sample_tickers.csv",
                    mime="text/csv"
                )
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file containing stock ticker symbols"
            )
            
            if uploaded_file is not None:
                try:
                    file_content = uploaded_file.getvalue().decode('utf-8')
                    parsed_tickers = parse_ticker_csv(file_content)
                    
                    if parsed_tickers:
                        valid_tickers, invalid_tickers = validate_tickers(parsed_tickers)
                        
                        st.success(f"✓ Found {len(valid_tickers)} valid ticker(s)")
                        
                        if invalid_tickers:
                            st.warning(f"⚠️ Skipped {len(invalid_tickers)} invalid ticker(s)")
                            with st.expander("View invalid tickers"):
                                st.write(", ".join(invalid_tickers))
                        
                        with st.expander("View uploaded tickers"):
                            st.write(", ".join(valid_tickers))
                        
                        tickers = valid_tickers
                    else:
                        st.error("❌ No valid tickers found in CSV")
                except Exception as e:
                    logger.error(f"Error parsing CSV: {e}")
                    st.error(f"❌ Error parsing CSV: {e}")
        
        st.session_state.tickers = tickers
        st.session_state.ticker_mode = ticker_mode
        
        st.markdown("---")
        
        # Analysis Settings
        st.subheader("⚙️ Settings")
        
        output_format = st.selectbox(
            "Output format:",
            ["Excel (.xlsx)", "CSV (.csv)"],
            help="Choose the output file format"
        )
        
        # Save location settings
        use_custom_path = st.checkbox(
            "Use custom save location",
            value=False,
            help="Choose a custom directory to save the output file"
        )
        
        if use_custom_path:
            custom_path = st.text_input(
                "Custom save path:",
                value=str(Path.cwd()),
                help="Enter the full directory path where you want to save the file"
            )
            filename = st.text_input(
                "Filename:",
                value="daily_stock_news",
                help="Enter the filename (without extension)"
            )
            extension = ".xlsx" if output_format == "Excel (.xlsx)" else ".csv"
            full_path = Path(custom_path) / f"{filename}{extension}"
            Config.OUTPUT_FILE = str(full_path)
            st.info(f"📁 File will be saved to: `{full_path}`")
        else:
            default_filename = "daily_stock_news.xlsx" if output_format == "Excel (.xlsx)" else "daily_stock_news.csv"
            Config.OUTPUT_FILE = default_filename
            default_path = Path.cwd() / default_filename
            st.info(f"📁 File will be saved to: `{default_path}`")
        
        append_mode = st.checkbox(
            "Append to existing file",
            value=Config.APPEND_MODE,
            help="Append results to existing file instead of overwriting"
        )
        Config.APPEND_MODE = append_mode
        
        st.markdown("---")
        
        # Run Analysis Button
        run_button = st.button(
            "Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=len(tickers) == 0
        )
        
        if run_button and len(tickers) > 0:
            st.session_state.analysis_complete = False
            st.session_state.signals = []
            st.session_state.progress_messages = []
            return True

        st.markdown("---")
        return False


def render_header():
    """Render the main header."""
    import base64
    
    # Load and encode logo
    logo_path = Path(__file__).parent / "centurion_logo.png"
    logo_html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" style="height: 3rem; vertical-align: middle; margin-right: 0.2rem;">'
    
    st.markdown(
        f'<div class="main-header">{logo_html}Centurion Capital LLC</div>'
        f'<p class="main-subtitle">Enterprise AI Platform for Event-Driven Alpha Signal Extraction</p>',
        unsafe_allow_html=True
    )


def render_stocks_being_analyzed():
    """Display the stocks currently selected for analysis."""
    if not st.session_state.tickers:
        return
    
    tickers = st.session_state.tickers
    ticker_mode = st.session_state.get('ticker_mode', 'Default Tickers')
    
    # Source label
    source_icons = {
        "Default Tickers": "📋",
        "Manual Entry": "✏️",
        "Upload CSV": "📁"
    }
    icon = source_icons.get(ticker_mode, "📊")
    
    st.subheader(f"{icon} Stocks to Analyze ({ticker_mode})")
    
    # Display tickers as tags in a flowing layout
    ticker_tags = " ".join([f"`{ticker}`" for ticker in tickers])
    st.markdown(ticker_tags)
    
    st.caption(f"Total: {len(tickers)} stock(s)")


def render_simple_summary_table(signals):
    """Render a simple summary table with key signal information aggregated by stock and source."""
    if not signals:
        return
    
    st.subheader("📰 News by Source")
    
    # Build summary data grouped by stock and source
    from collections import defaultdict
    
    # Group signals by (stock, source)
    grouped = defaultdict(list)
    stock_prices = {}
    
    for signal in signals:
        key = (signal.news_item.ticker, signal.news_item.source)
        grouped[key].append(signal)
        # Store price for each stock
        if signal.metrics and signal.metrics.current_price:
            stock_prices[signal.news_item.ticker] = signal.metrics.current_price
    
    summary_data = []
    for (ticker, source), group_signals in grouped.items():
        # Calculate average score for this stock-source combination
        avg_score = sum(s.decision_score for s in group_signals) / len(group_signals)
        
        # Get the most common decision
        decisions = [s.decision.value for s in group_signals]
        most_common_decision = max(set(decisions), key=decisions.count)
        
        # Get the most common sentiment
        sentiments = [s.news_item.sentiment_label.value if s.news_item.sentiment_label else 'neutral' for s in group_signals]
        most_common_sentiment = max(set(sentiments), key=sentiments.count)
        
        # Determine signal emoji based on avg score
        if most_common_decision == 'STRONG_BUY':
            signal_emoji = '🟢🟢'
        elif most_common_decision == 'BUY':
            signal_emoji = '🟢'
        elif most_common_decision == 'HOLD':
            signal_emoji = '🟡'
        elif most_common_decision == 'SELL':
            signal_emoji = '🔴'
        else:  # STRONG_SELL
            signal_emoji = '🔴🔴'
        
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
    
    # Sort by Stock, then by Avg Score (best signals first)
    df = df.sort_values(['Stock', 'Avg Score'], ascending=[True, False])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Also show aggregated view per stock
    st.subheader("📈 Overall Stock Signals")
    
    stock_summary = defaultdict(lambda: {'scores': [], 'decisions': [], 'sentiments': [], 'price': None})
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
        most_common_sentiment = max(set(data['sentiments']), key=data['sentiments'].count) if data['sentiments'] else 'neutral'
        
        if most_common_decision == 'STRONG_BUY':
            signal_emoji = '🟢🟢'
        elif most_common_decision == 'BUY':
            signal_emoji = '🟢'
        elif most_common_decision == 'HOLD':
            signal_emoji = '🟡'
        elif most_common_decision == 'SELL':
            signal_emoji = '🔴'
        else:
            signal_emoji = '🔴🔴'
        
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
    
    st.dataframe(stock_df, use_container_width=True, hide_index=True)


def render_metrics_cards(signals):
    """Render metric cards with key statistics."""
    if not signals:
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Count decisions
    decision_counts = {}
    for signal in signals:
        decision = signal.decision.value
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    with col1:
        st.metric(
            "🚀 STRONG BUY",
            decision_counts.get('STRONG_BUY', 0),
            help="Stocks with strong buy signals"
        )
    
    with col2:
        st.metric(
            "📈 BUY",
            decision_counts.get('BUY', 0),
            help="Stocks with buy signals"
        )
    
    with col3:
        st.metric(
            "⏸️ HOLD",
            decision_counts.get('HOLD', 0),
            help="Stocks to hold"
        )
    
    with col4:
        st.metric(
            "📉 SELL",
            decision_counts.get('SELL', 0),
            help="Stocks with sell signals"
        )
    
    with col5:
        st.metric(
            "⚠️ STRONG SELL",
            decision_counts.get('STRONG_SELL', 0),
            help="Stocks with strong sell signals"
        )


def render_decision_chart(signals):
    """Render decision distribution chart with stock names."""
    if not signals:
        return
    
    st.subheader("📊 Decisions")
    
    # Group stocks by decision
    from collections import defaultdict
    decision_stocks = defaultdict(set)
    
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
        color_discrete_map={
            'STRONG_BUY': '#00cc44',
            'BUY': '#66ff99',
            'HOLD': '#ffcc00',
            'SELL': '#ff9933',
            'STRONG_SELL': '#ff3333'
        },
        hover_data=['Stocks']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show breakdown below the chart
    st.markdown("**Stocks by Decision:**")
    
    col1, col2, col3 = st.columns(3)
    
    # Buy signals
    buy_stocks = decision_stocks.get('STRONG_BUY', set()) | decision_stocks.get('BUY', set())
    with col1:
        st.markdown("🟢 **BUY**")
        if buy_stocks:
            for stock in sorted(buy_stocks):
                # Find decision type for this stock
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


def render_sentiment_chart(signals):
    """Render sentiment analysis chart."""
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
            color_discrete_map={
                'positive': '#00cc44',
                'neutral': '#ffcc00',
                'negative': '#ff3333'
            },
            labels={'Confidence': 'Confidence Score'}
        )
        
        fig.update_layout(showlegend=True, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def render_fundamental_analysis(signals):
    """Render fundamental analysis metrics (Altman Z, Beneish M, Piotroski F scores)."""
    if not signals:
        return
    
    st.subheader("📊 Fundamental Analysis Metrics")
    
    st.markdown("""
    **Score Interpretations:**
    - **Altman Z-Score**: Bankruptcy risk (>2.99 Safe, 1.81-2.99 Grey Zone, <1.81 Distress)
    - **Beneish M-Score**: Earnings manipulation detection (>-2.22 Likely manipulator)
    - **Piotroski F-Score**: Financial health 0-9 (8-9 Strong, 5-7 Moderate, 0-4 Weak)
    """)
    
    # Group by stock to avoid duplicates
    from collections import defaultdict
    stock_metrics = {}
    
    for signal in signals:
        ticker = signal.news_item.ticker
        if ticker not in stock_metrics and signal.metrics:
            stock_metrics[ticker] = signal.metrics
    
    # Build table data
    health_data = []
    for ticker, metrics in stock_metrics.items():
        # Altman Z interpretation
        z_score = metrics.altman_z_score
        if z_score is not None:
            if z_score > 2.99:
                z_status = "🟢 Safe"
            elif z_score > 1.81:
                z_status = "🟡 Grey Zone"
            else:
                z_status = "🔴 Distress"
            z_display = f"{z_score:.2f} ({z_status})"
        else:
            z_display = "N/A"
        
        # Beneish M interpretation
        m_score = metrics.beneish_m_score
        if m_score is not None:
            if m_score > -2.22:
                m_status = "🔴 Likely Manipulator"
            else:
                m_status = "🟢 Unlikely"
            m_display = f"{m_score:.2f} ({m_status})"
        else:
            m_display = "N/A"
        
        # Piotroski F interpretation
        f_score = metrics.piotroski_f_score
        if f_score is not None:
            if f_score >= 8:
                f_status = "🟢 Strong"
            elif f_score >= 5:
                f_status = "🟡 Moderate"
            else:
                f_status = "🔴 Weak"
            f_display = f"{f_score}/9 ({f_status})"
        else:
            f_display = "N/A"
        
        health_data.append({
            'Stock': ticker,
            'Altman Z-Score': z_display,
            'Beneish M-Score': m_display,
            'Piotroski F-Score': f_display,
            'Price': f"${metrics.current_price:.2f}" if metrics.current_price else 'N/A'
        })
    
    if health_data:
        df = pd.DataFrame(health_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Visual chart for F-Scores
        f_scores = []
        for ticker, metrics in stock_metrics.items():
            if metrics.piotroski_f_score is not None:
                f_scores.append({
                    'Stock': ticker,
                    'F-Score': metrics.piotroski_f_score,
                    'Health': 'Strong' if metrics.piotroski_f_score >= 8 else ('Moderate' if metrics.piotroski_f_score >= 5 else 'Weak')
                })
        
        if f_scores:
            f_df = pd.DataFrame(f_scores)
            fig = px.bar(
                f_df,
                x='Stock',
                y='F-Score',
                color='Health',
                title='Piotroski F-Score by Stock (Financial Health)',
                color_discrete_map={
                    'Strong': '#00cc44',
                    'Moderate': '#ffcc00',
                    'Weak': '#ff3333'
                }
            )
            fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Moderate threshold")
            fig.add_hline(y=8, line_dash="dash", line_color="green", annotation_text="Strong threshold")
            fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fundamental analysis data available for the analyzed stocks.")


def render_score_distribution(signals):
    """Render decision score distribution."""
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
        size=[10]*len(df),
        title='Decision Scores by Stock',
        color_discrete_map={
            'STRONG_BUY': '#00cc44',
            'BUY': '#66ff99',
            'HOLD': '#ffcc00',
            'SELL': '#ff9933',
            'STRONG_SELL': '#ff3333'
        }
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


def render_signals_table(signals):
    """Render detailed signals table."""
    if not signals:
        return
    
    st.subheader("📋 Detailed Analysis Results")
    
    # Convert signals to DataFrame
    data = []
    for signal in signals:
        data.append({
            'Ticker': signal.news_item.ticker,
            'Decision': signal.decision.value,
            'Score': f"{signal.decision_score:.2f}",
            'Sentiment': signal.news_item.sentiment_label.value if signal.news_item.sentiment_label else 'N/A',
            'Confidence': f"{signal.news_item.sentiment_confidence:.1%}" if signal.news_item.sentiment_confidence else 'N/A',
            'Price': f"${signal.metrics.current_price:.2f}" if signal.metrics and signal.metrics.current_price else 'N/A',
            'RSI': f"{signal.metrics.rsi:.1f}" if signal.metrics and signal.metrics.rsi else 'N/A',
            'Z-Score': f"{signal.metrics.altman_z_score:.2f}" if signal.metrics and signal.metrics.altman_z_score else 'N/A',
            'F-Score': signal.metrics.piotroski_f_score if signal.metrics and signal.metrics.piotroski_f_score is not None else 'N/A',
            'Source': signal.news_item.source,
            'Title': signal.news_item.title[:50] + '...' if len(signal.news_item.title) > 50 else signal.news_item.title
        })
    
    df = pd.DataFrame(data)
    
    # Apply styling
    def highlight_decision(val):
        if val == 'STRONG_BUY':
            return 'background-color: #00cc44; color: white; font-weight: bold'
        elif val == 'BUY':
            return 'background-color: #66ff99'
        elif val == 'HOLD':
            return 'background-color: #ffcc00'
        elif val == 'SELL':
            return 'background-color: #ff9933'
        elif val == 'STRONG_SELL':
            return 'background-color: #ff3333; color: white; font-weight: bold'
        return ''
    
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


def render_top_signals(signals):
    """Render top buy and sell signals."""
    if not signals:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔝 Top Buy Signals")
        buy_signals = [s for s in signals if s.decision.value in ['STRONG_BUY', 'BUY']]
        buy_signals.sort(key=lambda x: x.decision_score, reverse=True)
        
        if buy_signals:
            for i, signal in enumerate(buy_signals[:5], 1):
                with st.expander(f"{i}. {signal.news_item.ticker} - {signal.decision.value} ({signal.decision_score:.2f})"):
                    st.write(f"**Title:** {signal.news_item.title}")
                    st.write(f"**Source:** {signal.news_item.source}")
                    st.write(f"**Sentiment:** {signal.news_item.sentiment_label.value if signal.news_item.sentiment_label else 'N/A'}")
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
                with st.expander(f"{i}. {signal.news_item.ticker} - {signal.decision.value} ({signal.decision_score:.2f})"):
                    st.write(f"**Title:** {signal.news_item.title}")
                    st.write(f"**Source:** {signal.news_item.source}")
                    st.write(f"**Sentiment:** {signal.news_item.sentiment_label.value if signal.news_item.sentiment_label else 'N/A'}")
                    st.write(f"**Reasoning:** {signal.reasoning}")
                    if signal.news_item.url:
                        st.write(f"**Link:** [{signal.news_item.url}]({signal.news_item.url})")
        else:
            st.info("No sell signals found")


async def run_analysis_async(tickers):
    """Run the analysis asynchronously."""
    system = AlgoTradingSystem(tickers=tickers)
    
    # Progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with st.spinner('🔄 Analyzing...'):
        # Fetch news
        status_placeholder.info("📰 Scraping news from multiple sources...")
        all_news = await system.news_aggregator.fetch_news_for_tickers(tickers)
        progress_placeholder.progress(20)
        
        if not all_news:
            status_placeholder.warning("⚠️ No news found")
            return []
        
        status_placeholder.success(f"✓ Collected {len(all_news)} news items")
        
        # Analyze sentiment
        status_placeholder.info("🧠 Analyzing sentiment...")
        analyzed_news = system.sentiment_analyzer.analyze_news_items(all_news)
        progress_placeholder.progress(40)
        status_placeholder.success(f"✓ Analyzed {len(analyzed_news)} items")
        
        # Calculate metrics and generate signals
        status_placeholder.info("📊 Calculating metrics and generating signals...")
        signals = []
        
        for i, news_item in enumerate(analyzed_news):
            metrics = system.metrics_calculator.get_stock_metrics(news_item.ticker)
            signal = system.decision_engine.generate_signal(news_item, metrics)
            signals.append(signal)
            
            # Update progress
            progress = 40 + int((i + 1) / len(analyzed_news) * 50)
            progress_placeholder.progress(min(progress, 90))
        
        status_placeholder.success(f"✓ Generated {len(signals)} trading signals")
        
        # Save results
        status_placeholder.info("💾 Saving results...")
        save_path = system.storage_manager.save_signals(signals, append=Config.APPEND_MODE)
        progress_placeholder.progress(100)
        
        # Show save location
        if save_path:
            status_placeholder.success(f"✅ Analysis complete! Results saved to: `{save_path}`")
        else:
            status_placeholder.success("✅ Analysis complete!")
        
        return signals


def render_fundamental_page():
    """Render the Fundamental Analysis page (button-based navigation)."""
    from collections import defaultdict
    
    # Back button
    if st.button("← Back to Main", key="back_from_fundamental"):
        st.session_state.current_page = 'main'
        st.rerun()
    
    st.markdown("---")
    
    # Page header
    logo_path = Path(__file__).parent / "centurion_logo.png"
    logo_html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" style="height: 2.5rem; vertical-align: middle; margin-right: 0.5rem;">'
    
    st.markdown(
        f'<div class="main-header">{logo_html}Fundamental Analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Altman Z-Score • Beneish M-Score • Piotroski F-Score"
        "</p>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    signals = st.session_state.get('signals', [])
    
    if not signals:
        st.warning("⚠️ No analysis data available.")
        st.info("""
        **To view fundamental analysis:**
        1. Click **Back to Main**
        2. Select your stocks to analyze
        3. Click **Run Analysis**
        4. Return to this page to view detailed fundamental metrics
        """)
        return
    
    # Score interpretations
    st.markdown("""
    ### 📖 Score Interpretations
    
    | Score | What it Measures | Interpretation |
    |-------|------------------|----------------|
    | **Altman Z-Score** | Bankruptcy risk | >2.99 Safe, 1.81-2.99 Grey Zone, <1.81 Distress |
    | **Beneish M-Score** | Earnings manipulation | >-2.22 Likely manipulator, <-2.22 Unlikely |
    | **Piotroski F-Score** | Financial health (0-9) | 8-9 Strong, 5-7 Moderate, 0-4 Weak |
    """)
    
    st.markdown("---")
    
    # Group by stock to avoid duplicates
    stock_metrics = {}
    for signal in signals:
        ticker = signal.news_item.ticker
        if ticker not in stock_metrics and signal.metrics:
            stock_metrics[ticker] = signal.metrics
    
    if not stock_metrics:
        st.info("No fundamental data available for the analyzed stocks.")
        return
    
    # Build table data
    health_data = []
    for ticker, metrics in stock_metrics.items():
        # Altman Z interpretation
        z_score = metrics.altman_z_score
        if z_score is not None:
            z_status = "🟢 Safe" if z_score > 2.99 else ("🟡 Grey Zone" if z_score > 1.81 else "🔴 Distress")
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
            f_status = "🟢 Strong" if f_score >= 8 else ("🟡 Moderate" if f_score >= 5 else "🔴 Weak")
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
    
    # Display main metrics table
    st.subheader("📋 All Stocks Overview")
    df = pd.DataFrame(health_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Three charts side by side
    col1, col2, col3 = st.columns(3)
    
    # Altman Z-Score Chart
    with col1:
        st.subheader("📉 Altman Z-Score")
        z_data = [{'Stock': t, 'Z-Score': m.altman_z_score, 
                   'Risk': 'Safe' if m.altman_z_score > 2.99 else ('Grey Zone' if m.altman_z_score > 1.81 else 'Distress')}
                  for t, m in stock_metrics.items() if m.altman_z_score is not None]
        if z_data:
            z_df = pd.DataFrame(z_data)
            fig = px.bar(z_df, x='Stock', y='Z-Score', color='Risk', title='Bankruptcy Risk',
                        color_discrete_map={'Safe': '#00cc44', 'Grey Zone': '#ffcc00', 'Distress': '#ff3333'})
            fig.add_hline(y=2.99, line_dash="dash", line_color="green", annotation_text="Safe")
            fig.add_hline(y=1.81, line_dash="dash", line_color="red", annotation_text="Distress")
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Z-Score data")
    
    # Beneish M-Score Chart
    with col2:
        st.subheader("🔍 Beneish M-Score")
        m_data = [{'Stock': t, 'M-Score': m.beneish_m_score,
                   'Risk': 'Likely Manipulator' if m.beneish_m_score > -2.22 else 'Unlikely'}
                  for t, m in stock_metrics.items() if m.beneish_m_score is not None]
        if m_data:
            m_df = pd.DataFrame(m_data)
            fig = px.bar(m_df, x='Stock', y='M-Score', color='Risk', title='Manipulation Detection',
                        color_discrete_map={'Likely Manipulator': '#ff3333', 'Unlikely': '#00cc44'})
            fig.add_hline(y=-2.22, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No M-Score data")
    
    # Piotroski F-Score Chart
    with col3:
        st.subheader("💪 Piotroski F-Score")
        f_data = [{'Stock': t, 'F-Score': m.piotroski_f_score,
                   'Health': 'Strong' if m.piotroski_f_score >= 8 else ('Moderate' if m.piotroski_f_score >= 5 else 'Weak')}
                  for t, m in stock_metrics.items() if m.piotroski_f_score is not None]
        if f_data:
            f_df = pd.DataFrame(f_data)
            fig = px.bar(f_df, x='Stock', y='F-Score', color='Health', title='Financial Health',
                        color_discrete_map={'Strong': '#00cc44', 'Moderate': '#ffcc00', 'Weak': '#ff3333'})
            fig.add_hline(y=8, line_dash="dash", line_color="green", annotation_text="Strong")
            fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Moderate")
            fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 10], height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No F-Score data")
    
    st.markdown("---")
    
    # Summary metrics
    st.subheader("📊 Summary")
    safe_count = sum(1 for m in stock_metrics.values() if m.altman_z_score and m.altman_z_score > 2.99)
    strong_count = sum(1 for m in stock_metrics.values() if m.piotroski_f_score and m.piotroski_f_score >= 8)
    clean_count = sum(1 for m in stock_metrics.values() if m.beneish_m_score and m.beneish_m_score <= -2.22)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Safe (Z-Score)", f"{safe_count}/{len(stock_metrics)}")
    with col2:
        st.metric("💪 Strong (F-Score)", f"{strong_count}/{len(stock_metrics)}")
    with col3:
        st.metric("✅ Clean (M-Score)", f"{clean_count}/{len(stock_metrics)}")


def render_backtesting_page():
    """Render the Strategy Backtesting page (button-based navigation)."""
    import json
    from datetime import timedelta
    
    # Back button
    if st.button("← Back to Main", key="back_from_backtest"):
        st.session_state.current_page = 'main'
        st.rerun()
    
    st.markdown("---")
    
    # Page header
    logo_path = Path(__file__).parent / "centurion_logo.png"
    logo_html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" style="height: 2.5rem; vertical-align: middle; margin-right: 0.5rem;">'
    
    st.markdown(
        f'<div class="main-header">{logo_html}Strategy Backtesting</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Test and analyze trading strategies with historical data"
        "</p>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Get available strategies
    strategies = list_strategies()
    strategy_options = {s['name']: s for s in strategies}
    
    # Two-column layout
    config_col, results_col = st.columns([1, 2])
    
    with config_col:
        st.subheader("⚙️ Configuration")
        
        # Strategy category filter
        categories = sorted(list(set(s['category'] for s in strategies)))
        selected_category = st.selectbox(
            "Strategy Category",
            options=["All"] + categories,
            help="Filter strategies by category"
        )
        
        # Filter strategies by category
        if selected_category != "All":
            filtered_strategies = {k: v for k, v in strategy_options.items() if v['category'] == selected_category}
        else:
            filtered_strategies = strategy_options
        
        # Strategy selection
        selected_name = st.selectbox(
            "Select Strategy",
            options=sorted(filtered_strategies.keys()),
            help="Choose a backtesting strategy"
        )
        
        if selected_name:
            strategy_info = filtered_strategies[selected_name]
            st.caption(strategy_info['description'])
            
            # Get strategy class and parameters
            strategy_cls = get_strategy(strategy_info['id'])
            params = strategy_cls.get_parameters()
            
            st.markdown("---")
            st.subheader("📊 Parameters")
            
            # Dynamic parameter inputs
            param_values = {}
            for param_name, param_config in params.items():
                param_type = param_config.get('type', 'float')
                default = param_config.get('default')
                description = param_config.get('description', '')
                
                if param_type == 'int':
                    param_values[param_name] = st.number_input(
                        param_name.replace('_', ' ').title(),
                        value=int(default) if default else 14,
                        step=1,
                        help=description
                    )
                elif param_type == 'float':
                    param_values[param_name] = st.number_input(
                        param_name.replace('_', ' ').title(),
                        value=float(default) if default else 0.0,
                        format="%.4f",
                        help=description
                    )
                elif param_type == 'str':
                    param_values[param_name] = st.text_input(
                        param_name.replace('_', ' ').title(),
                        value=str(default) if default else '',
                        help=description
                    )
            
            st.markdown("---")
            st.subheader("📅 Data Settings")
            
            # Ticker input
            if strategy_info['id'] == 'pairs_trading':
                t1, t2 = st.columns(2)
                with t1:
                    ticker1 = st.text_input("Ticker 1", value="GLD")
                with t2:
                    ticker2 = st.text_input("Ticker 2", value="SLV")
                param_values['tickers'] = [ticker1.upper(), ticker2.upper()] if ticker1 and ticker2 else []
            else:
                ticker = st.text_input("Ticker Symbol", value="AAPL")
                param_values['tickers'] = [ticker.upper()] if ticker else ["AAPL"]
            
            period = st.selectbox("Data Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            end_date = datetime.now()
            period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
            start_date = end_date - timedelta(days=period_days.get(period, 365))
            param_values['start_date'] = start_date.strftime('%Y-%m-%d')
            param_values['end_date'] = end_date.strftime('%Y-%m-%d')
            
            capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)
            param_values['capital'] = float(capital)
            
            st.markdown("---")
            run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
            
            if run_backtest:
                with st.spinner("Running backtest..."):
                    try:
                        strategy = strategy_cls()
                        result = strategy.run(**param_values)
                        st.session_state.backtest_result = result
                        st.session_state.selected_strategy = selected_name
                        if result.success:
                            st.success("✅ Backtest completed!")
                        else:
                            st.error(f"❌ Failed: {result.error_message}")
                    except Exception as e:
                        logger.error(f"Error running backtest: {e}")
                        st.error(f"❌ Error: {str(e)}")
    
    with results_col:
        st.subheader("📈 Results")
        result = st.session_state.backtest_result
        
        if result is None:
            st.info("👈 Configure a strategy and click **Run Backtest** to see results")
        elif not result.success:
            st.error(f"❌ Backtest failed: {result.error_message}")
        else:
            # Display metrics
            if result.metrics:
                st.markdown("#### Performance Metrics")
                flat_metrics = {}
                for key, value in result.metrics.items():
                    if isinstance(value, dict):
                        if key == 'aggregate':
                            for sk, sv in value.items():
                                flat_metrics[f"Avg {sk.replace('_', ' ').title()}"] = sv
                        else:
                            for sk, sv in value.items():
                                if sk in ['total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades']:
                                    flat_metrics[f"{key} {sk.replace('_', ' ').title()}"] = sv
                    else:
                        flat_metrics[key.replace('_', ' ').title()] = value
                
                priority_keys = ['total_return', 'avg_return', 'sharpe_ratio', 'max_drawdown']
                priority_metrics = [(k, v) for k, v in flat_metrics.items() if any(pk in k.lower() for pk in priority_keys)]
                other_metrics = [(k, v) for k, v in flat_metrics.items() if not any(pk in k.lower() for pk in priority_keys)]
                
                if priority_metrics:
                    cols = st.columns(min(4, len(priority_metrics)))
                    for i, (name, val) in enumerate(priority_metrics[:4]):
                        with cols[i]:
                            if isinstance(val, float):
                                if 'return' in name.lower() or 'drawdown' in name.lower():
                                    st.metric(name, f"{val:.2%}")
                                else:
                                    st.metric(name, f"{val:.4f}")
                            else:
                                st.metric(name, str(val))
                
                if other_metrics:
                    cols2 = st.columns(min(4, len(other_metrics)))
                    for i, (name, val) in enumerate(other_metrics[:4]):
                        with cols2[i]:
                            st.metric(name, f"{val:.4f}" if isinstance(val, float) else str(val))
            
            st.markdown("---")
            
            # Charts
            if result.charts:
                st.markdown("#### Charts")
                for chart in result.charts:
                    st.caption(chart.title)
                    if chart.chart_type == 'matplotlib':
                        try:
                            img_data = chart.data.split(',', 1)[1] if chart.data.startswith('data:') else chart.data
                            st.image(base64.b64decode(img_data), use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display chart: {e}")
                    elif chart.chart_type == 'plotly':
                        try:
                            fig = go.Figure(json.loads(chart.data))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display chart: {e}")
            
            # Tables
            if result.tables:
                st.markdown("---")
                st.markdown("#### Data Tables")
                for table in result.tables:
                    st.caption(table.title)
                    if table.data:
                        st.dataframe(pd.DataFrame(table.data), use_container_width=True)
            
            # Signals
            has_signals = result.signals is not None and (
                (isinstance(result.signals, pd.DataFrame) and not result.signals.empty) or
                (isinstance(result.signals, list) and len(result.signals) > 0)
            )
            if has_signals:
                st.markdown("---")
                st.markdown("#### Trading Signals")
                signals_df = result.signals if isinstance(result.signals, pd.DataFrame) else pd.DataFrame(result.signals)
                
                def highlight_signals(val):
                    if val == 'BUY': return 'background-color: #90EE90'
                    elif val == 'SELL': return 'background-color: #FFB6C1'
                    return ''
                
                if 'signal' in signals_df.columns:
                    st.dataframe(signals_df.style.map(highlight_signals, subset=['signal']), use_container_width=True)
                else:
                    st.dataframe(signals_df, use_container_width=True)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Render sidebar and check if analysis should run
    should_run = render_sidebar()
    
    # Check current page and render accordingly
    current_page = st.session_state.get('current_page', 'main')
    
    if current_page == 'fundamental':
        render_fundamental_page()
        return
    
    if current_page == 'backtesting':
        render_backtesting_page()
        return
    
    # Main page
    render_header()
    
    # Main content area
    if should_run and st.session_state.tickers:
        # Run analysis
        st.session_state.signals = asyncio.run(run_analysis_async(st.session_state.tickers))
        st.session_state.analysis_complete = True
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.signals:
        st.markdown("---")
        
        # Show stocks that were analyzed
        render_stocks_being_analyzed()
        
        # Navigation buttons to other pages
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        with btn_col1:
            if st.button("📊 Fundamental Analysis", key="results_fundamental", use_container_width=True, type="secondary"):
                st.session_state.current_page = 'fundamental'
                st.rerun()
        with btn_col2:
            if st.button("🔬 Backtest Strategy", key="results_backtest", use_container_width=True, type="secondary"):
                st.session_state.current_page = 'backtesting'
                st.rerun()
        
        st.markdown("---")
        
        # Simple summary table at the top
        render_simple_summary_table(st.session_state.signals)
        
        st.markdown("---")
        
        # Metrics cards
        render_metrics_cards(st.session_state.signals)
        
        st.markdown("---")
        
        # Charts in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Detailed Table", "🔝 Top Signals", "📈 Sentiment Charts"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                render_decision_chart(st.session_state.signals)
            with col2:
                render_score_distribution(st.session_state.signals)
        
        with tab2:
            render_signals_table(st.session_state.signals)
        
        with tab3:
            render_top_signals(st.session_state.signals)
        
        with tab4:
            render_sentiment_chart(st.session_state.signals)
    
    elif not st.session_state.analysis_complete:
        # Welcome screen
        st.markdown("---")
        
        st.info("👈 Configure your settings in the sidebar and click **Run Analysis** to start")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://img.icons8.com/clouds/400/000000/stocks.png", width=300)
        
        st.markdown("### 🎯 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📰 Multi-Source News**")
            st.caption("Aggregates from Yahoo Finance, Finviz, Investing.com, and more")
        
        with col2:
            st.markdown("**🧠 AI Sentiment Analysis**")
            st.caption("DistilBERT-powered sentiment classification")
        
        with col3:
            st.markdown("**📊 Comprehensive Metrics**")
            st.caption("Fundamentals + Technicals analysis")
        
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. **Select Input Method**: Choose default tickers, enter manually, or upload a CSV
        2. **Configure Settings**: Select output format and append mode
        3. **Run Analysis**: Click the Run Analysis button
        4. **Review Results**: Explore charts, tables, and detailed signal information
        5. **Download Data**: Export results for further analysis
        """)


if __name__ == "__main__":
    main()