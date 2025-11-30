"""
Streamlit UI for the Algo Trading Alert System.

Run with: streamlit run app.py
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import io

from config import Config
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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
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


def render_sidebar():
    """Render the sidebar with options and controls."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/stock-market.png", width=80)
        st.title("🎯 Control Panel")
        
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
                value="AAPL, MSFT, GOOGL, TSLA",
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
                    st.error(f"❌ Error parsing CSV: {e}")
        
        st.session_state.tickers = tickers
        
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
    st.markdown('<div class="main-header">📈 Algo Trading Alert System</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "AI-Powered News-Driven Trading Signal Generator"
        "</p>",
        unsafe_allow_html=True
    )


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
    """Render decision distribution chart."""
    if not signals:
        return
    
    st.subheader("📊 Decision Distribution")
    
    # Count decisions
    decision_counts = {}
    for signal in signals:
        decision = signal.decision.value
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'Decision': list(decision_counts.keys()),
        'Count': list(decision_counts.values())
    })
    
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
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


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


def render_score_distribution(signals):
    """Render decision score distribution."""
    if not signals:
        return
    
    st.subheader("📈 Decision Score Distribution")
    
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
    
    styled_df = df.style.applymap(highlight_decision, subset=['Decision'])
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


def main():
    """Main Streamlit application."""
    initialize_session_state()
    render_header()
    
    # Render sidebar and check if analysis should run
    should_run = render_sidebar()
    
    # Main content area
    if should_run and st.session_state.tickers:
        # Run analysis
        st.session_state.signals = asyncio.run(run_analysis_async(st.session_state.tickers))
        st.session_state.analysis_complete = True
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.signals:
        st.markdown("---")
        
        # Metrics cards
        render_metrics_cards(st.session_state.signals)
        
        st.markdown("---")
        
        # Charts in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Detailed Table", "🔝 Top Signals", "📈 Charts"])
        
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
