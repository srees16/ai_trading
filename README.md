# Algo Trading Alert System

A comprehensive Python-based news-driven algorithmic trading alert system that combines web scraping, sentiment analysis, fundamental analysis, and technical analysis to generate trading signals. Features an interactive Streamlit web interface with CSV upload capabilities.

## 🚀 Key Features

### Core Analysis Engine
- **Multi-Source News Scraping**: Aggregates news from Yahoo Finance, Finviz, Investing.com, TradingView, and more
- **AI-Powered Sentiment Analysis**: Uses DistilBERT transformer model for accurate sentiment detection
- **Comprehensive Stock Metrics**:
  - Fundamentals: PEG ratio, ROE, EPS, Free Cash Flow, DCF value, Intrinsic value
  - Technicals: RSI, MACD, Fibonacci retracement, Bollinger Bands, Maximum drawdown
- **Intelligent Decision Engine**: Combines sentiment, fundamentals, and technicals to generate trading decisions (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- **Real-Time Alerts**: Desktop popup notifications for high-confidence trading signals
- **Data Persistence**: Automatic logging to Excel/CSV with daily append functionality
- **Async Architecture**: Concurrent scraping for optimal performance

### Interactive Web Interface
- **Streamlit Dashboard**: Modern, interactive web UI with real-time updates
- **CSV Upload**: Upload custom ticker lists in various formats
- **Visual Analytics**: Interactive charts, pie charts, scatter plots, and bar graphs
- **Color-Coded Results**: Easy-to-read decision indicators
- **Export Capabilities**: Download results as CSV for further analysis
- **Multiple Input Methods**: Default tickers, manual entry, or CSV upload
- **Progress Tracking**: Live updates during analysis execution

## 📁 Project Structure

```
algo_trade/
├── main.py                      # Main orchestration script
├── config.py                    # Configuration settings
├── models.py                    # Data models and interfaces
├── requirements.txt             # Python dependencies
├── scrapers/                    # News scraping modules
│   ├── __init__.py
│   ├── yahoo_finance.py
│   ├── finviz.py
│   ├── investing.py
│   ├── tradingview.py
│   └── aggregator.py
├── sentiment/                   # Sentiment analysis
│   ├── __init__.py
│   └── analyzer.py
├── metrics/                     # Stock metrics calculation
│   ├── __init__.py
│   └── calculator.py
├── decision_engine/             # Trading decision logic
│   ├── __init__.py
│   └── engine.py
├── notifications/               # Alert system
│   ├── __init__.py
│   └── manager.py
└── storage/                     # Data persistence
    ├── __init__.py
    └── manager.py
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Windows, macOS, or Linux

### Setup Steps

1. **Navigate to the project directory**:
   ```powershell
   cd "c:\Users\suraboyi\Videos\Python pet Pdrive\algo_trade"
   ```

2. **Create a virtual environment (recommended)**:
   ```powershell
   python -m venv myenv
   .\myenv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

   **Note**: The first run will download the DistilBERT model (~250MB), which may take a few minutes.

### Dependencies Installed
- **Web Framework**: streamlit, plotly, streamlit-aggrid
- **Data Processing**: pandas, numpy, openpyxl
- **Financial Data**: yfinance
- **Web Scraping**: aiohttp, beautifulsoup4, lxml
- **AI/ML**: transformers, torch
- **Notifications**: plyer

## 📊 Usage

### 🎯 Quick Start - Streamlit Web UI (Recommended)

Launch the interactive web interface:

```powershell
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

**Alternative launch methods:**
```powershell
# Using batch script (Windows)
.\run_streamlit.bat
```

### 🎨 Streamlit Interface Features

#### 1. **Three Input Methods**

**Option A: Default Tickers**
- Pre-configured list of popular stocks
- No setup required
- Good for quick testing

**Option B: Manual Entry**
- Type ticker symbols separated by commas
- Example: `AAPL, MSFT, GOOGL, TSLA`
- Best for small custom lists

**Option C: Upload CSV File** ⭐ **Recommended**
- Click "Choose a CSV file" in sidebar
- Upload your prepared CSV
- Download the sample CSV format if needed

#### 2. **CSV File Format**

Your CSV file can be in any of these formats:

**Simple single column:**
```csv
Ticker
AAPL
MSFT
GOOGL
```

**With company names:**
```csv
Ticker,Company
AAPL,Apple Inc.
MSFT,Microsoft Corporation
GOOGL,Alphabet Inc.
```

**No header (first column used):**
```csv
AAPL
MSFT
GOOGL
```

**Alternative column names:**
- Recognized headers: `Ticker`, `Symbol`, `Stock`, `Tickers`, `Symbols`, `Stocks`
- System auto-detects and uses first appropriate column

**Validation:**
- ✅ Valid: 1-5 characters, alphanumeric with dots/dashes (e.g., `AAPL`, `BRK.B`)
- ❌ Invalid: More than 5 characters, special characters, empty values
- Auto-filters invalid tickers and removes duplicates

#### 3. **Dashboard Components**

**Sidebar Controls:**
- 🔘 Input method selection
- 📁 CSV file uploader  
- ⬇️ Download sample CSV template
- ⚙️ Output format (Excel/CSV)
- 🔄 Append mode toggle
- ▶️ Run Analysis button
- ℹ️ System information

**Main Dashboard Tabs:**

**Tab 1: 📊 Overview**
- 5 metric cards showing signal counts (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- Decision distribution pie chart
- Score distribution scatter plot
- Visual summary of all results

**Tab 2: 📋 Detailed Table**
- Complete data table with all columns
- Color-coded decisions (🟢 Buy, 🟡 Hold, 🔴 Sell)
- Sortable and filterable columns
- Download results as CSV button

**Tab 3: 🔝 Top Signals**
- Top 5 buy opportunities with detailed reasoning
- Top 5 sell warnings with detailed reasoning
- Expandable details for each signal
- Direct links to news articles

**Tab 4: 📈 Charts**
- Sentiment confidence visualization
- Interactive bar charts
- Hover for detailed information

#### 4. **Running Analysis**

1. Select your input method in sidebar
2. Configure output settings (format, append mode)
3. Click **"▶️ Run Analysis"** button
4. Watch real-time progress:
   - 📰 Scraping news (20%)
   - 🧠 Analyzing sentiment (40%)
   - 📊 Calculating metrics (90%)
   - 💾 Saving results (100%)
5. Explore results across all tabs

#### 5. **Color Coding**

| Decision | Color | Badge | Meaning |
|----------|-------|-------|---------|
| STRONG_BUY | 🟢 Bright Green | High confidence | Strong buy signal |
| BUY | 🟢 Light Green | Positive | Buy signal |
| HOLD | 🟡 Yellow | Neutral | Hold position |
| SELL | 🟠 Orange | Negative | Sell signal |
| STRONG_SELL | 🔴 Red | High confidence | Strong sell signal |

### 💻 Command Line Usage (Alternative)

Run the system directly with Python:

```powershell
python main.py
```

This will:
1. Scrape news for default tickers (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM, V, WMT)
2. Analyze sentiment using AI
3. Calculate fundamental and technical metrics
4. Generate trading signals
5. Send popup alerts for strong signals
6. Save results to `daily_stock_news.xlsx`

**Custom Tickers via Code:**

Edit `main.py` to monitor specific stocks:

```python
# In main.py, modify the main() function:
async def main():
    tickers = ["AAPL", "TSLA", "NVDA", "AMD", "INTC"]
    system = AlgoTradingSystem(tickers=tickers)
    await system.run()
```

### 📝 Example Workflows

#### Workflow 1: Tech Sector Analysis

1. Create `tech_stocks.csv`:
```csv
Ticker
AAPL
MSFT
GOOGL
AMZN
META
NVDA
AMD
INTC
ORCL
CSCO
```

2. Run Streamlit: `streamlit run app.py`
3. Upload the CSV file
4. Review results in "Top Signals" tab
5. Download results for Excel analysis

#### Workflow 2: Daily Portfolio Monitoring

1. Export your portfolio to CSV format
2. Upload to Streamlit app
3. Enable "Append to existing file"
4. Run daily analysis
5. Build historical dataset over time

#### Workflow 3: Sector Comparison

1. Create separate CSV files for each sector (tech, finance, energy)
2. Run analysis for each sector
3. Compare decision distributions
4. Identify best-performing sectors

#### Workflow 4: Single Stock Deep Dive

1. Create CSV with single ticker:
```csv
Ticker
TSLA
```
2. Run detailed analysis
3. Review all metrics and news sentiment
4. Check technical indicators and fundamentals

## 📈 Output & Results

### 📊 Streamlit Dashboard Output

The web interface provides comprehensive visual analytics:

**Metrics Cards (Top of Page):**
- 🚀 STRONG_BUY count with green badge
- 📈 BUY count with light green badge
- ⏸️ HOLD count with yellow badge
- 📉 SELL count with orange badge
- ⚠️ STRONG_SELL count with red badge

**Overview Tab:**
- Pie chart showing decision distribution across all stocks
- Scatter plot of decision scores by ticker
- Visual summary of portfolio analysis

**Detailed Table Tab:**
- Full data table with columns: Ticker, Decision, Score, Sentiment, Confidence, Price, RSI, MACD, PEG, ROE, Source, Title
- Color-coded decision badges for quick scanning
- Sortable columns for custom analysis
- CSV download button for Excel export
- Comprehensive metrics in one view

**Top Signals Tab:**
- **Top 5 Buy Signals**: Best opportunities with full reasoning
- **Top 5 Sell Signals**: Strongest warnings with detailed analysis
- Expandable sections showing:
  - News title and source
  - Sentiment analysis results
  - Fundamental metrics (PEG, ROE, EPS)
  - Technical indicators (RSI, MACD)
  - Combined decision score
  - Direct link to news article

**Charts Tab:**
- Sentiment confidence bar chart by stock
- Interactive hover details
- Color-coded by sentiment (positive/neutral/negative)
- Visual pattern recognition

### 💻 Console Output

When running via command line, the system provides real-time progress updates:
   tickers = df['Ticker'].tolist()
   
   system = AlgoTradingSystem(tickers=tickers)
   await system.run()
   ```

### Configuration

Customize settings in `config.py`:

```python
# Decision weights
SENTIMENT_WEIGHT = 0.4
FUNDAMENTAL_WEIGHT = 0.3
TECHNICAL_WEIGHT = 0.3

# Output file
OUTPUT_FILE = "daily_stock_news.xlsx"

# Tickers to monitor
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", ...]
```

## 📈 Output

### Console Output

The system provides real-time progress updates:

```
======================================================================
Starting analysis at 2025-11-29 10:30:45
======================================================================

📰 Step 1: Scraping news from multiple sources...
✓ Collected 47 news items

🧠 Step 2: Analyzing sentiment...
✓ Analyzed sentiment for 47 items

🔔 Step 3: Checking for high-confidence alerts...
Found 3 high-confidence news items.

📊 Step 4: Calculating metrics and generating trading signals...
  Analyzing AAPL... ✓ Decision: BUY
  Analyzing MSFT... ✓ Decision: HOLD
  ...

💾 Step 5: Saving results to file...
✓ Appended 47 signals to existing file.

📈 Summary of Trading Signals:
----------------------------------------------------------------------
  BUY: 12
  HOLD: 28
  SELL: 5
  STRONG_BUY: 2
----------------------------------------------------------------------
```

### 📁 File Output

Results are automatically saved to `daily_stock_news.xlsx` (or `.csv`) with comprehensive columns:

**Timestamp & Identification:**
- timestamp: Analysis date/time
- ticker: Stock symbol
- source: News source website

**News Details:**
- title: News headline
- url: Link to full article
- category: BREAKING, EARNINGS, DEALS_MA, MACRO_ECONOMIC, or GENERAL

**Sentiment Analysis:**
- sentiment_label: POSITIVE, NEUTRAL, or NEGATIVE
- sentiment_score: Numerical score (-1 to +1)
- sentiment_confidence: Model confidence (0-100%)

**Trading Decision:**
- decision: STRONG_BUY, BUY, HOLD, SELL, or STRONG_SELL
- decision_score: Combined score (-1 to +1)
- reasoning: Detailed explanation of decision

**Financial Metrics:**
- current_price: Current stock price
- rsi: Relative Strength Index
- macd: Moving Average Convergence Divergence
- bollinger_upper/lower: Bollinger Band values
- peg_ratio: Price/Earnings to Growth ratio
- roe: Return on Equity
- eps: Earnings Per Share
- fcf: Free Cash Flow
- intrinsic_value: Calculated fair value

**File Management:**
- **Append Mode**: Add new results to existing file
- **Overwrite Mode**: Create fresh file each run
- **Deduplication**: Automatically removes duplicate entries
- **Excel/CSV Support**: Choose your preferred format

### 🔔 Popup Alerts

Desktop notifications appear automatically for:
- **High Confidence News**: Sentiment confidence > 85%
- **Strong Signals**: STRONG_BUY or STRONG_SELL decisions
- **Breaking News**: Categorized as BREAKING

**Alert Content:**
- Stock ticker and signal type
- News headline
- Sentiment and confidence
- Quick action recommendation

## 🧩 System Architecture

### Component Overview

1. **Web Interface (`app.py`, `utils.py`)**
   - Streamlit-based interactive dashboard
   - CSV parsing and validation utilities
   - Real-time progress tracking
   - Interactive visualizations with Plotly

2. **News Scrapers (`scrapers/`)**
   - BaseNewsScraper: Extensible abstract class with common methods
   - YahooFinanceScraper: Yahoo Finance news
   - FinvizScraper: Finviz market news
   - InvestingScraper: Investing.com articles
   - TradingViewScraper: TradingView market analysis
   - NewsAggregator: Concurrent scraping coordinator
   - Async architecture for optimal performance

3. **Sentiment Analyzer (`sentiment/analyzer.py`)**
   - DistilBERT transformer model for NLP
   - Contextual sentiment classification
   - Confidence scoring (0-100%)
   - GPU acceleration support

4. **Metrics Calculator (`metrics/calculator.py`)**
   - **Fundamentals via yfinance**:
     - PEG ratio (Price/Earnings to Growth)
     - ROE (Return on Equity)
     - EPS (Earnings Per Share)
     - FCF (Free Cash Flow)
     - DCF & Intrinsic Value calculation
   - **Technical Indicators** (native pandas/numpy):
     - RSI (Relative Strength Index)
     - MACD (Moving Average Convergence Divergence)
     - Bollinger Bands (upper/lower)
     - Fibonacci Retracement levels
     - Maximum Drawdown analysis

5. **Decision Engine (`decision_engine/engine.py`)**
   - Weighted scoring algorithm
   - Multi-factor analysis combination
   - Threshold-based decision mapping
   - Detailed reasoning generation

6. **Notification Manager (`notifications/manager.py`)**
   - Cross-platform desktop alerts
   - Configurable thresholds
   - Non-blocking notifications

7. **Storage Manager (`storage/manager.py`)**
   - Excel/CSV export with openpyxl
   - Automatic deduplication
   - Append mode support
   - Data integrity validation

8. **Core Orchestration (`main.py`)**
   - AlgoTradingSystem main class
   - Pipeline coordination
   - Error handling and logging

### Decision Algorithm

Trading signals are generated using a weighted combination of three factors:

```
Combined Score = (Sentiment × 0.4) + (Fundamentals × 0.3) + (Technicals × 0.3)

Sentiment Score:
  - Based on DistilBERT sentiment analysis
  - Range: -1 (very negative) to +1 (very positive)
  - Weighted by confidence level

Fundamental Score:
  - PEG ratio (lower is better)
  - ROE percentage (higher is better)
  - Intrinsic value vs current price
  - Normalized to -1 to +1 range

Technical Score:
  - RSI levels (oversold/overbought)
  - MACD signal crossovers
  - Bollinger Band position
  - Normalized to -1 to +1 range

Decision Mapping:
  Combined Score ≥ 0.7   → STRONG_BUY  (High confidence buy)
  Combined Score ≥ 0.4   → BUY         (Positive signal)
  Combined Score ≤ -0.7  → STRONG_SELL (High confidence sell)
  Combined Score ≤ -0.4  → SELL        (Negative signal)
  Otherwise              → HOLD        (Neutral position)
```

### Data Flow

```
User Input (Tickers) 
    ↓
News Scraping (Async, multiple sources)
    ↓
Sentiment Analysis (DistilBERT AI)
    ↓
Metrics Calculation (Fundamentals + Technicals)
    ↓
Decision Engine (Weighted scoring)
    ↓
Outputs: Dashboard + Excel/CSV + Alerts
```

## 🔧 Troubleshooting

### Installation Issues

**ImportError: No module named 'XXX'**
```powershell
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

**Virtual environment activation fails**
```powershell
# Windows PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\myenv\Scripts\Activate.ps1
```

### Streamlit Issues

**Streamlit command not found**
```powershell
# Install streamlit explicitly
pip install streamlit

# Or run using Python module
python -m streamlit run app.py
```

**Port already in use**
```powershell
# Use different port
streamlit run app.py --server.port 8502
```

**Browser doesn't open automatically**
- Manually navigate to `http://localhost:8501`
- Check firewall settings
- Try different browser

### CSV Upload Issues

**CSV file not loading**
- ✅ Ensure file extension is `.csv`
- ✅ Save as CSV (Comma delimited), not Excel format
- ✅ Check for valid ticker symbols (1-5 characters)
- ✅ Remove special characters except dots and dashes
- ✅ Test with provided `sample_tickers.csv` first

**Tickers showing as invalid**
- Verify symbols are valid US stock tickers
- Check for extra spaces or line breaks
- Ensure uppercase format (auto-converted)
- Maximum 5 characters per ticker

### Analysis Issues

**Analysis taking too long**
- ⏳ Large lists (>20 tickers) need several minutes
- ⏳ First run downloads DistilBERT model (~250MB)
- ⏳ Internet speed affects news scraping
- ⏳ Be patient, progress bar shows status

**No results appearing**
- 🔍 Verify tickers are US stock symbols
- 🔍 Check internet connection
- 🔍 Review console/terminal for error messages
- 🔍 Some tickers may have limited news coverage
- 🔍 Try with fewer tickers first

**Scraping Issues**
- Some websites may block automated scraping
- System handles errors gracefully and continues
- Uses multiple sources to ensure data availability
- Check internet connectivity

### Performance Issues

**Slow first run**
- Downloads DistilBERT model (~250MB) on first execution
- Subsequent runs will be significantly faster
- Model is cached locally

**High memory usage**
- DistilBERT model requires ~1GB RAM
- Large ticker lists increase memory usage
- Consider processing in smaller batches

### Notification Issues

**No popup notifications**
- `plyer` library may not work on all systems
- Notifications will print to console as fallback
- Check OS notification permissions
- Not critical for system functionality

### Data Output Issues

**Excel file not saving**
```powershell
# Check file isn't open in Excel
# Verify write permissions in directory
# Try CSV format instead
```

**Append mode not working**
- Ensure existing file has same format
- Check file isn't locked by another program
- Verify column headers match

### Common Error Messages

**"ModuleNotFoundError: No module named 'pandas_ta'"**
- This dependency was removed - reinstall requirements:
```powershell
pip install -r requirements.txt
```

**"Connection timeout" or "HTTP Error"**
- Check internet connection
- Some news sources may be temporarily unavailable
- System continues with available sources

**"Invalid ticker symbol"**
- Verify ticker format (1-5 characters)
- Check symbol exists on US exchanges
- Use uppercase format

## 💡 Pro Tips & Best Practices

### For Best Results

1. **Start Small**: Test with 5-10 tickers before running large analyses
2. **Use CSV Upload**: Easier to manage and update ticker lists
3. **Enable Append Mode**: Build historical database for trend analysis
4. **Download Sample**: Use `sample_tickers.csv` to understand format
5. **Regular Analysis**: Run daily for portfolio monitoring
6. **Export Data**: Download results from UI for Excel/Google Sheets analysis
7. **Check Multiple Tabs**: Each tab provides different insights
8. **Review Top Signals**: Focus on top buy/sell signals for quick decisions
9. **Batch Processing**: Upload 20-50 tickers for comprehensive sector analysis
10. **Internet Speed**: Ensure stable connection for news scraping

### Understanding Results

- **STRONG_BUY/STRONG_SELL**: High confidence signals (score ≥ 0.7 or ≤ -0.7)
- **BUY/SELL**: Moderate signals (score between 0.4-0.7 or -0.4 to -0.7)
- **HOLD**: Neutral position (score between -0.4 and 0.4)
- **Sentiment Confidence**: Higher percentage = more reliable sentiment
- **Combined Score**: Weighted average of all factors
- **Reasoning**: Check this for detailed explanation of each decision

### Learning Path

**Day 1**: Run with `sample_tickers.csv` to understand system  
**Day 2**: Create custom list with 5 personal holdings  
**Day 3**: Upload full portfolio for comprehensive analysis  
**Day 4**: Enable append mode for daily tracking  
**Day 5**: Analyze historical data and patterns  
**Week 2+**: Use insights to inform trading decisions  

## 📝 Important Notes

### Technical Considerations

- **Web Scraping**: Some scrapers provide simplified implementations due to website complexity. You may need to adjust selectors if website structures change.
- **API Rate Limits**: System uses async I/O to optimize requests and respect rate limits. Be mindful when running frequent analyses.
- **Data Accuracy**: Financial metrics are fetched from yfinance and may have slight delays. Real-time accuracy depends on data source.
- **Model Performance**: DistilBERT sentiment analysis is highly accurate but not perfect. Always verify with news content.
- **Extensible Architecture**: BaseNewsScraper class allows easy addition of new news sources.

### Data Privacy & Security

- **Local Processing**: All analysis runs locally on your machine
- **No External APIs**: Except for news scraping and yfinance data
- **CSV Security**: Your ticker lists are processed in memory only
- **No Data Sharing**: Results saved locally, not sent to external servers

### Performance Considerations

- **First Run**: Downloads DistilBERT model (~250MB), takes 5-10 minutes
- **Subsequent Runs**: Much faster with cached model
- **Memory**: Requires ~1-2GB RAM for optimal performance
- **CPU**: Benefits from multi-core processors for parallel scraping
- **Internet**: Stable connection required for news scraping

## 🚀 Future Enhancements

Potential improvements for future versions:

### Planned Features
- Additional news sources (Bloomberg, Reuters, CNBC, Seeking Alpha)
- Webhook notifications (Discord, Slack, Telegram)
- Historical backtesting capabilities
- Machine learning model training on historical decisions
- Real-time streaming updates
- Cryptocurrency support
- Options analysis integration
- Portfolio optimization suggestions

### Community Contributions Welcome
- Custom scraper implementations
- Additional technical indicators
- Alternative sentiment models
- Performance optimizations
- UI/UX improvements

## 📚 Additional Resources

### Sample Files Included

- **`sample_tickers.csv`**: Example ticker list in proper format
- **`run_streamlit.bat`**: Windows batch script for quick launch
- **`.gitignore`**: Git configuration for clean repositories

### Configuration

All settings can be customized in `config.py`:

```python
# Decision weights (must sum to 1.0)
SENTIMENT_WEIGHT = 0.4      # Sentiment analysis impact
FUNDAMENTAL_WEIGHT = 0.3    # Fundamental metrics impact
TECHNICAL_WEIGHT = 0.3      # Technical indicators impact

# Thresholds for decision mapping
STRONG_BUY_THRESHOLD = 0.7
BUY_THRESHOLD = 0.4
SELL_THRESHOLD = -0.4
STRONG_SELL_THRESHOLD = -0.7

# Alert configuration
ALERT_CONFIDENCE_THRESHOLD = 0.85

# Output settings
OUTPUT_FILE = "daily_stock_news.xlsx"
APPEND_MODE = True

# Default tickers to monitor
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", ...]

# News categorization keywords
BREAKING_KEYWORDS = ["breaking", "urgent", "alert", ...]
EARNINGS_KEYWORDS = ["earnings", "revenue", "profit", ...]
DEALS_KEYWORDS = ["merger", "acquisition", "deal", ...]
```

### Project Structure Reference

```
algo_trade/
├── app.py                    # Streamlit web interface
├── main.py                   # Core system orchestrator
├── config.py                 # Configuration settings
├── models.py                 # Data models & enums
├── utils.py                  # CSV utilities
├── requirements.txt          # Python dependencies
├── sample_tickers.csv        # Example ticker list
├── run_streamlit.bat         # Quick launch script
├── .gitignore               # Git ignore rules
├── README.md                # This documentation
├── scrapers/                # News scraping modules
│   ├── __init__.py          # BaseNewsScraper class
│   ├── yahoo_finance.py     # Yahoo Finance scraper
│   ├── finviz.py            # Finviz scraper
│   ├── investing.py         # Investing.com scraper
│   ├── tradingview.py       # TradingView scraper
│   └── aggregator.py        # Concurrent coordinator
├── sentiment/               # AI sentiment analysis
│   ├── __init__.py
│   └── analyzer.py          # DistilBERT implementation
├── metrics/                 # Financial metrics
│   ├── __init__.py
│   └── calculator.py        # Fundamentals & technicals
├── decision_engine/         # Trading logic
│   ├── __init__.py
│   └── engine.py            # Decision algorithm
├── notifications/           # Alert system
│   ├── __init__.py
│   └── manager.py           # Desktop notifications
└── storage/                 # Data persistence
    ├── __init__.py
    └── manager.py           # Excel/CSV export
```

## 📄 License

This project is provided as-is for educational and informational purposes.

## 🤝 Contributing

Feel free to fork, modify, and enhance the system for your needs!

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly
5. Submit a pull request

Areas for contribution:
- New news source scrapers
- Additional technical indicators
- UI/UX improvements
- Performance optimizations
- Documentation enhancements
- Bug fixes

## 🆘 Getting Help

### Quick Troubleshooting Checklist

1. ✅ Python 3.8+ installed
2. ✅ Virtual environment activated
3. ✅ All dependencies installed: `pip install -r requirements.txt`
4. ✅ Internet connection stable
5. ✅ CSV file in correct format
6. ✅ Valid ticker symbols (1-5 characters)
7. ✅ Streamlit running: `streamlit run app.py`

### Common Issues

- **Import errors**: Reinstall dependencies
- **CSV upload fails**: Check file format and ticker validation
- **Slow performance**: First run downloads AI model
- **No results**: Verify internet connection and ticker validity

## 📞 Support

For issues, questions, or contributions:
- Review this comprehensive README
- Check the troubleshooting section
- Test with `sample_tickers.csv`
- Verify all dependencies are installed

---

## ⚠️ Disclaimer

**IMPORTANT: READ CAREFULLY**

This software is provided for **educational and informational purposes only**. It does not constitute:

- Financial advice
- Investment recommendations  
- Professional trading guidance
- Legal or tax advice

**Key Points:**

- 📊 **Not Financial Advice**: All trading decisions carry risk. Past performance does not guarantee future results.
- 🔬 **Research Tool**: This system is designed to assist with research, not make trading decisions for you.
- 💰 **Risk Warning**: Stock trading involves substantial risk of loss. Only invest what you can afford to lose.
- 🎓 **Educational Use**: Perfect for learning about algorithmic trading, sentiment analysis, and financial data processing.
- ⚖️ **Consult Professionals**: Always consult qualified financial advisors before making investment decisions.
- 🔍 **Verify Data**: Cross-check all data and signals with multiple sources.
- 📈 **No Guarantees**: The system's predictions and signals are not guaranteed to be accurate or profitable.

**Use at your own risk. The developers and contributors assume no liability for financial losses.**

---

**Ready to get started? Run `streamlit run app.py` and begin analyzing! 🚀📈**

*Last Updated: November 2025*
