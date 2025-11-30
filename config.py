"""
Configuration settings for the algo-trading alert system.
"""

from typing import List


class Config:
    """Global configuration."""
    
    # Sentiment Analysis
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    SENTIMENT_HIGH_CONFIDENCE_THRESHOLD = 0.85
    
    # Storage
    OUTPUT_FILE = "daily_stock_news.xlsx"
    APPEND_MODE = True
    
    # Scraping
    REQUEST_TIMEOUT = 10  # seconds
    MAX_CONCURRENT_REQUESTS = 5
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    )
    
    # Metrics
    HISTORICAL_DAYS = 365  # Days of historical data for technical analysis
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    
    # Decision Engine Weights
    SENTIMENT_WEIGHT = 0.4
    FUNDAMENTAL_WEIGHT = 0.3
    TECHNICAL_WEIGHT = 0.3
    
    # Decision Thresholds
    STRONG_BUY_THRESHOLD = 0.7
    BUY_THRESHOLD = 0.4
    SELL_THRESHOLD = -0.4
    STRONG_SELL_THRESHOLD = -0.7
    
    # Notification
    NOTIFICATION_DURATION = 10  # seconds
    
    # Tickers to monitor (can be expanded)
    DEFAULT_TICKERS: List[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "TSLA", "NVDA", "JPM", "V", "WMT"
    ]
    
    # News keywords for categorization
    BREAKING_KEYWORDS = ["breaking", "urgent", "alert", "just in"]
    DEALS_KEYWORDS = ["merger", "acquisition", "deal", "buyout", "takeover"]
    MACRO_KEYWORDS = ["fed", "interest rate", "inflation", "gdp", "unemployment", "treasury"]
    EARNINGS_KEYWORDS = ["earnings", "quarterly", "q1", "q2", "q3", "q4", "revenue", "profit"]
