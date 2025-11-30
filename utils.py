"""
Utility functions for CSV processing and data handling.
"""

import pandas as pd
from typing import List, Optional
from io import StringIO


def parse_ticker_csv(file_content: str) -> List[str]:
    """
    Parse CSV file content to extract ticker symbols.
    
    Supports various CSV formats:
    - Single column with header (Ticker, Symbol, Stock, etc.)
    - Single column without header
    - Multiple columns (extracts first column or column named 'ticker'/'symbol')
    
    Args:
        file_content: String content of the CSV file
        
    Returns:
        List of unique ticker symbols (uppercase, stripped)
    """
    try:
        # Read CSV
        df = pd.read_csv(StringIO(file_content))
        
        # Try to find ticker column
        ticker_col = None
        
        # Look for common ticker column names
        common_names = ['ticker', 'symbol', 'stock', 'tickers', 'symbols', 'stocks', 'name']
        for col in df.columns:
            if col.lower().strip() in common_names:
                ticker_col = col
                break
        
        # If no ticker column found, use first column
        if ticker_col is None:
            ticker_col = df.columns[0]
        
        # Extract tickers
        tickers = df[ticker_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
        
        # Filter out empty strings and invalid tickers
        tickers = [t for t in tickers if t and len(t) > 0 and len(t) <= 5]
        
        return tickers
    
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        # Try simple line-by-line parsing as fallback
        try:
            lines = file_content.strip().split('\n')
            tickers = []
            for line in lines:
                # Split by comma and take first item
                ticker = line.split(',')[0].strip().upper()
                if ticker and len(ticker) <= 5 and ticker.isalpha():
                    tickers.append(ticker)
            return list(set(tickers))
        except:
            return []


def validate_tickers(tickers: List[str]) -> tuple[List[str], List[str]]:
    """
    Validate ticker symbols.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Tuple of (valid_tickers, invalid_tickers)
    """
    valid = []
    invalid = []
    
    for ticker in tickers:
        # Basic validation: 1-5 uppercase letters
        if ticker and 1 <= len(ticker) <= 5 and ticker.replace('.', '').replace('-', '').isalnum():
            valid.append(ticker)
        else:
            invalid.append(ticker)
    
    return valid, invalid


def create_sample_csv() -> str:
    """
    Create a sample CSV content for user reference.
    
    Returns:
        Sample CSV string
    """
    sample = """Ticker,Company
AAPL,Apple Inc.
MSFT,Microsoft Corporation
GOOGL,Alphabet Inc.
AMZN,Amazon.com Inc.
TSLA,Tesla Inc.
NVDA,NVIDIA Corporation
META,Meta Platforms Inc.
JPM,JPMorgan Chase
V,Visa Inc.
WMT,Walmart Inc."""
    
    return sample
