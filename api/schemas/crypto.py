"""
Pydantic schemas for Crypto API endpoints.

Covers: crypto data fetching, mean reversion strategy, and backtesting.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Crypto Market Data
# ---------------------------------------------------------------------------

class CryptoPriceRequest(BaseModel):
    """Request to fetch crypto price data from Binance."""
    symbols: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        examples=[["BTCUSDT", "ETHUSDT"]],
        description="Binance trading pair symbols",
    )
    start_date: Optional[str] = Field(
        None,
        examples=["2024-01-01"],
        description="Start date in YYYY-MM-DD format",
    )
    end_date: Optional[str] = Field(
        None,
        examples=["2025-01-01"],
        description="End date in YYYY-MM-DD format",
    )


class CryptoPriceData(BaseModel):
    """Price data for a single crypto pair."""
    symbol: str
    dates: List[str]
    prices: List[float]
    volumes: List[float] = Field(default_factory=list)
    data_points: int = 0


class CryptoPriceResponse(BaseModel):
    """Response containing crypto price data."""
    success: bool = True
    data: List[CryptoPriceData] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Crypto Backtesting
# ---------------------------------------------------------------------------

class CryptoBacktestRequest(BaseModel):
    """Request to run crypto mean reversion backtest."""
    symbols: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        examples=[["BTCUSDT", "ETHUSDT"]],
        description="Binance symbols for cointegration analysis",
    )
    start_date: Optional[str] = Field(
        None,
        examples=["2023-01-01"],
        description="Backtest start date (YYYY-MM-DD)",
    )
    end_date: Optional[str] = Field(
        None,
        examples=["2025-01-01"],
        description="Backtest end date (YYYY-MM-DD)",
    )
    initial_capital: float = Field(100000.0, gt=0)
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters (z_entry, z_exit, lookback, etc.)",
    )


class CryptoBacktestResponse(BaseModel):
    """Response from a crypto backtest run."""
    success: bool = True
    symbols: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Crypto Strategy Listing
# ---------------------------------------------------------------------------

class CryptoStrategyInfo(BaseModel):
    """Metadata for a crypto strategy."""
    id: str
    name: str
    description: str
    category: str = "crypto"
    min_tickers: int = 2


class CryptoStrategyListResponse(BaseModel):
    """Response listing available crypto strategies."""
    success: bool = True
    strategies: List[CryptoStrategyInfo]
