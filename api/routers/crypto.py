"""
Crypto API router.

Endpoints for fetching Binance price data, listing crypto strategies,
and running the mean reversion backtest.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from api.schemas.common import ErrorResponse, SuccessResponse
from api.schemas.crypto import (
    CryptoBacktestRequest,
    CryptoBacktestResponse,
    CryptoPriceData,
    CryptoPriceRequest,
    CryptoPriceResponse,
    CryptoStrategyInfo,
    CryptoStrategyListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crypto", tags=["Crypto"])


# -----------------------------------------------------------------------
# Market Data
# -----------------------------------------------------------------------

@router.post(
    "/prices",
    response_model=CryptoPriceResponse,
    summary="Fetch crypto price data from Binance",
)
async def get_crypto_prices(request: CryptoPriceRequest):
    """
    Download daily close prices for the requested crypto symbols from
    the Binance public REST API. No API key required.
    """
    try:
        from trading_strategies.crypto.binance_data import fetch_crypto_prices

        # Build the symbol -> column-name map expected by fetch_crypto_prices
        symbols_map: Dict[str, str] = {}
        for sym in request.symbols:
            s = sym.upper()
            # Convert e.g. "BTCUSDT" -> "btc" column name
            col = s.replace("USDT", "").replace("BUSD", "").lower()
            symbols_map[s] = col

        df = fetch_crypto_prices(
            symbols=symbols_map,
            start=request.start_date or "2024-01-01",
            end=request.end_date,
        )

        data: List[CryptoPriceData] = []
        for col in df.columns:
            series = df[col].dropna()
            data.append(
                CryptoPriceData(
                    symbol=col,
                    dates=[d.strftime("%Y-%m-%d") for d in series.index],
                    prices=series.tolist(),
                    data_points=len(series),
                )
            )

        return CryptoPriceResponse(success=True, data=data)
    except Exception as exc:
        logger.exception("Crypto price fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------

@router.get(
    "/strategies",
    response_model=CryptoStrategyListResponse,
    summary="List available crypto strategies",
)
async def list_crypto_strategies():
    """Return metadata for all registered crypto strategies."""
    strategies = [
        CryptoStrategyInfo(
            id="crypto_mean_reversion",
            name="Crypto Mean Reversion (Z-Score)",
            description=(
                "Binance-sourced crypto portfolio with Z-Score mean-reversion "
                "signals — full EDA, cointegration tests & backtesting optimisation"
            ),
            category="crypto",
            min_tickers=2,
        ),
    ]
    return CryptoStrategyListResponse(success=True, strategies=strategies)


@router.get(
    "/strategies/{strategy_id}",
    response_model=CryptoStrategyInfo,
    summary="Get details for a specific crypto strategy",
)
async def get_crypto_strategy(strategy_id: str):
    """Return parameters and metadata for a crypto strategy."""
    if strategy_id != "crypto_mean_reversion":
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")

    try:
        from trading_strategies.crypto.mean_reversion_strategy import (
            CryptoMeanReversionStrategy,
        )

        strat = CryptoMeanReversionStrategy()
        params = strat.get_parameters()
        return CryptoStrategyInfo(
            id="crypto_mean_reversion",
            name=strat.name,
            description=strat.description,
            category="crypto",
            min_tickers=2,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Backtesting
# -----------------------------------------------------------------------

@router.post(
    "/backtest",
    response_model=CryptoBacktestResponse,
    summary="Run crypto mean reversion backtest",
)
async def run_crypto_backtest(request: CryptoBacktestRequest):
    """
    Execute the full CryptoMeanReversionStrategy pipeline:
    EDA, cointegration tests, portfolio construction, backtesting,
    and parameter optimisation.
    """
    start_time = time.time()

    try:
        from trading_strategies.crypto.mean_reversion_strategy import (
            CryptoMeanReversionStrategy,
        )

        strategy = CryptoMeanReversionStrategy()

        result = strategy.run(
            tickers=request.symbols,
            start_date=request.start_date or "2023-01-01",
            end_date=request.end_date or "",
            capital=request.initial_capital,
            **request.parameters,
        )

        elapsed = time.time() - start_time

        if not result.success:
            return CryptoBacktestResponse(
                success=False,
                symbols=request.symbols,
                error_message=result.error_message or "Strategy execution failed",
                execution_time=elapsed,
            )

        # Convert StrategyResult to response dict
        result_dict = result.to_dict()

        return CryptoBacktestResponse(
            success=True,
            symbols=request.symbols,
            metrics=result_dict.get("metrics", {}),
            charts=result_dict.get("charts", []),
            tables=result_dict.get("tables", []),
            execution_time=elapsed,
            metadata=result_dict.get("metadata", {}),
        )
    except Exception as exc:
        elapsed = time.time() - start_time
        logger.exception("Crypto backtest failed")
        raise HTTPException(status_code=500, detail=str(exc))
