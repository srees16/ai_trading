"""
Shared Backtest Utilities.

Provides common functions used across standalone _bktest.py scripts:
- mdd: Maximum drawdown calculation
- candlestick: Matplotlib candlestick chart rendering
- portfolio: Basic portfolio value calculation
- profit: Portfolio profit/loss plotting

These were previously duplicated in individual _bktest.py files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mdd(series: pd.Series) -> float:
    """
    Calculate maximum drawdown from a value series.
    
    For every point, compares the current value to the prior peak.
    Returns the largest percentage decline (negative number).
    
    Args:
        series: Asset value series (e.g. portfolio total asset)
    
    Returns:
        Maximum drawdown as a negative float (e.g. -0.15 for 15% drawdown)
    """
    minimum = 0
    for i in range(1, len(series)):
        drawdown = series.iloc[i] / max(series.iloc[:i]) - 1
        if minimum > drawdown:
            minimum = drawdown
    return minimum


def candlestick(
    df: pd.DataFrame,
    ax=None,
    highlight=None,
    titlename: str = '',
    highcol: str = 'High',
    lowcol: str = 'Low',
    opencol: str = 'Open',
    closecol: str = 'Close',
    xcol: str = 'Date',
    colorup: str = 'r',
    colordown: str = 'g',
    highlightcolor: str = '#FFFF00',
    **kwargs
):
    """
    Plot a candlestick chart using matplotlib fill_between.
    
    Renders OHLC data as filled bars with high/low wicks. Optionally
    highlights specific candles (e.g. pattern matches).
    
    Args:
        df: DataFrame with OHLC data
        ax: Matplotlib axes (creates new figure if None)
        highlight: Column name for highlight markers (-1 = highlight)
        titlename: Chart title
        highcol, lowcol, opencol, closecol: Column names for OHLC
        xcol: Column name for x-axis dates
        colorup: Color for bearish candles (open > close)
        colordown: Color for bullish candles
        highlightcolor: Color for highlighted candles
        **kwargs: Additional matplotlib kwargs
    """
    # Bar width: 0.6 default via 7 fill-between points
    dif = [(-3 + i) / 10 for i in range(7)]
    
    if not ax:
        ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    
    for i in range(len(df)):
        x = [i + j for j in dif]
        y1 = [df[opencol].iloc[i]] * 7
        y2 = [df[closecol].iloc[i]] * 7
        
        barcolor = colorup if y1[0] > y2[0] else colordown
        
        # High wick
        if df[highcol].iloc[i] != max(df[opencol].iloc[i], df[closecol].iloc[i]):
            plt.plot(
                [i, i],
                [df[highcol].iloc[i],
                 max(df[opencol].iloc[i], df[closecol].iloc[i]) * 1.001],
                c='k', **kwargs
            )
        
        # Low wick
        if df[lowcol].iloc[i] != min(df[opencol].iloc[i], df[closecol].iloc[i]):
            plt.plot(
                [i, i],
                [df[lowcol].iloc[i],
                 min(df[opencol].iloc[i], df[closecol].iloc[i]) * 0.999],
                c='k', **kwargs
            )
        
        # Body
        plt.fill_between(x, y1, y2, edgecolor='k', facecolor=barcolor, **kwargs)
        
        # Highlight specific candles
        if highlight and df[highlight].iloc[i] == -1:
            plt.fill_between(x, y1, y2, edgecolor='k',
                             facecolor=highlightcolor, **kwargs)
    
    # X-axis ticks
    if xcol in df.columns and len(df) > 5:
        step = len(df) // 5
        plt.xticks(
            range(0, len(df), step),
            df[xcol][0::step].dt.date if hasattr(df[xcol].iloc[0], 'date') else df[xcol][0::step]
        )
    
    plt.title(titlename)


def portfolio(
    data: pd.DataFrame,
    capital0: float = 10000,
    positions: int = 100,
    signal_col: str = 'signals',
    cumsum_col: str = None
) -> pd.DataFrame:
    """
    Calculate basic portfolio value from trading signals.
    
    Computes holdings (from cumulative position * price * shares),
    cash (initial capital minus cost of trades), and total asset value.
    
    Args:
        data: DataFrame with Close prices and signal columns
        capital0: Initial capital
        positions: Number of shares per trade
        signal_col: Name of the trading signal column
        cumsum_col: Name of cumulative position column (auto-created if None)
    
    Returns:
        DataFrame with holdings, cash, total asset, return, and signals
    """
    if cumsum_col is None:
        cumsum_col = 'cumsum'
        data[cumsum_col] = data[signal_col].cumsum()
    
    port = pd.DataFrame()
    port['holdings'] = data[cumsum_col] * data['Close'] * positions
    port['cash'] = capital0 - (data[signal_col] * data['Close'] * positions).cumsum()
    port['total asset'] = port['holdings'] + port['cash']
    port['return'] = port['total asset'].pct_change()
    port['signals'] = data[signal_col]
    
    if 'Date' in data.columns:
        port['date'] = data['Date']
        port.set_index('date', inplace=True)
    
    return port


def profit(port: pd.DataFrame, asset_col: str = 'total asset', title: str = 'Total Asset'):
    """
    Plot portfolio asset value over time with trade markers.
    
    Args:
        port: Portfolio DataFrame from portfolio()
        asset_col: Column name for asset values
        title: Chart title
    """
    fig = plt.figure()
    bx = fig.add_subplot(111)
    
    bx.plot(port.index, port[asset_col], label=title)
    
    # Long/short markers
    if 'signals' in port.columns:
        longs = port['signals'] == 1
        shorts = port['signals'] < 0
        if longs.any():
            bx.plot(
                port.index[longs],
                port[asset_col][longs],
                lw=0, marker='^', c='g', label='long'
            )
        if shorts.any():
            bx.plot(
                port.index[shorts],
                port[asset_col][shorts],
                lw=0, marker='v', c='r', label='short'
            )
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Asset Value')
    plt.title(title)
    plt.show()
