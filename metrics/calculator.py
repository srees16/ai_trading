"""
Stock Metrics Calculator Module.

Calculates fundamental and technical metrics for stocks
using Yahoo Finance data and standard financial formulas.
"""

import logging

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict

from models import StockMetrics
from config import Config

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates fundamental and technical metrics for stocks."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def get_stock_metrics(self, ticker: str) -> Optional[StockMetrics]:
        """
        Calculate all metrics for a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            StockMetrics object with all calculated metrics
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get stock info
            info = stock.info
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=Config.HISTORICAL_DAYS)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning("No historical data for %s", ticker)
                return None
            
            # Calculate fundamentals
            fundamentals = self._calculate_fundamentals(ticker, stock, info)
            
            # Calculate technicals
            technicals = self._calculate_technicals(hist)
            
            # Create metrics object
            metrics = StockMetrics(
                ticker=ticker,
                timestamp=datetime.now(),
                peg_ratio=fundamentals.get('peg_ratio'),
                roe=fundamentals.get('roe'),
                eps=fundamentals.get('eps'),
                free_cash_flow=fundamentals.get('free_cash_flow'),
                dcf_value=fundamentals.get('dcf_value'),
                intrinsic_value=fundamentals.get('intrinsic_value'),
                altman_z_score=fundamentals.get('altman_z_score'),
                beneish_m_score=fundamentals.get('beneish_m_score'),
                piotroski_f_score=fundamentals.get('piotroski_f_score'),
                rsi=technicals.get('rsi'),
                macd=technicals.get('macd'),
                macd_signal=technicals.get('macd_signal'),
                macd_histogram=technicals.get('macd_histogram'),
                fibonacci_levels=technicals.get('fibonacci_levels'),
                bollinger_upper=technicals.get('bollinger_upper'),
                bollinger_middle=technicals.get('bollinger_middle'),
                bollinger_lower=technicals.get('bollinger_lower'),
                max_drawdown=technicals.get('max_drawdown'),
                current_price=technicals.get('current_price')
            )
            
            return metrics
        
        except Exception as e:
            logger.error("Error calculating metrics for %s: %s", ticker, e)
            return None
    
    def _calculate_fundamentals(
        self, 
        ticker: str, 
        stock: yf.Ticker, 
        info: dict
    ) -> Dict[str, Optional[float]]:
        """Calculate fundamental metrics."""
        fundamentals = {}
        
        try:
            # PEG Ratio
            peg = info.get('pegRatio')
            fundamentals['peg_ratio'] = float(peg) if peg else None
            
            # ROE (Return on Equity)
            roe = info.get('returnOnEquity')
            fundamentals['roe'] = float(roe) * 100 if roe else None
            
            # EPS (Earnings Per Share)
            eps = info.get('trailingEps')
            fundamentals['eps'] = float(eps) if eps else None
            
            # Free Cash Flow
            fcf = info.get('freeCashflow')
            fundamentals['free_cash_flow'] = float(fcf) if fcf else None
            
            # DCF Value (simplified calculation)
            dcf = self._calculate_dcf(info)
            fundamentals['dcf_value'] = dcf
            
            # Intrinsic Value (using Graham's formula)
            intrinsic = self._calculate_intrinsic_value(info)
            fundamentals['intrinsic_value'] = intrinsic
            
            # Advanced Fundamental Metrics
            # Altman Z-Score (bankruptcy risk)
            altman_z = self._calculate_altman_z_score(stock, info)
            fundamentals['altman_z_score'] = altman_z
            
            # Beneish M-Score (earnings manipulation detection)
            beneish_m = self._calculate_beneish_m_score(stock)
            fundamentals['beneish_m_score'] = beneish_m
            
            # Piotroski F-Score (financial health)
            piotroski_f = self._calculate_piotroski_f_score(stock, info)
            fundamentals['piotroski_f_score'] = piotroski_f
        
        except Exception as e:
            logger.error("Error calculating fundamentals: %s", e)
        
        return fundamentals
    
    def _calculate_dcf(self, info: dict) -> Optional[float]:
        """
        Calculate simplified DCF (Discounted Cash Flow) value.
        
        This is a simplified DCF calculation using free cash flow.
        """
        try:
            fcf = info.get('freeCashflow')
            shares = info.get('sharesOutstanding')
            
            if not fcf or not shares:
                return None
            
            # Simplified DCF with 10% discount rate and 3% growth
            discount_rate = 0.10
            growth_rate = 0.03
            
            # Terminal value
            terminal_value = fcf * (1 + growth_rate) / (discount_rate - growth_rate)
            
            # DCF per share
            dcf_value = terminal_value / shares
            
            return float(dcf_value)
        
        except:
            return None
    
    def _calculate_intrinsic_value(self, info: dict) -> Optional[float]:
        """
        Calculate intrinsic value using Benjamin Graham's formula.
        
        IV = EPS × (8.5 + 2g) × 4.4 / Y
        where g is growth rate and Y is current yield on AAA bonds (approx 4.5%)
        """
        try:
            eps = info.get('trailingEps')
            growth = info.get('earningsGrowth')
            
            if not eps:
                return None
            
            # Default growth rate if not available
            g = (growth * 100) if growth else 10
            
            # Graham's formula
            intrinsic_value = eps * (8.5 + 2 * g) * 4.4 / 4.5
            
            return float(intrinsic_value)
        
        except:
            return None
    
    def _calculate_technicals(self, hist: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate technical indicators using native pandas/numpy."""
        technicals = {}
        
        try:
            # Current price
            technicals['current_price'] = float(hist['Close'].iloc[-1])
            
            # RSI (Relative Strength Index)
            rsi = self._calculate_rsi(hist['Close'], Config.RSI_PERIOD)
            technicals['rsi'] = float(rsi) if rsi is not None else None
            
            # MACD
            macd_dict = self._calculate_macd(
                hist['Close'], 
                Config.MACD_FAST, 
                Config.MACD_SLOW, 
                Config.MACD_SIGNAL
            )
            technicals['macd'] = macd_dict.get('macd')
            technicals['macd_signal'] = macd_dict.get('signal')
            technicals['macd_histogram'] = macd_dict.get('histogram')
            
            # Bollinger Bands
            bb_dict = self._calculate_bollinger_bands(
                hist['Close'], 
                Config.BOLLINGER_PERIOD, 
                Config.BOLLINGER_STD
            )
            technicals['bollinger_upper'] = bb_dict.get('upper')
            technicals['bollinger_middle'] = bb_dict.get('middle')
            technicals['bollinger_lower'] = bb_dict.get('lower')
            
            # Fibonacci Retracement Levels
            fib_levels = self._calculate_fibonacci(hist)
            technicals['fibonacci_levels'] = fib_levels
            
            # Maximum Drawdown
            max_dd = self._calculate_max_drawdown(hist)
            technicals['max_drawdown'] = max_dd
        
        except Exception as e:
            logger.error("Error calculating technicals: %s", e)
        
        return technicals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not rsi.empty else None
        except:
            return None
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Dict[str, Optional[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            
            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal_line.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        except:
            return {'macd': None, 'signal': None, 'histogram': None}
    
    def _calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: int = 2
    ) -> Dict[str, Optional[float]]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'upper': float(upper_band.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower_band.iloc[-1])
            }
        except:
            return {'upper': None, 'middle': None, 'lower': None}
    
    def _calculate_fibonacci(self, hist: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        try:
            high = hist['High'].max()
            low = hist['Low'].min()
            diff = high - low
            
            levels = {
                '0.0': float(high),
                '0.236': float(high - 0.236 * diff),
                '0.382': float(high - 0.382 * diff),
                '0.500': float(high - 0.500 * diff),
                '0.618': float(high - 0.618 * diff),
                '0.786': float(high - 0.786 * diff),
                '1.0': float(low)
            }
            
            return levels
        
        except:
            return {}
    
    def _calculate_max_drawdown(self, hist: pd.DataFrame) -> Optional[float]:
        """Calculate maximum drawdown percentage."""
        try:
            prices = hist['Close']
            cumulative_max = prices.cummax()
            drawdown = (prices - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()
            
            return float(max_drawdown * 100)  # Return as percentage
        
        except:
            return None

    def _calculate_altman_z_score(self, stock: yf.Ticker, info: dict) -> Optional[float]:
        """
        Calculate Altman Z-Score for bankruptcy risk assessment.
        
        Z = 1.2×A + 1.4×B + 3.3×C + 0.6×D + 1.0×E
        
        Where:
        - A = Working Capital / Total Assets
        - B = Retained Earnings / Total Assets
        - C = EBIT / Total Assets
        - D = Market Value of Equity / Total Liabilities
        - E = Sales / Total Assets
        
        Interpretation:
        - Z > 2.99: Safe zone (low bankruptcy risk)
        - 1.81 < Z < 2.99: Grey zone (moderate risk)
        - Z < 1.81: Distress zone (high bankruptcy risk)
        """
        try:
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            
            if balance_sheet.empty or income_stmt.empty:
                return None
            
            # Get latest values
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
            
            if not total_assets or total_assets == 0:
                return None
            
            # A = Working Capital / Total Assets
            current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else 0
            current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 0
            working_capital = current_assets - current_liabilities
            A = working_capital / total_assets
            
            # B = Retained Earnings / Total Assets
            retained_earnings = balance_sheet.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in balance_sheet.index else 0
            B = retained_earnings / total_assets
            
            # C = EBIT / Total Assets
            ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else (
                income_stmt.loc['Operating Income'].iloc[0] if 'Operating Income' in income_stmt.index else 0
            )
            C = ebit / total_assets
            
            # D = Market Value of Equity / Total Liabilities
            market_cap = info.get('marketCap', 0)
            total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else (
                balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 1
            )
            D = market_cap / total_liabilities if total_liabilities and total_liabilities != 0 else 0
            
            # E = Sales / Total Assets
            revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
            E = revenue / total_assets
            
            # Calculate Z-Score
            z_score = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
            
            return round(float(z_score), 2)
        
        except Exception as e:
            logger.error("Error calculating Altman Z-Score: %s", e)
            return None

    def _calculate_beneish_m_score(self, stock: yf.Ticker) -> Optional[float]:
        """
        Calculate Beneish M-Score for earnings manipulation detection.
        
        M = -4.84 + 0.92×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI 
            + 0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI
        
        Where:
        - DSRI = Days Sales in Receivables Index
        - GMI = Gross Margin Index
        - AQI = Asset Quality Index
        - SGI = Sales Growth Index
        - DEPI = Depreciation Index
        - SGAI = SG&A Index
        - TATA = Total Accruals to Total Assets
        - LVGI = Leverage Index
        
        Interpretation:
        - M > -2.22: Likely earnings manipulator
        - M < -2.22: Unlikely to be manipulator
        """
        try:
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cashflow
            
            if balance_sheet.empty or income_stmt.empty or len(balance_sheet.columns) < 2:
                return None
            
            # Current and prior year data
            def safe_get(df, key, col=0):
                return df.loc[key].iloc[col] if key in df.index and len(df.columns) > col else 0
            
            # Current year (col 0) and prior year (col 1)
            receivables_t = safe_get(balance_sheet, 'Accounts Receivable', 0) or safe_get(balance_sheet, 'Net Receivables', 0)
            receivables_t1 = safe_get(balance_sheet, 'Accounts Receivable', 1) or safe_get(balance_sheet, 'Net Receivables', 1)
            
            revenue_t = safe_get(income_stmt, 'Total Revenue', 0)
            revenue_t1 = safe_get(income_stmt, 'Total Revenue', 1)
            
            if not revenue_t or not revenue_t1 or revenue_t1 == 0:
                return None
            
            # DSRI - Days Sales in Receivables Index
            dsri_t = receivables_t / revenue_t if revenue_t else 0
            dsri_t1 = receivables_t1 / revenue_t1 if revenue_t1 else 0
            DSRI = dsri_t / dsri_t1 if dsri_t1 and dsri_t1 != 0 else 1
            
            # GMI - Gross Margin Index
            gross_profit_t = safe_get(income_stmt, 'Gross Profit', 0)
            gross_profit_t1 = safe_get(income_stmt, 'Gross Profit', 1)
            gm_t = gross_profit_t / revenue_t if revenue_t else 0
            gm_t1 = gross_profit_t1 / revenue_t1 if revenue_t1 else 0
            GMI = gm_t1 / gm_t if gm_t and gm_t != 0 else 1
            
            # AQI - Asset Quality Index
            total_assets_t = safe_get(balance_sheet, 'Total Assets', 0)
            total_assets_t1 = safe_get(balance_sheet, 'Total Assets', 1)
            current_assets_t = safe_get(balance_sheet, 'Current Assets', 0)
            current_assets_t1 = safe_get(balance_sheet, 'Current Assets', 1)
            ppe_t = safe_get(balance_sheet, 'Net PPE', 0) or safe_get(balance_sheet, 'Property Plant Equipment Net', 0)
            ppe_t1 = safe_get(balance_sheet, 'Net PPE', 1) or safe_get(balance_sheet, 'Property Plant Equipment Net', 1)
            
            aq_t = 1 - (current_assets_t + ppe_t) / total_assets_t if total_assets_t else 0
            aq_t1 = 1 - (current_assets_t1 + ppe_t1) / total_assets_t1 if total_assets_t1 else 0
            AQI = aq_t / aq_t1 if aq_t1 and aq_t1 != 0 else 1
            
            # SGI - Sales Growth Index
            SGI = revenue_t / revenue_t1 if revenue_t1 else 1
            
            # DEPI - Depreciation Index
            depreciation_t = safe_get(income_stmt, 'Depreciation And Amortization', 0) or safe_get(cash_flow, 'Depreciation And Amortization', 0)
            depreciation_t1 = safe_get(income_stmt, 'Depreciation And Amortization', 1) or safe_get(cash_flow, 'Depreciation And Amortization', 1)
            dep_rate_t = depreciation_t / (depreciation_t + ppe_t) if (depreciation_t + ppe_t) else 0
            dep_rate_t1 = depreciation_t1 / (depreciation_t1 + ppe_t1) if (depreciation_t1 + ppe_t1) else 0
            DEPI = dep_rate_t1 / dep_rate_t if dep_rate_t and dep_rate_t != 0 else 1
            
            # SGAI - SG&A Index
            sga_t = safe_get(income_stmt, 'Selling General And Administration', 0)
            sga_t1 = safe_get(income_stmt, 'Selling General And Administration', 1)
            sga_ratio_t = sga_t / revenue_t if revenue_t else 0
            sga_ratio_t1 = sga_t1 / revenue_t1 if revenue_t1 else 0
            SGAI = sga_ratio_t / sga_ratio_t1 if sga_ratio_t1 and sga_ratio_t1 != 0 else 1
            
            # TATA - Total Accruals to Total Assets
            net_income = safe_get(income_stmt, 'Net Income', 0)
            operating_cf = safe_get(cash_flow, 'Operating Cash Flow', 0) or safe_get(cash_flow, 'Cash Flow From Continuing Operating Activities', 0)
            TATA = (net_income - operating_cf) / total_assets_t if total_assets_t else 0
            
            # LVGI - Leverage Index
            total_debt_t = safe_get(balance_sheet, 'Total Debt', 0) or safe_get(balance_sheet, 'Long Term Debt', 0)
            total_debt_t1 = safe_get(balance_sheet, 'Total Debt', 1) or safe_get(balance_sheet, 'Long Term Debt', 1)
            leverage_t = total_debt_t / total_assets_t if total_assets_t else 0
            leverage_t1 = total_debt_t1 / total_assets_t1 if total_assets_t1 else 0
            LVGI = leverage_t / leverage_t1 if leverage_t1 and leverage_t1 != 0 else 1
            
            # Calculate M-Score
            m_score = (-4.84 + 0.92 * DSRI + 0.528 * GMI + 0.404 * AQI + 
                       0.892 * SGI + 0.115 * DEPI - 0.172 * SGAI + 
                       4.679 * TATA - 0.327 * LVGI)
            
            return round(float(m_score), 2)
        
        except Exception as e:
            logger.error("Error calculating Beneish M-Score: %s", e)
            return None

    def _calculate_piotroski_f_score(self, stock: yf.Ticker, info: dict) -> Optional[int]:
        """
        Calculate Piotroski F-Score for financial health assessment.
        
        Score from 0 to 9 based on 9 criteria:
        
        Profitability (4 points):
        1. Net Income > 0 (1 point)
        2. Operating Cash Flow > 0 (1 point)
        3. ROA current year > ROA prior year (1 point)
        4. Operating Cash Flow > Net Income (1 point)
        
        Leverage/Liquidity (3 points):
        5. Long-term Debt ratio decreased (1 point)
        6. Current Ratio increased (1 point)
        7. No new shares issued (1 point)
        
        Operating Efficiency (2 points):
        8. Gross Margin increased (1 point)
        9. Asset Turnover increased (1 point)
        
        Interpretation:
        - 8-9: Strong (buy candidates)
        - 5-7: Moderate
        - 0-4: Weak (avoid or sell)
        """
        try:
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cashflow
            
            if balance_sheet.empty or income_stmt.empty:
                return None
            
            score = 0
            
            def safe_get(df, key, col=0):
                return df.loc[key].iloc[col] if key in df.index and len(df.columns) > col else 0
            
            # Current year (col 0) and prior year (col 1)
            net_income_t = safe_get(income_stmt, 'Net Income', 0)
            net_income_t1 = safe_get(income_stmt, 'Net Income', 1)
            
            total_assets_t = safe_get(balance_sheet, 'Total Assets', 0)
            total_assets_t1 = safe_get(balance_sheet, 'Total Assets', 1)
            
            revenue_t = safe_get(income_stmt, 'Total Revenue', 0)
            revenue_t1 = safe_get(income_stmt, 'Total Revenue', 1)
            
            # 1. Net Income > 0
            if net_income_t and net_income_t > 0:
                score += 1
            
            # 2. Operating Cash Flow > 0
            operating_cf = safe_get(cash_flow, 'Operating Cash Flow', 0) or safe_get(cash_flow, 'Cash Flow From Continuing Operating Activities', 0)
            if operating_cf and operating_cf > 0:
                score += 1
            
            # 3. ROA current year > ROA prior year
            roa_t = net_income_t / total_assets_t if total_assets_t else 0
            roa_t1 = net_income_t1 / total_assets_t1 if total_assets_t1 else 0
            if roa_t > roa_t1:
                score += 1
            
            # 4. Operating Cash Flow > Net Income (quality of earnings)
            if operating_cf and net_income_t and operating_cf > net_income_t:
                score += 1
            
            # 5. Long-term Debt ratio decreased
            long_debt_t = safe_get(balance_sheet, 'Long Term Debt', 0)
            long_debt_t1 = safe_get(balance_sheet, 'Long Term Debt', 1)
            debt_ratio_t = long_debt_t / total_assets_t if total_assets_t else 0
            debt_ratio_t1 = long_debt_t1 / total_assets_t1 if total_assets_t1 else 0
            if debt_ratio_t < debt_ratio_t1:
                score += 1
            
            # 6. Current Ratio increased
            current_assets_t = safe_get(balance_sheet, 'Current Assets', 0)
            current_assets_t1 = safe_get(balance_sheet, 'Current Assets', 1)
            current_liab_t = safe_get(balance_sheet, 'Current Liabilities', 0)
            current_liab_t1 = safe_get(balance_sheet, 'Current Liabilities', 1)
            current_ratio_t = current_assets_t / current_liab_t if current_liab_t else 0
            current_ratio_t1 = current_assets_t1 / current_liab_t1 if current_liab_t1 else 0
            if current_ratio_t > current_ratio_t1:
                score += 1
            
            # 7. No new shares issued
            shares_t = safe_get(balance_sheet, 'Share Issued', 0) or safe_get(balance_sheet, 'Ordinary Shares Number', 0)
            shares_t1 = safe_get(balance_sheet, 'Share Issued', 1) or safe_get(balance_sheet, 'Ordinary Shares Number', 1)
            if shares_t and shares_t1 and shares_t <= shares_t1:
                score += 1
            
            # 8. Gross Margin increased
            gross_profit_t = safe_get(income_stmt, 'Gross Profit', 0)
            gross_profit_t1 = safe_get(income_stmt, 'Gross Profit', 1)
            gm_t = gross_profit_t / revenue_t if revenue_t else 0
            gm_t1 = gross_profit_t1 / revenue_t1 if revenue_t1 else 0
            if gm_t > gm_t1:
                score += 1
            
            # 9. Asset Turnover increased
            asset_turnover_t = revenue_t / total_assets_t if total_assets_t else 0
            asset_turnover_t1 = revenue_t1 / total_assets_t1 if total_assets_t1 else 0
            if asset_turnover_t > asset_turnover_t1:
                score += 1
            
            return int(score)
        
        except Exception as e:
            logger.error("Error calculating Piotroski F-Score: %s", e)
            return None
