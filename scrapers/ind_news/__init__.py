# Centurion Capital LLC - Indian Financial News Scrapers
"""
Modular scrapers for major Indian financial news platforms.

Sources:
    - Moneycontrol
    - The Economic Times (Markets / Stocks)
    - Mint (Markets / Companies)
    - Business Standard (Markets / Companies)
    - The Hindu Business Line
    - Pulse by Zerodha
    - NDTV Profit
"""

from scrapers.ind_news.moneycontrol import MoneycontrolScraper
from scrapers.ind_news.economic_times import EconomicTimesScraper
from scrapers.ind_news.livemint import LiveMintScraper
from scrapers.ind_news.business_standard import BusinessStandardScraper
from scrapers.ind_news.hindu_businessline import HinduBusinessLineScraper
from scrapers.ind_news.zerodha_pulse import ZerodhaPulseScraper
from scrapers.ind_news.ndtv_profit import NDTVProfitScraper
from scrapers.ind_news.google_news_india import GoogleNewsIndiaScraper

__all__ = [
    "MoneycontrolScraper",
    "EconomicTimesScraper",
    "LiveMintScraper",
    "BusinessStandardScraper",
    "HinduBusinessLineScraper",
    "ZerodhaPulseScraper",
    "NDTVProfitScraper",
    "GoogleNewsIndiaScraper",
]
