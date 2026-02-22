"""
Kite Connect - Login & Stock List Generator.

Uses the shared ``kite_session`` module for login and the shared
``config`` module for credentials.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth.kite_session import create_kite_session

# The request_token line below is kept because ``kite_auth.py``
# updates it in-place via regex.  Do NOT remove this line.
request_token='Jy5Q4oB7veIgUT8Vkghjb5ltvCtF8Ikl'

kite = None  # module-level reference set by zerodha_login()


def zerodha_login():
    """Authenticate with Kite Connect and return the user profile."""
    global kite
    logging.basicConfig(level=logging.DEBUG)
    kite = create_kite_session()
    return kite.profile()


def add_stocks():
    """Return a list of NSE equity instrument names (excluding indices, SDLs, etc.)."""
    stock_list = []
    instrument = kite.instruments(exchange="NSE")
    for stock in instrument:
        if stock['segment'] == 'INDICES' or stock['name'] == '' or stock['name'].startswith('SDL') or stock['name'].startswith('2.5%') or stock['name'].startswith('2.50%'):
            continue
        stock_list.append(stock['name'])
    return stock_list


if __name__ == "__main__":
    zerodha_login()
    stocks = sorted(add_stocks())
    with open('stocks.txt', 'w') as f:
        f.write('\n'.join(stocks))