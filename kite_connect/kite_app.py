'''
api key: hzcjwdgbs8wpon7p
api secret: nz978uwv2jmkxbh9t5kf8nittl1cydvy
kite.zerodha.com/connect/login?api_key=hzcjwdgbs8wpon7p

run this for generating request token for each new request:
    https://kite.zerodha.com/connect/login?api_key=hzcjwdgbs8wpon7p
'''

import logging
from kiteconnect import KiteConnect, exceptions as kite_exceptions
from get_request_token import fetch_request_token

api_key='hzcjwdgbs8wpon7p'
api_secret='nz978uwv2jmkxbh9t5kf8nittl1cydvy'
request_token='XIdEK5H6ZUP6KOVRj83H5llqt0W2SEgb'

def zerodha_login():
    global request_token
    global kite
    logging.basicConfig(level=logging.DEBUG)

    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token=request_token, api_secret=api_secret)
        kite.set_access_token(data["access_token"])
        profile = kite.profile()
        return profile
    except (kite_exceptions.TokenException, kite_exceptions.InputException) as e:
        print(f"\n  [!] Token expired or invalid: {e}")
        print("  [!] Launching login flow to fetch a new request token...\n")
        request_token = fetch_request_token()

        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(request_token=request_token, api_secret=api_secret)
        kite.set_access_token(data["access_token"])
        profile = kite.profile()
        return profile

def add_stocks():
    stock_list = []
    instrument = kite.instruments(exchange="NSE")
    for stock in instrument:
        if stock['segment'] == 'INDICES' or stock['name'] == '' or stock['name'].startswith('SDL') or stock['name'].startswith('2.5%') or stock['name'].startswith('2.50%'):
            continue
        stock_list.append(stock['name'])
    return stock_list

zerodha_login()
stocks = sorted(add_stocks())
with open('stocks.txt', 'w') as f:
    f.write('\n'.join(stocks))