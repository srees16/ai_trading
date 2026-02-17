# Resistance/Support AND Candles Patterns

import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

# Fetch data for quantum computing stocks using yfinance
tickers = ['RGTI', 'QBTS', 'IONQ', 'NBIS']
ticker = tickers[0]  # Use RGTI as the primary ticker for backtesting

df = yf.download(ticker, start='2021-01-01', end='2025-01-01', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Rename columns to lowercase for compatibility
df.columns = [c.lower() for c in df.columns]
df = df.reset_index()
df = df.rename(columns={'Date': 'local time', 'index': 'local time'})

# Add volume column if missing and filter zero volume
if 'volume' in df.columns:
    df = df[df['volume'] != 0]
df.reset_index(drop=True, inplace=True)
df.isna().sum()
df.tail()

# Support and Resistance FUNCTIONS
def support(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.low[i]>df1.low[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.low[i]<df1.low[i-1]):
            return 0
    return 1

def resistance(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.high[i]<df1.high[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.high[i]>df1.high[i-1]):
            return 0
    return 1

length = len(df)
high = list(df['high'])
low = list(df['low'])
close = list(df['close'])
open = list(df['open'])
bodydiff = [0] * length

highdiff = [0] * length
lowdiff = [0] * length
ratio1 = [0] * length
ratio2 = [0] * length

def isEngulfing(l):
    row=l
    bodydiff[row] = abs(open[row]-close[row])
    if bodydiff[row]<0.001:
        bodydiff[row]=0.001      

    bodydiffmin = 0.05  # Adjusted for stock prices
    if (bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
        open[row-1]<close[row-1] and
        open[row]>close[row] and 
        (open[row]-close[row-1])>=-0e-5 and close[row]<open[row-1]): #+0e-5 -5e-5
        return 1

    elif(bodydiff[row]>bodydiffmin and bodydiff[row-1]>bodydiffmin and
        open[row-1]>close[row-1] and
        open[row]<close[row] and 
        (open[row]-close[row-1])<=+0e-5 and close[row]>open[row-1]):#-0e-5 +5e-5
        return 2
    else:
        return 0
       
def isStar(l):
    bodydiffmin = 0.05  # Adjusted for stock prices
    row=l
    highdiff[row] = high[row]-max(open[row],close[row])
    lowdiff[row] = min(open[row],close[row])-low[row]
    bodydiff[row] = abs(open[row]-close[row])
    if bodydiff[row]<0.001:
        bodydiff[row]=0.001
    ratio1[row] = highdiff[row]/bodydiff[row]
    ratio2[row] = lowdiff[row]/bodydiff[row]

    if (ratio1[row]>1 and lowdiff[row]<0.2*highdiff[row] and bodydiff[row]>bodydiffmin):# and open[row]>close[row]):
        return 1
    elif (ratio2[row]>1 and highdiff[row]<0.2*lowdiff[row] and bodydiff[row]>bodydiffmin):# and open[row]<close[row]):
        return 2
    else:
        return 0
    
def closeResistance(l,levels,lim):
    if len(levels)==0:
        return 0
    c1 = abs(df.high[l]-min(levels, key=lambda x:abs(x-df.high[l])))<=lim
    c2 = abs(max(df.open[l],df.close[l])-min(levels, key=lambda x:abs(x-df.high[l])))<=lim
    c3 = min(df.open[l],df.close[l])<min(levels, key=lambda x:abs(x-df.high[l]))
    c4 = df.low[l]<min(levels, key=lambda x:abs(x-df.high[l]))
    if( (c1 or c2) and c3 and c4 ):
        return 1
    else:
        return 0
    
def closeSupport(l,levels,lim):
    if len(levels)==0:
        return 0
    c1 = abs(df.low[l]-min(levels, key=lambda x:abs(x-df.low[l])))<=lim
    c2 = abs(min(df.open[l],df.close[l])-min(levels, key=lambda x:abs(x-df.low[l])))<=lim
    c3 = max(df.open[l],df.close[l])>min(levels, key=lambda x:abs(x-df.low[l]))
    c4 = df.high[l]>min(levels, key=lambda x:abs(x-df.low[l]))
    if( (c1 or c2) and c3 and c4 ):
        return 1
    else:
        return 0

n1=2
n2=2
backCandles=30
signal = [0] * length

for row in range(backCandles, len(df)-n2):
    ss = []
    rr = []
    for subrow in range(row-backCandles+n1, row+1):
        if support(df, subrow, n1, n2):
            ss.append(df.low[subrow])
        if resistance(df, subrow, n1, n2):
            rr.append(df.high[subrow])
    #!!!! parameters - adjusted for stock prices
    if ((isEngulfing(row)==1 or isStar(row)==1) and closeResistance(row, rr, 0.5) ):#and df.RSI[row]<30
        signal[row] = 1
    elif((isEngulfing(row)==2 or isStar(row)==2) and closeSupport(row, ss, 0.5)):#and df.RSI[row]>70
        signal[row] = 2
    else:
        signal[row] = 0

df['signal']=signal

df[df['signal']==2].count()

# Rename columns for backtesting.py compatibility
df.columns = ['local time', 'Open', 'High', 'Low', 'Close', 'Volume', 'signal']
df = df.iloc[100:min(200, len(df))]
df['local time'] = pd.to_datetime(df['local time'])
df = df.set_index('local time')
df

def SIGNAL():
    return df.signal

'''A new strategy needs to extend Strategy class and override its two abstract methods: init() and next().
Method init() is invoked before the strategy is run. Within it, one ideally precomputes in efficient, 
vectorized manner whatever indicators and signals the strategy depends on.
Method next() is then iteratively called by the Backtest instance, once for each data point (data frame row), 
simulating the incremental availability of each new full candlestick bar.

Note, backtesting.py cannot make decisions / trades within candlesticks - any new orders are executed on the
next candle's open (or the current candle's close if trade_on_close=True). 
If you find yourself wishing to trade within candlesticks (e.g. daytrading), you instead need to begin 
with more fine-grained (e.g. hourly) data.'''

class MyCandlesStrat(Strategy):  
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next() 
        if self.signal1==2:
            sl1 = self.data.Close[-1] - self.data.Close[-1] * 0.10  # 10% stop loss
            tp1 = self.data.Close[-1] + self.data.Close[-1] * 0.08  # 8% take profit
            self.buy(sl=sl1, tp=tp1)
        elif self.signal1==1:
            sl1 = self.data.Close[-1] + self.data.Close[-1] * 0.10  # 10% stop loss
            tp1 = self.data.Close[-1] - self.data.Close[-1] * 0.08  # 8% take profit
            self.sell(sl=sl1, tp=tp1)

bt = Backtest(df, MyCandlesStrat, cash=10_000, commission=.002, exclusive_orders=True, finalize_trades=True)
stat = bt.run()
print(stat)

bt.plot()