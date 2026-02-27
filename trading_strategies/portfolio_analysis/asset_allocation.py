import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

#  Fancy graphics
plt.style.use('seaborn-v0_8')

# Getting Yahoo finance data using yfinance
def getdata(tickers, start, end):
    OHLC = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)
        # Flatten MultiIndex columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        OHLC[ticker] = df
    return OHLC

# Assets under consideration - Quantum Computing Stocks
tickers = ['RGTI', 'QBTS', 'IONQ', 'NBIS']

# Download data (use more recent start date for quantum stocks)
data = getdata(tickers, '2021-01-01', '2025-01-01')

ICP = pd.DataFrame({'RGTI': data['RGTI']['Close'],
                    'QBTS': data['QBTS']['Close'],
                    'IONQ': data['IONQ']['Close'],
                    'NBIS': data['NBIS']['Close']}).ffill()

# since last commit, yahoo finance decided to mess up (more) some of the tickers data, so now we have to drop rows...
ICP = ICP.dropna()

BuyHold_RGTI   = ICP['RGTI'] /float(ICP['RGTI'].iloc[0]) -1
BuyHold_QBTS   = ICP['QBTS'] /float(ICP['QBTS'].iloc[0]) -1
BuyHold_IONQ   = ICP['IONQ'] /float(ICP['IONQ'].iloc[0]) -1
BuyHold_NBIS   = ICP['NBIS'] /float(ICP['NBIS'].iloc[0]) -1

BuyHold_25Each = BuyHold_RGTI*(1/4) + BuyHold_QBTS*(1/4) + BuyHold_IONQ*(1/4) + BuyHold_NBIS*(1/4)

plt.figure(figsize=(16,6))
plt.plot(BuyHold_RGTI*100,   label='Buy & Hold RGTI')
plt.plot(BuyHold_QBTS*100,   label='Buy & Hold QBTS')
plt.plot(BuyHold_IONQ*100,   label='Buy & Hold IONQ')
plt.plot(BuyHold_NBIS*100,   label='Buy & Hold NBIS')
plt.plot(BuyHold_25Each*100, label='Buy & Hold 25% Each')
plt.xlabel('Time')
plt.ylabel('Cumulative Return (in %)')
plt.margins(x=0.005,y=0.02)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k')
plt.legend()
plt.show()

ROLL_WINDOW = min(21, len(ICP)-1)  # Adaptive window based on available data
RGTI1Y = ICP['RGTI'] /ICP['RGTI'].shift(ROLL_WINDOW) -1
QBTS1Y = ICP['QBTS'] /ICP['QBTS'].shift(ROLL_WINDOW) -1
IONQ1Y = ICP['IONQ'] /ICP['IONQ'].shift(ROLL_WINDOW) -1
NBIS1Y = ICP['NBIS'] /ICP['NBIS'].shift(ROLL_WINDOW) -1

Each251Y = RGTI1Y*(1/4) + QBTS1Y*(1/4) + IONQ1Y*(1/4) + NBIS1Y*(1/4)

plt.figure(figsize=(16,6))
plt.plot(RGTI1Y*100,   label=f'Rolling {ROLL_WINDOW}-Day Return RGTI')
plt.plot(QBTS1Y*100,   label='  ""  ""  QBTS')
plt.plot(IONQ1Y*100,   label='  ""  ""  IONQ')
plt.plot(NBIS1Y*100,   label='  ""  ""  NBIS')
plt.plot(Each251Y*100, label='  ""  ""  25% Each')
plt.xlabel('Time')
plt.ylabel('Returns (in %)')
plt.margins(x=0.005,y=0.02)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k')
plt.legend()
plt.show()

marr      = 0 #minimal acceptable rate of return (usually equal to the risk free rate)
RGTI1YS   = (RGTI1Y.mean() -marr) /RGTI1Y.std()
QBTS1YS   = (QBTS1Y.mean() -marr) /QBTS1Y.std()
IONQ1YS   = (IONQ1Y.mean() -marr) /IONQ1Y.std()
NBIS1YS   = (NBIS1Y.mean() -marr) /NBIS1Y.std()
Each251YS = (Each251Y.mean()-marr) /Each251Y.std()

print(f'RGTI {ROLL_WINDOW}-Day Buy & Hold Sharpe Ratio =',round(RGTI1YS,2))
print('QBTS     "" "" =',round(QBTS1YS  ,2))
print('IONQ     "" "" =',round(IONQ1YS  ,2))
print('NBIS     "" "" =',round(NBIS1YS  ,2))
print('25% Each "" "" =',round(Each251YS,2))

from scipy.optimize import minimize

def multi(x):
    a, b, c, d = x
    return a, b, c, d   #the "optimal" weights we wish to discover

def maximize_sharpe(x): #objective function
    weights = (RGTI1Y*multi(x)[0] + QBTS1Y*multi(x)[1]
               + IONQ1Y*multi(x)[2] + NBIS1Y*multi(x)[3])
    return -(weights.mean()/weights.std())

def constraint(x):      #since we're not using leverage nor short positions
    return 1 - (multi(x)[0]+multi(x)[1]+multi(x)[2]+multi(x)[3])

cons = ({'type':'ineq','fun':constraint})
bnds = ((0,1),(0,1),(0,1),(0,1))
initial_guess = (1, 0, 0, 0)

'''this algorithm (SLSQP) easly gets stuck on a local
optimal solution, genetic algorithms usually yield better results
so my inital guess is close to the global optimal solution'''

ms = minimize(maximize_sharpe, initial_guess, method='SLSQP',
              bounds=bnds, constraints=cons, options={'maxiter': 10000})

msBuyHoldAll = (BuyHold_RGTI*ms.x[0] + BuyHold_QBTS*ms.x[1]
                + BuyHold_IONQ*ms.x[2] + BuyHold_NBIS*ms.x[3])

msBuyHold1yAll = (RGTI1Y*ms.x[0] + QBTS1Y*ms.x[1]
                   + IONQ1Y*ms.x[2] + NBIS1Y*ms.x[3])

plt.figure(figsize=(16,6))
plt.plot(BuyHold_RGTI*100,   label='Buy & Hold RGTI')
plt.plot(BuyHold_25Each*100, label='  ""  ""  25% of Each')
plt.plot(msBuyHoldAll*100,   label='  ""  ""  Max Sharpe')
plt.xlabel('Time')
plt.ylabel('Cumulative Return (in %)')
plt.margins(x=0.005,y=0.02)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k')
plt.legend()
plt.show()

print('RGTI Weight =',round(ms.x[0]*100,2),'%')
print('QBTS ""     =',round(ms.x[1]*100,2),'%')
print('IONQ ""     =',round(ms.x[2]*100,2),'%')
print('NBIS ""     =',round(ms.x[3]*100,2),'%')
print()
print('Sharpe =',round(msBuyHold1yAll.mean()/msBuyHold1yAll.std(),3))
print()
print('Median yearly excess return over RGTI =',round((msBuyHold1yAll.median()-RGTI1Y.median())*100,1),'%')

def maximize_median_yearly_return(x): #different objective function
    weights = (RGTI1Y*multi(x)[0] + QBTS1Y*multi(x)[1]
               + IONQ1Y*multi(x)[2] + NBIS1Y*multi(x)[3])
    return -(float(weights.median()))

mm = minimize(maximize_median_yearly_return, initial_guess, method='SLSQP',
              bounds=bnds, constraints=cons, options={'maxiter': 10000})

mmBuyHoldAll = (BuyHold_RGTI*mm.x[0] + BuyHold_QBTS*mm.x[1]
                + BuyHold_IONQ*mm.x[2] + BuyHold_NBIS*mm.x[3])

mmBuyHold1yAll = (RGTI1Y*mm.x[0] + QBTS1Y*mm.x[1]
                   + IONQ1Y*mm.x[2] + NBIS1Y*mm.x[3])

plt.figure(figsize=(16,6))
plt.plot(BuyHold_RGTI*100,   label='Buy & Hold RGTI')
plt.plot(BuyHold_25Each*100, label='  ""  ""  25% of Each')
plt.plot(mmBuyHoldAll*100,   label='  ""  ""  Max 1Y Median')
plt.xlabel('Time')
plt.ylabel('Cumulative Return (in %)')
plt.margins(x=0.005,y=0.02)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k')
plt.legend()
plt.show()

print('RGTI Weight =',round(mm.x[0]*100,2),'%')
print('QBTS ""     =',round(mm.x[1]*100,2),'%')
print('IONQ ""     =',round(mm.x[2]*100,2),'%')
print('NBIS ""     =',round(mm.x[3]*100,2),'%')
print()
print('Sharpe =',round(mmBuyHold1yAll.mean()/mmBuyHold1yAll.std(),3))
print()
print('Median yearly excess return over RGTI =',round((mmBuyHold1yAll.median()-RGTI1Y.median())*100,1),'%')

LOOKBACK = min(21, len(ICP)-1)
YTD_RGTI   = ICP['RGTI'].iloc[-LOOKBACK:] /float(ICP['RGTI'].iloc[-LOOKBACK]) -1
YTD_QBTS   = ICP['QBTS'].iloc[-LOOKBACK:] /float(ICP['QBTS'].iloc[-LOOKBACK]) -1
YTD_IONQ   = ICP['IONQ'].iloc[-LOOKBACK:] /float(ICP['IONQ'].iloc[-LOOKBACK]) -1
YTD_NBIS   = ICP['NBIS'].iloc[-LOOKBACK:] /float(ICP['NBIS'].iloc[-LOOKBACK]) -1

YTD_25Each = YTD_RGTI*(1/4) + YTD_QBTS*(1/4) + YTD_IONQ*(1/4) + YTD_NBIS*(1/4)

YTD_max_sharpe = YTD_RGTI*ms.x[0] + YTD_QBTS*ms.x[1] + YTD_IONQ*ms.x[2] + YTD_NBIS*ms.x[3]
YTD_max_median = YTD_RGTI*mm.x[0] + YTD_QBTS*mm.x[1] + YTD_IONQ*mm.x[2] + YTD_NBIS*mm.x[3]

plt.figure(figsize=(15,6))
plt.plot(YTD_RGTI*100,       label=f'Last {LOOKBACK} Days Buy & Hold RGTI')
plt.plot(YTD_25Each*100,     label='  ""  ""  25% of Each')
plt.plot(YTD_max_sharpe*100, label='  ""  ""  Max Sharpe')
plt.plot(YTD_max_median*100, label='  ""  ""  Max Median')
plt.xlabel('Time')
plt.ylabel('Cumulative Return (in %)')
plt.margins(x=0.005,y=0.02)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k')
plt.legend()
plt.show()

print(f'Buy & Hold RGTI {LOOKBACK}-Day Performance =',round(float(YTD_RGTI.iloc[-1]*100),1),'%')
print(' "" "" 25% of Each   "" "" =',round(float(YTD_25Each.iloc[-1]*100),1),'%')
print(' "" "" Max Sharpe    "" "" =',round(float(YTD_max_sharpe.iloc[-1]*100),1),'%')
print(' "" "" Max 1Y Median "" "" =',round(float(YTD_max_median.iloc[-1]*100),1),'%')

ICP['RGTIRet']  = ICP['RGTI'] /ICP['RGTI'].shift(1) -1
ICP['NBISRet']  = ICP['NBIS'] /ICP['NBIS'].shift(1) -1

ICP['Strat'] = ICP['RGTIRet'] * 0.8 + ICP['NBISRet'] * 0.2

ICP.loc[RGTI1Y.shift(1) > -0.17, 'Strat'] = ICP['NBISRet']*0 + ICP['RGTIRet']*1
ICP.loc[NBIS1Y.shift(1) > 0.29, 'Strat']  = ICP['NBISRet']*1 + ICP['RGTIRet']*0

DynAssAll    = ICP['Strat'].cumsum()
DynAssAll1y  = ICP['Strat'].rolling(window=ROLL_WINDOW).sum()
DynAssAllytd = ICP['Strat'].iloc[-LOOKBACK:].cumsum()

plt.figure(figsize=(15,6))
plt.plot(BuyHold_RGTI*100,   label='Buy & Hold RGTI')
plt.plot(mmBuyHoldAll*100,   label='  ""  ""  Max 1Y Median')
plt.plot(DynAssAll*100,      label='Dynamic Asset Allocation')
plt.xlabel('Time')
plt.ylabel('Cumulative Return (in %)')
plt.margins(x=0.005,y=0.02)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k')
plt.legend()
plt.show()

print('Median yearly excess return over RGTI =',round(float(DynAssAll1y.median()-RGTI1Y.median())*100,1),'%')

plt.figure(figsize=(15,6))
plt.plot(YTD_RGTI*100,       label=f'Last {LOOKBACK} Days Buy & Hold RGTI')
plt.plot(YTD_max_median*100, label='  ""  ""  Max Median')
plt.plot(DynAssAllytd*100,   label='Dynamic Asset Allocation')
plt.xlabel('Time')
plt.ylabel('Cumulative Return (in %)')
plt.margins(x=0.005,y=0.02)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='k')
plt.legend()
plt.show()

print(f'Buy & Hold RGTI {LOOKBACK}-Day Performance =',round(float(YTD_RGTI.iloc[-1]*100),1),'%')
print(' "" "" Max Median "" ""  =',round(float(YTD_max_median.iloc[-1]*100),1),'%')
print(' Strategy Performance   =',round(float(DynAssAllytd.iloc[-1]*100),1),'%')