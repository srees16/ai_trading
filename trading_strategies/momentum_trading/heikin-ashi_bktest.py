"""heikin ashi is a Japanese way to filter out the noise for momentum trading
it can prevent the occurrence of sideway chops
basically we do a few transformations on four key benchmarks - Open, Close, High, Low
apply some unique rules on ha Open, Close, High, Low to trade
details of heikin ashi indicators and rules can be found in the following link
https://quantiacs.com/Blog/Intro-to-Algorithmic-Trading-with-Heikin-Ashi.aspx
"""
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import scipy.integrate
import scipy.stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest_utils import mdd, candlestick, portfolio, profit

'''Heikin Ashi has a unique method to filter out the noise
its open, close, high, low require a different approach
please refer to the website mentioned above'''
def heikin_ashi(data):
    
    df=data.copy()
    
    df.reset_index(inplace=True)
        
    #heikin ashi close
    df['HA close']=(df['Open']+df['Close']+df['High']+df['Low'])/4

    #initialize heikin ashi open
    df['HA open']=float(0)
    df.loc[0, 'HA open']=df['Open'].iloc[0]

    #heikin ashi open
    for n in range(1,len(df)):
        df.at[n,'HA open']=(df['HA open'][n-1]+df['HA close'][n-1])/2
        
    #heikin ashi high/low
    temp=pd.concat([df['HA open'],df['HA close'],df['Low'],df['High']],axis=1)
    df['HA high']=temp.apply(max,axis=1)
    df['HA low']=temp.apply(min,axis=1)

    # Keep only required columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'HA close', 'HA open', 'HA high', 'HA low']]
    
    return df

'''setting up signal generations
trigger conditions can be found from the website mentioned above
they kinda look like marubozu candles
there s a short strategy as well
the trigger condition of short strategy is the reverse of long strategy
you have to satisfy all four conditions to long/short
nevertheless, the exit signal only has three conditions'''
def signal_generation(df,method,stls):
        
    data=method(df)
    
    data['signals']=0

    #i use cumulated sum to check how many positions i have longed
    #i would ignore the exit signal prior if not holding positions
    #i also keep tracking how many long positions i have got
    #long signals cannot exceed the stop loss limit
    data['cumsum']=0

    for n in range(1,len(data)):
        
        #long triggered
        if (data['HA open'][n]>data['HA close'][n] and data['HA open'][n]==data['HA high'][n] and
            np.abs(data['HA open'][n]-data['HA close'][n])>np.abs(data['HA open'][n-1]-data['HA close'][n-1]) and
            data['HA open'][n-1]>data['HA close'][n-1]):
            
            data.at[n,'signals']=1
            data['cumsum']=data['signals'].cumsum()

            #accumulate too many longs
            if data['cumsum'][n]>stls:
                data.at[n,'signals']=0
        
        #exit positions
        elif (data['HA open'][n]<data['HA close'][n] and data['HA open'][n]==data['HA low'][n] and 
        data['HA open'][n-1]<data['HA close'][n-1]):
            
            data.at[n,'signals']=-1
            data['cumsum']=data['signals'].cumsum()
        

            #clear all longs
            #if there are no long positions in my portfolio
            #ignore the exit signal
            if data['cumsum'][n]>0:
                data.at[n,'signals']=-1*(data['cumsum'][n-1])

            if data['cumsum'][n]<0:
                data.at[n,'signals']=0
                
    return data

# candlestick is now imported from backtest_utils

    
'''plotting the backtesting result'''
def plot(df,ticker):    
    
    df.set_index(df['Date'],inplace=True)
    
    #first plot is Heikin-Ashi candlestick
    #use candlestick function and set Heikin-Ashi O,C,H,L
    ax1=plt.subplot2grid((200,1), (0,0), rowspan=120,ylabel='HA price')
    candlestick(df,ax1,titlename='',highcol='HA high',lowcol='HA low',
                opencol='HA open',closecol='HA close',xcol='Date',
                colorup='r',colordown='g')
    plt.grid(True)
    plt.xticks([])
    plt.title('Heikin-Ashi')

    #the second plot is the actual price with long/short positions as up/down arrows
    ax2=plt.subplot2grid((200,1), (120,0), rowspan=80,ylabel='price',xlabel='')
    ax2.plot(df.index,df['Close'],label=ticker)

    #long/short positions are attached to the real close price of the stock
    #set the line width to zero
    #thats why we only observe markers
    ax2.plot(df.loc[df['signals']==1].index,df['Close'][df['signals']==1],marker='^',lw=0,c='g',label='long')
    ax2.plot(df.loc[df['signals']<0].index,df['Close'][df['signals']<0],marker='v',lw=0,c='r',label='short')

    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

# portfolio, profit, and mdd are now imported from backtest_utils

'''omega ratio is a variation of sharpe ratio
the risk free return is replaced by a given threshold
in this case, the return of benchmark
integral is needed to calculate the return above and below the threshold
you can use summation to do approximation as well
it is a more reasonable ratio to measure the risk adjusted return
normal distribution doesnt explain the fat tail of returns
so i use student T cumulated distribution function instead 
to make our life easier, i do not use empirical distribution
the cdf of empirical distribution is much more complex
check wikipedia for more details
https://en.wikipedia.org/wiki/Omega_ratio'''
def omega(risk_free,degree_of_freedom,maximum,minimum):

    y=scipy.integrate.quad(lambda g:1-scipy.stats.t.cdf(g,degree_of_freedom),risk_free,maximum)
    x=scipy.integrate.quad(lambda g:scipy.stats.t.cdf(g,degree_of_freedom),minimum,risk_free)

    z=(y[0])/(x[0])

    return z

'''sortino ratio is another variation of sharpe ratio
the standard deviation of all returns is substituted with standard deviation of negative returns
sortino ratio measures the impact of negative return on return
i am also using student T probability distribution function instead of normal distribution
check wikipedia for more details
https://en.wikipedia.org/wiki/Sortino_ratio'''
def sortino(risk_free,degree_of_freedom,growth_rate,minimum):

    v=np.sqrt(np.abs(scipy.integrate.quad(lambda g:((risk_free-g)**2)*scipy.stats.t.pdf(g,degree_of_freedom),risk_free,minimum)))
    s=(growth_rate-risk_free)/v[0]

    return s

'''stats calculation'''
def stats(portfolio,trading_signals,stdate,eddate,capital0=10000):

    stats=pd.DataFrame([0])

    #get the min and max of return
    maximum=np.max(portfolio['return'])
    minimum=np.min(portfolio['return'])    

    #growth_rate denotes the average growth rate of portfolio 
    #use geometric average instead of arithmetic average for percentage growth
    growth_rate=(float(portfolio['total asset'].iloc[-1]/capital0))**(1/len(trading_signals))-1

    #calculating the standard deviation
    std=float(np.sqrt((((portfolio['return']-growth_rate)**2).sum())/len(trading_signals)))

    #use S&P500 as benchmark
    try:
        benchmark=yf.download('^GSPC',start=stdate,end=eddate,progress=False)
        
        # Flatten MultiIndex columns from newer yfinance versions
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.droplevel(1)

        if benchmark.empty or len(benchmark) == 0:
            raise ValueError("Empty benchmark data")
            
        #return of benchmark
        return_of_benchmark=float(benchmark['Close'].iloc[-1]/benchmark['Open'].iloc[0]-1)
        del benchmark
    except Exception as e:
        print(f"Warning: Could not download benchmark data: {e}")
        print("Using 0% as benchmark return")
        return_of_benchmark = 0.0

    #rate_of_benchmark denotes the average growth rate of benchmark 
    #use geometric average instead of arithmetic average for percentage growth
    rate_of_benchmark=(return_of_benchmark+1)**(1/len(trading_signals))-1

    #backtesting stats
    #CAGR stands for cumulated average growth rate
    stats['CAGR']=stats['portfolio return']=float(0)
    stats.loc[0, 'CAGR']=growth_rate
    stats.loc[0, 'portfolio return']=portfolio['total asset'].iloc[-1]/capital0-1
    stats['benchmark return']=return_of_benchmark
    stats['sharpe ratio']=(growth_rate-rate_of_benchmark)/std
    stats['maximum drawdown']=mdd(portfolio['total asset'])

    '''calmar ratio is sorta like sharpe ratio
    the standard deviation is replaced by maximum drawdown
    it is the measurement of return after worse scenario adjustment
    check wikipedia for more details
    https://en.wikipedia.org/wiki/Calmar_ratio'''
    stats['calmar ratio']=growth_rate/stats['maximum drawdown']
    stats['omega ratio']=omega(rate_of_benchmark,len(trading_signals),maximum,minimum)
    stats['sortino ratio']=sortino(rate_of_benchmark,len(trading_signals),growth_rate,minimum)

    #note that i use stop loss limit to limit the numbers of longs
    #and when clearing positions, we clear all the positions at once
    #so every long is always one, and short cannot be larger than the stop loss limit
    stats['numbers of longs']=trading_signals['signals'].loc[trading_signals['signals']==1].count()
    stats['numbers of shorts']=trading_signals['signals'].loc[trading_signals['signals']<0].count()
    stats['numbers of trades']=stats['numbers of shorts']+stats['numbers of longs']  

    #to get the total length of trades
    #given that cumsum indicates the holding of positions
    #we can get all the possible outcomes when cumsum doesnt equal zero
    #then we count how many non-zero positions there are
    #we get the estimation of total length of trades
    stats['total length of trades']=trading_signals['signals'].loc[trading_signals['cumsum']!=0].count()
    stats['average length of trades']=stats['total length of trades']/stats['numbers of trades']
    stats['profit per trade']=float(0)
    stats.loc[0, 'profit per trade']=(portfolio['total asset'].iloc[-1]-capital0)/stats['numbers of trades'].iloc[0]

    del stats[0]
    print(stats)

def main():
    
    #initializing

    #stop loss positions, the maximum long positions we can get
    #without certain constraints, you will long indefinites times 
    #as long as the market condition triggers the signal
    #in a whipsaw condition, it is suicidal
    stls=3
    ticker='NVDA'
    stdate='2015-04-01'
    eddate='2018-02-15'

    #slicer is used for plotting
    #a three year dataset with 750 data points would be too much
    slicer=700

    #downloading data
    df=yf.download(ticker,start=stdate,end=eddate)
    
    # Check if download succeeded
    if df.empty or len(df) == 0:
        print(f"Error: Failed to download data for {ticker}. Please check your internet connection or try again later.")
        return
    
    # Flatten MultiIndex columns from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    trading_signals=signal_generation(df,heikin_ashi,stls)

    viz=trading_signals[slicer:]
    plot(viz,ticker)

    portfolio_details=portfolio(viz)
    profit(portfolio_details)

    stats(portfolio_details,trading_signals,stdate,eddate)

    #note that this is the only py file with complete stats calculation
    
    
    
if __name__ == '__main__':
    main()