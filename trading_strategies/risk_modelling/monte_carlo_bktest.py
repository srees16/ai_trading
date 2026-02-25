"""
ref: https://github.com/je-suis-tm/quant-trading

Monte Carlo Simulation for Stock Price Forecasting

Assuming you already know how monte carlo works.
If not, please see: https://datascienceplus.com/how-to-apply-monte-carlo-simulation-to-forecast-stock-prices-using-python/

Monte Carlo simulation is a buzz word for people outside of financial industry. In the industry, everybody jokes about it but no one actually uses it, including my risk quant friends - they be like why the heck use that. You may argue its application in option pricing to monitor fat tail events, but seriously, did anyone predict 2008 financial crisis? Or did anyone foresee the VIX surging in early 2018?

The weakness of Monte Carlo, perhaps in every forecast methodology, is that our pseudo random number is generated via empirical distribution. In other words, we use the past to predict the future. If something has never happened in the past, how can you predict it with our limited imagination? It's like muggles trying to understand the wizard world. Laplace smoothing is actually better than Monte Carlo in this case.

The idea presented here is very straightforward:
- We construct a model to get mean and variance of its residual (return)
- We generate the next possible price by geometric Brownian motion
- We run this simulation as many times as possible
- Naturally we should acquire a large amount of data in the end
- We pick the forecast that has the least std against the original data series
- We check if the best forecast can predict the future direction (instead of actual price)
- And how well Monte Carlo catches black swans
"""
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import random as rd
from sklearn.model_selection import train_test_split

#generate gradient color dynamically
def get_gradient_colors(n):
    """Generate n gradient colors from yellow to red."""
    colors = []
    for i in range(n):
        r = 255
        g = int(251 - (251 - 74) * i / max(n - 1, 1))
        b = int(119 - (119 - 74) * i / max(n - 1, 1))
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    return colors

'''
This is where the actual simulation happens.
- testsize: denotes how much percentage of dataset would be used for testing
- simulation: denotes the number of simulations
Theoretically speaking, the larger the better. Given the constrained computing power, we have to take a balance point between efficiency and effectiveness.
'''
def monte_carlo(data,testsize=0.5,simulation=100,**kwargs):    
    
    #train test split as usual
    df,test=train_test_split(data,test_size=testsize,shuffle=False,**kwargs)
    forecast_horizon=len(test)
    
    #we only care about close price. If there was dividends issued, we use adjusted close price instead
    df=df.loc[:,['Close']]
        
    #here we use log return
    returnn=np.log(df['Close'].iloc[1:]/df['Close'].shift(1).iloc[1:])
    drift=returnn.mean()-returnn.var()/2
    
    #we use dictionary to store predicted time series
    d={}
    
    #we use geometric brownian motion to compute the next price
    # https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    for counter in range(simulation):
        d[counter]=[df['Close'].iloc[0]]
      
        #we dont just forecast the future
        #we need to compare the forecast with the historical data as well
        #thats why the data range is training horizon plus testing horizon
        for i in range(len(df)+forecast_horizon-1):
         
            #we use standard normal distribution to generate pseudo random number
            #which is sufficient for our monte carlo simulation
            sde=drift+returnn.std()*rd.gauss(0,1)
            temp=d[counter][-1]*np.exp(sde)
        
            d[counter].append(temp.item())
    
    #to determine which simulation is the best fit
    #we use simple criterias, the smallest standard deviation
    #we iterate through every simulation and compare it with actual data
    #the one with the least standard deviation wins
    std=float('inf')
    pick=0
    for counter in range(simulation):
    
        temp=np.std(np.subtract(
                    d[counter][:len(df)],df['Close'].values.flatten()))
        if temp<std:
            std=temp
            pick=counter
    
    return forecast_horizon,d,pick

#result plotting
def plot(df,forecast_horizon,d,pick,ticker):
    
    #the first plot is to plot every simulation
    #and highlight the best fit with the actual dataset
    #we only look at training horizon in the first figure
    ax=plt.figure(figsize=(10,5)).add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(int(len(d))):
        if i!=pick:
            ax.plot(df.index[:len(df)-forecast_horizon], \
                    d[i][:len(df)-forecast_horizon], \
                    alpha=0.05)
    ax.plot(df.index[:len(df)-forecast_horizon], \
            d[pick][:len(df)-forecast_horizon], \
            c='#5398d9',linewidth=5,label='Best Fitted')
    ax.plot(df.index[:len(df)-forecast_horizon], \
            df['Close'].iloc[:len(df)-forecast_horizon].values, \
            c='#d75b66',linewidth=5,label='Actual')
    plt.title(f'Monte Carlo Simulation\nTicker: {ticker}')
    plt.legend(loc=0)
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.show()
    
    #the second figure plots both training and testing horizons
    #we compare the best fitted plus forecast with the actual history
    #the figure reveals why monte carlo simulation in trading is house of cards
    #it is merely illusion that monte carlo simulation can forecast any asset price or direction
    ax=plt.figure(figsize=(10,5)).add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(d[pick],label='Best Fitted',c='#edd170')
    plt.plot(df['Close'].tolist(),label='Actual',c='#02231c')
    plt.axvline(len(df)-forecast_horizon,linestyle=':',c='k')
    plt.text(len(df)-forecast_horizon-50, \
             max(max(df['Close']),max(d[pick])),'Training', \
             horizontalalignment='center', \
             verticalalignment='center')
    plt.text(len(df)-forecast_horizon+50, \
             max(max(df['Close']),max(d[pick])),'Testing', \
             horizontalalignment='center', \
             verticalalignment='center')
    plt.title(f'Training versus Testing\nTicker: {ticker}\n')
    plt.legend(loc=0)
    plt.ylabel('Price')
    plt.xlabel('T+Days')
    plt.show()

'''
We also gotta test if the surge in simulations increases the prediction accuracy.
- simu_start: denotes the minimum simulation number
- simu_end: denotes the maximum simulation number
- simu_delta: denotes how many steps it takes to reach the max from the min
It's kinda like range(simu_start, simu_end, simu_delta)
'''
def test(df,ticker,simu_start=100,simu_end=1000,simu_delta=100,**kwargs):
    
    table=pd.DataFrame()
    table['Simulations']=np.arange(simu_start,simu_end+simu_delta,simu_delta)
    table.set_index('Simulations',inplace=True)
    table['Prediction']=0

    #for each simulation
    #we test if the prediction is accurate
    #for instance
    #if the end of testing horizon is larger than the end of training horizon
    #we denote the return direction as +1
    #if both actual and predicted return direction align
    #we conclude the prediction is accurate
    #vice versa
    for i in np.arange(simu_start,simu_end+1,simu_delta):
        print(i)
        
        forecast_horizon,d,pick=monte_carlo(df,simulation=i,**kwargs)
        
        actual_return=np.sign( \
                              df['Close'].iloc[len(df)-forecast_horizon]-df['Close'].iloc[-1])
        
        best_fitted_return=np.sign(d[pick][len(df)-forecast_horizon]-d[pick][-1])
        table.at[i,'Prediction']=np.where(actual_return==best_fitted_return,1,-1)
        
    #we plot the horizontal bar chart 
    #to show the accuracy does not increase over the number of simulations
    ax=plt.figure(figsize=(10,5)).add_subplot(111)
    ax.spines['right'].set_position('center')
    ax.spines['top'].set_visible(False)

    plt.barh(np.arange(1,len(table)*2+1,2),table['Prediction'], \
             color=get_gradient_colors(len(table)))

    plt.xticks([-1,1],['Failure','Success'])
    plt.yticks(np.arange(1,len(table)*2+1,2),table.index)
    plt.xlabel('Prediction Accuracy')
    plt.ylabel('Times of Simulation')
    plt.title(f"Prediction accuracy doesn't depend on the numbers of simulation.\nTicker: {ticker}\n")
    plt.show()

'''
Let's try something extreme - pick GE, the worst performing stock in 2018. See how Monte Carlo works for both direction prediction and fat tail simulation. Why the extreme? Well, if we are risk quants, we care about Value at Risk, don't we? If quants only look at one sigma event, the portfolio performance would be devastating.
'''
def main():
    
    stdate='2016-01-15'
    eddate='2019-01-15'
    ticker='GE'

    df=yf.download(ticker,start=stdate,end=eddate)
    #flatten MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index=pd.to_datetime(df.index)
    
    if df.empty:
        print(f"Error: No data downloaded for {ticker}. Check ticker symbol, date range, or internet connection.")
        return
    
    forecast_horizon,d,pick=monte_carlo(df)
    plot(df,forecast_horizon,d,pick,ticker)
    test(df,ticker)

if __name__ == '__main__':
    main()