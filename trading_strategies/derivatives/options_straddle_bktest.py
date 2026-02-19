"""after a long while of struggle, i finally decided to write something on options strategy
the biggest issue of options trading is to find the backtesting data
the most difficult part is options greeks
after all, data is the new black gold
here are a couple of websites u can try your luck
currently they offer free trial for a limited period
http://base2.optionsdatamine.com/page.php
https://www.historicaloptiondata.com/
in order to save u guys from the hassle, I also include a small dataset of stoxx 50 index
the dataset has 3 spreadsheets, the spot spreadsheet refers to spot price of stoxx 50
aug spreadsheet refers to options settle at august 2019
jul spreadsheet refers to options settle at july 2019
https://github.com/je-suis-tm/quant-trading/tree/master/data

if you dont know what options straddle is
i recommend u to read a tutorial from fidelity
who else can explain the concept of options than one of the largest mutual funds
https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/long-straddle
in simple words, options are a financial derivative 
that enables u to trade underlying asset at certain price in the future
and options straddle enable you to profit from a certain level of volatility
in this script, we are only gonna talk about long straddle
basically long straddle implies buy call option and put option of same strike price and same strike date
preferably at the same option price as well
otherwise asymmetric option price means there is more one-sided risk than the other
you may wanna consider strangle or strap/strip in this case
short straddle is literally shorting call option and put option of the same strike price and the same strike date
preferably at the same option price as well
long straddle has unlimited profit for upside movement and limited loss
short straddle has unlimited loss for upside movement and limited profit
short straddle is commonly used in a sideway market
long straddle is commonly used in event driven strategy

for instance, brexit on 30th of October 2019, its do or die, no ifs and buts
if bojo delivers a no-deal Brexit, uk sterling gonna sink
or he secures a new deal without backstop from macron and merkel
even though unlikely, uk sterling gonna spike
or he has to postpone and look like an idiot, uk sterling still gonna surge
either way, there will be a lot of volatility around that particular date
to secure a profit from either direction, that is when options straddle kick in

but hey, options are 3 dimensional
apart from strike date, option price, which strike price should we pick
well, that is a one million us dollar question
who says quantitative trading is about algos and calculus?
this is when u need to consult with some economists to get a base case
their fundamental analysis will determine your best/worst scenario
therefore, u can pick a good strike price to maximize your profit
or the simplest way is to find a strike price closer to the current spot price

nevertheless, as u can see in our stoxx 50 dataset
not all strike price offer both call and put options
and even if they offer both, the price of options may be very different
there could be more upside/downside from the market consensus
we can pick the options which offer both call and put options
and we only trade when both option prices are converging
and please don't arrogantly believe that you outsmart the rest of the players in the market
all the information you have obtained from any tips may have already been priced in
finding a good pair of call and put options at the same strike price,
the same strike date and almost the same price is tough

to make our life easier, we only consider european options with cash settlement in this script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import yfinance as yf

'''as we have gathered all the available call and put options
this function will only extract strike price existing in both call and put options
this is a fundamental requirement of options straddle'''

def find_strike_price(df):
    
    temp=[re.search('\d{4}',i).group() for i in df.columns]
    target=[]

    for i in set(temp):
        if temp.count(i)>1:
            target.append(i)
            
    return target

'''this function is merely data cleansing
merging option price information with spot price'''

def straddle(options,spot,contractsize,strikeprice):
        
    option=options[[i for i in options.columns if strikeprice in i]] 
    
    df=pd.merge(spot,option,left_index=True,right_index=True)

    temp=[]
    for i in df.columns:
        if 'C'+strikeprice in i:
            temp.append('call')
        elif 'P'+strikeprice in i:
            temp.append('put')
        elif 'Index' in i:
            temp.append('spot')
        else:
            temp.append(i)

    df.columns=temp
    
    '''we multiply contract size with spot price here
    it makes our life a lot easier later with visualization'''

    df['spot']=df['spot'].apply(lambda x:x*contractsize)
    
    return df

'''signal generation is actually very simple
just find the option pair at the closest price we can'''

def signal_generation(df,threshold):
    
    df['signals']=np.where(
        np.abs(
            df['call']-df['put'])<threshold,
        1,0)  

    return df

'''ploting the payoff diagram'''
def plot(df,strikeprice,contractsize):
    
    '''finding trading signal
    if no signal is found
    we declare no suitable entry point for options straddle'''
    
    ind=df[df['signals']!=0].index

    if ind.empty:
        print('Strike Price at',strikeprice,'\nNo trades available.\n')
        return 
    
    #calculate how much profit we can gain outta this
    
    profit=np.abs(
        df['spot'].iloc[-1]-int(strikeprice)*contractsize
    )-df['call'][ind[0]]-df['put'][ind[0]]

    y=[]
    
    #we use these two variables to plot how much we can profit at different spot price
    
    begin=round(int(strikeprice)*contractsize-5*(df['call'][ind[0]]+df['put'][ind[0]]),0)
    end=round(int(strikeprice)*contractsize+5*(df['call'][ind[0]]+df['put'][ind[0]]),0)+1
    
    x=list(np.arange(int(begin),int(end)))
    
    #as u can see from the pic
    # https://github.com/je-suis-tm/quant-trading/blob/master/preview/options%20straddle%20payoff%20diagram.png
    #we only make money (green color) if the spot price is outside of a range
    #group1 and group2 are variables that indicate which range our line plot gets red/green color
    #they keep track of the indices that we switch from profit to loss or from loss to profit
    #as indices are always positive, we initialize them to negative values

    group1,group2=-10,-10
    for j in x:
        temp=np.abs(j-int(strikeprice)*contractsize)-(df['call'][ind[0]]+df['put'][ind[0]])
        y.append(temp)
        if temp<0 and group1<0:
            group1=x.index(j)
        if temp>0 and group1>0 and group2<0:
            group2=x.index(j)
        

    ax=plt.figure(figsize=(10,5)).add_subplot(111)
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #pnl in different colors, red is loss, green is profit
    
    plt.plot(x[:group1],y[:group1],c='#57bc90',lw=5)
    plt.plot(x[group2:],y[group2:],c='#57bc90',lw=5)
    plt.plot(x[group1:group2],y[group1:group2],c='#ec576b',lw=5)
    
    #ploting strike price
    
    plt.plot([int(strikeprice)*contractsize,
              int(strikeprice)*contractsize],
              [0,-(df['call'][ind[0]]+df['put'][ind[0]])],
              linestyle=':',lw=3,c='#ec576b',alpha=0.5)
    
    #ploting spot price
    
    plt.axvline(df['spot'].iloc[-1],lw=5,
                linestyle='--',c='#e5e338',alpha=0.5)
    
    #adding annotations
    
    plt.annotate('Strike Price',
                 xy=(int(strikeprice)*contractsize,
                     0),
                 xytext=(int(strikeprice)*contractsize,
                     df['call'][ind[0]]+df['put'][ind[0]]),
                 arrowprops=dict(arrowstyle='simple',
                                 facecolor='#c5c1c0',),
                 va='center',ha='center'
                 )
 
    plt.annotate('Lower Breakeven Point',
                 xy=(int(strikeprice)*contractsize-(df['call'][ind[0]]+df['put'][ind[0]]),
                     0),
                 xytext=(int(strikeprice)*contractsize-1.5*(df['call'][ind[0]]+df['put'][ind[0]]),
                         -df['call'][ind[0]]-df['put'][ind[0]]),
                 arrowprops=dict(arrowstyle='simple',
                                 facecolor='#c5c1c0'),
                 va='center',ha='center'
                 )
 
    plt.annotate('Upper Breakeven Point',
                 xy=(int(strikeprice)*contractsize+(df['call'][ind[0]]+df['put'][ind[0]]),
                     0),
                 xytext=(int(strikeprice)*contractsize+1.5*(df['call'][ind[0]]+df['put'][ind[0]]),
                         -df['call'][ind[0]]-df['put'][ind[0]]),
                 arrowprops=dict(arrowstyle='simple',
                                 facecolor='#c5c1c0'),
                 va='center',ha='center'
                 )

    plt.annotate('Spot Price',
                 xy=(df['spot'].iloc[-1],
                     2*(df['call'][ind[0]]+df['put'][ind[0]])),
                 xytext=(df['spot'].iloc[-1]*1.003,
                         2*(df['call'][ind[0]]+df['put'][ind[0]])),
                 arrowprops=dict(arrowstyle='simple',
                                 facecolor='#c5c1c0'),
                 va='center',ha='left'
                 )
    
    #limit x ticks to 3 for a tidy look
    
    plt.locator_params(axis='x',nbins=3)
    
    plt.title(f'Long Straddle Options Strategy\nP&L {round(profit,2)}')
    plt.ylabel('Profit & Loss')
    plt.xlabel('Price',labelpad=50)
    plt.show()

'''for AAPL options, contract size is 1 (per-share basis)
yfinance quotes option prices per share'''

contractsize=1

'''the threshold determines the price disparity between call and put options
the same call and put option price for the same strike price and the same strike date
only exists in an ideal world, in reality, it is like royal flush
when the price difference of call and put is smaller than 10 dollars
we consider them close enough for a straddle entry'''

threshold=10

def main():
    
    from datetime import datetime, timedelta
    
    ticker_symbol='AAPL'
    stdate='2020-01-01'
    eddate='2022-12-31'
    
    '''fetch current options chain from yfinance first
    note: yfinance does not provide historical options data
    current option prices are used for demonstration'''
    ticker=yf.Ticker(ticker_symbol)
    exp_dates=ticker.options
    
    if not exp_dates:
        print("No options expiration dates available.")
        return
    
    # Use an expiry at least 30 days out for meaningful time value
    target_date=datetime.now()+timedelta(days=30)
    exp_date=None
    for d in exp_dates:
        if datetime.strptime(d,'%Y-%m-%d')>=target_date:
            exp_date=d
            break
    if exp_date is None:
        exp_date=exp_dates[-1]
    print(f"Using options expiration: {exp_date}")
    
    chain=ticker.option_chain(exp_date)
    
    # Filter to integer strikes with valid lastPrice
    all_calls=chain.calls[(chain.calls['strike'] % 1 == 0) & (chain.calls['lastPrice']>0)].copy()
    all_puts=chain.puts[(chain.puts['strike'] % 1 == 0) & (chain.puts['lastPrice']>0)].copy()
    
    # Find strikes available in both calls and puts
    common_strikes=sorted(set(all_calls['strike'].astype(int)) & set(all_puts['strike'].astype(int)))
    
    if not common_strikes:
        print("No common integer strike prices found.")
        return
    
    print(f"Found {len(common_strikes)} common strikes: {common_strikes}")
    
    '''download AAPL spot data from yfinance'''
    spot_raw=yf.download(ticker_symbol,start=stdate,end=eddate)
    
    # Flatten MultiIndex columns from newer yfinance versions
    if isinstance(spot_raw.columns, pd.MultiIndex):
        spot_raw.columns = spot_raw.columns.droplevel(1)
    
    if spot_raw.empty:
        print(f"Error: Failed to download spot data for {ticker_symbol}")
        return
    
    # Create spot DataFrame with 'Index' in column name (required by straddle function)
    spot=pd.DataFrame(index=spot_raw.index)
    spot['AAPL Index']=spot_raw['Close']
    
    # Build options DataFrame with columns matching expected format
    options_data={}
    for strike in common_strikes:
        strike_str=f'{strike:04d}'
        call_price=all_calls[all_calls['strike']==strike]['lastPrice'].values[0]
        put_price=all_puts[all_puts['strike']==strike]['lastPrice'].values[0]
        options_data[f'AAPL C{strike_str} {exp_date}']=call_price
        options_data[f'AAPL P{strike_str} {exp_date}']=put_price
    
    # Replicate current option prices across all spot dates
    # Historical options data is not available via yfinance
    options=pd.DataFrame(options_data,index=spot.index)
    
    target=find_strike_price(options)
    
    if not target:
        print("No valid strike prices found for straddle analysis.")
        return
    
    '''we iterate through all the available option pairs
    to find the optimal strike price to maximize our profit'''
    
    for strikeprice in target:
      
        df=straddle(options,spot,contractsize,strikeprice)
        
        signal=signal_generation(df,threshold)
        
        plot(signal,strikeprice,contractsize)

if __name__ == '__main__':
    main()