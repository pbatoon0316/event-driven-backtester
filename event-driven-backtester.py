#%% Dependencies
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math

import yfinance as yf               # For downloading data
from finta import TA                # For easily managing Technical Indicators
from tqdm import tqdm               # For making a nice progress bar

import warnings                     # To supress pandas warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


##################################
#%% Main event driven backtester
##################################
def backtest(data, trade_size=1000, max_loss=0.05):

    # Get tickers
    tickers = list(data.columns.get_level_values(1).unique())

    # Initialize output df
    df_backtest = pd.DataFrame()

    # Loop through tickers
    for ticker in tqdm(tickers,desc='Running Backtest'):

        df = data.loc[:, (slice(None), ticker)].copy()
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
      
        # Intialize or reset variable & df for entry/exit position
        df_trades = pd.DataFrame()
        deploy_capital = True
        trade_size = trade_size
        pnl = 0
        pnl_total = 0
        trades = 0
      
        ##################################
        ### Calculate parameters and indicators - Here you can include your special indicators
        ##################################
        df[['macd_crossover','macd_signal']] = TA.MACD(df)
        df['ema10'] = TA.EMA(df,period=10)
        df['ema20'] = TA.EMA(df,period=20)
        df['ema50'] = TA.EMA(df, period=50)
        df['ema100'] = TA.EMA(df, period=100)
        df['ema200'] = TA.EMA(df,period=200)
        df['RSI'] = TA.RSI(df, period=14)
        df['smooth RSI'] = df['RSI'].rolling(5).mean()
        ##################################

        ##################################
        #%% Entry and exit conditions - This section you can edit to your liking based on your entry/exit conditions.
        ##################################
        # Entry
        c_macd_below_zero = df['macd_signal'] < 0
        c_macd_crossover = df['macd_crossover'] > df['macd_signal']
        c_ema = df['close'] < df['ema200']
        c_rsi = df['smooth RSI'] > 60
        c_entry = c_macd_below_zero & c_macd_crossover & c_ema & c_rsi

        # Exit
        c_exit = df['close'] < df['ema20']
        c_max_loss = trade_size * max_loss
        ##################################

        ### Loop through dates
        for i in range(len(df)):

            if deploy_capital == True:

                if c_entry[i] == True:

                    price = df['close'].iloc[i]
                    units = trade_size / price #math.floor(trade_size / price)
                    basis = units * price
                    pnl = 0

                    # Log trade
                    date = str(df.index[i])[:10]
                    open_date = dt.datetime.strptime(date, "%Y-%m-%d")
                    action = 'Open Long'

                    df_position = pd.DataFrame({'date': [date], 'action': [action], 'units': [units],
                                        'execution price': [price], 'cost basis': [-1*basis],
                                        'pnl': [pnl], 'duration':[0], 'ticker': [ticker]})

                    df_trades = pd.concat([df_trades, df_position])

                    # Block trades / deployment of capital
                    deploy_capital = False

                else:
                    # No entry conditions met
                    pass

            elif deploy_capital == False:

                price = df['close'].iloc[i]
                pnl = (units * price) - basis

                # Stop Loss hit
                if pnl < (-1*c_max_loss):

                    basis = units * price
                    date = str(df.index[i])[:10]
                    close_date = dt.datetime.strptime(date, "%Y-%m-%d")
                    trade_duration = (close_date - open_date).days
                    action = 'Stop Loss'

                    # Log trade
                    df_position = pd.DataFrame({'date': [date], 'action': [action], 'units': [units],
                                        'execution price': [price], 'cost basis': [basis],
                                        'pnl': [pnl], 'duration':[trade_duration], 'ticker': [ticker]})
                    df_trades = pd.concat([df_trades, df_position])

                    # Enable trades / deployment of capital
                    deploy_capital = True

                # Exit conditions hit
                elif c_exit[i]:

                    basis = units * price
                    date = str(df.index[i])[:10]
                    close_date = dt.datetime.strptime(date, "%Y-%m-%d")
                    trade_duration = (close_date - open_date).days
                    action = 'Close Long'

                    # Log trade
                    df_position = pd.DataFrame({'date': [date], 'action': [action], 'units': [units],
                                        'execution price': [price], 'cost basis': [basis],
                                        'pnl': [pnl], 'duration':[trade_duration], 'ticker': [ticker]})
                    df_trades = pd.concat([df_trades, df_position])

                    # Enable trades / deployment of capital
                    deploy_capital = True

                # Flatten for end of backtest
                elif i == len(df)-1:

                    basis = units * price
                    date = str(df.index[i])[:10]
                    close_date = dt.datetime.strptime(date, "%Y-%m-%d")
                    trade_duration = (close_date - open_date).days
                    action = 'Flatten'

                    # Log trade
                    df_position = pd.DataFrame({'date': [date], 'action': [action], 'units': [units],
                                        'execution price': [price], 'cost basis': [basis],
                                        'pnl': [pnl], 'duration':[trade_duration], 'ticker': [ticker]})
                    df_trades = pd.concat([df_trades, df_position])

                    # Enable trades / deployment of capital
                    deploy_capital = True

            else:
                pass

        ### Report/print results
        try:
            df_backtest = pd.concat([df_backtest, df_trades])
            pnl_total = df_trades['pnl'].sum()
            trades = len(df_trades) // 2

            #print(f'{ticker}: {trades} trades taken. Total P&L = ${round(pnl_total,2)}')
        except:
            #print(f'{ticker}: 0 trades taken. Total P&L = $0')
            pass
          
    #%% Plot total backtest result
    plt.figure()
    plt.hist(df_backtest['pnl'], bins=200)
    plt.xlim(df_backtest['pnl'].min()*1.1, df_backtest['pnl'].max()*1.1)
    plt.axvline(x=0, ymin=0, ymax=1, linestyle='dashed', color='black')
    plt.ylim(top=50)
    plt.xlabel('P&L ($)')
    plt.ylabel('Occurences')
    plt.show()
  
    print('\n-Backtest Complete-')
  
    return df_backtest

##################################
#%% Sample backtest results. Carry out statistical sampling simulation to establish statistical significance
##################################
def sample_backtest(df_backtest, iterations=1000, samples=100):

  #%% Iteratively sample for statistics
  iterations = iterations
  sample = samples

  df_backtest_sample = pd.DataFrame()
  for i in tqdm(range(iterations),desc='Sampling Population'):
      df_sample = df_backtest[['pnl','duration']].sample(n=sample, replace=True)
      df_sample = df_sample[['pnl','duration']].describe().loc[['mean']]
      df_backtest_sample = pd.concat([df_backtest_sample, df_sample])

  #%% Plot sampling results
  plt.figure()
  plt.title(f'Backtest Results: {iterations} Iterations, n={sample}', loc='left')
  plt.hist(df_backtest_sample['pnl'], bins=100)
  plt.xlim(df_backtest_sample['pnl'].min()*1.1, df_backtest_sample['pnl'].max()*1.1)
  plt.axvline(x=0, ymin=0, ymax=1, linestyle='dashed', color='black')
  plt.xlabel('P&L ($)')
  plt.ylabel('Occurences')
  plt.show()

  return df_backtest_sample


##################################
#%% Example of use
##################################

#%% Run backtest to generate population of trades
size = 1000       # $1000 if capital deploted
max_loss = 0.05   # $50 of risk before stop loss hit

df_backtest = backtest(data, trade_size=size, max_loss=max_loss)
df_backtest = df_backtest.loc[df_backtest['pnl'] != 0].copy()                  # For exploratory data analysis I like studying the exits. This line is optional
df_backtest.loc[df_backtest['pnl'] < -max_loss*size,'pnl'] = -max_loss*size    # Sometimes the stoploss executes imprecisely, so recorded max losses are larger than requested. This line is optional

#%% Plot sampled backtest
df_backtest_sample = sample_backtest(df_backtest, iterations=1000, samples=100)
