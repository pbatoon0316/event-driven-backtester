
# event-driven-backtester.py

The following is a home-built backtesting framework for evaluating trading strategies using historical stock price data. It utilizes various technical indicators and entry/exit conditions to simulate and analyze trading decisions. The main difference between this backtesting framework and other vector-based programs is that this does not hold overlapping positions within the same ticker. When a trade is entered, no new positions are established until the trade hits stop loss or exit criteria.

#### Dependencies

The following libraries are imported as dependencies:
- `pandas` (imported as `pd`): Used for data manipulation and analysis.
- `datetime` (imported as `dt`): Used for handling dates and times.
- `matplotlib.pyplot` (imported as `plt`): Used for data visualization.
- `math`: Used for mathematical operations.
- `yfinance` (imported as `yf`): Used for downloading stock price data.
- `finta` (imported as `TA`): Provides a collection of technical indicators.
- `tqdm`: Used for displaying a progress bar during the backtesting process.
- `warnings`: Used to suppress warnings generated by `pandas`.

## Main Event-Driven Backtester

The `backtest` function is the main backtesting algorithm. It accepts the following parameters:
- `data`: A list of ticker symbols (default: `['SPY','IWM','QQQ','TLT','VXN']`). These tickers represent the stocks to be backtested. ** Please note that this requires multiple tickers as input (via `yfinance` Multi Column Index. If you would like to backtest just one instrument, please include a second ticker as placeholder.
- `trade_size`: The amount of capital allocated per trade (default: `1000`).
- `max_loss`: The maximum acceptable loss per trade as a fraction of the trade size (default: `0.05`). For example, 5% of trade size.

The backtest function performs the following steps:

1. Extracts the ticker symbols from the input data.
2. Initializes an empty DataFrame (`df_backtest`) to store the backtest results.
3. Loops through each ticker symbol.
4. Copies the data for the current ticker symbol and performs necessary data transformations.
5. Initializes variables and DataFrames used for tracking trade positions.
6. Calculates additional parameters and technical indicators for the current ticker symbol.
7. Defines the entry and exit conditions for trades based on the calculated indicators.
8. Loops through each date in the data.
   - If no position is currently open, checks if the entry conditions are met. If so, opens a long position.
   - If a position is open, checks if the stop-loss condition is met or if the exit conditions are met. If so, closes the position.
   - If none of the above conditions are met and it is the last date in the data, closes the position (flattens).
   - Updates trade-related variables and logs the trade details in the `df_trades` DataFrame.
9. Concatenates the `df_trades` DataFrame for the current ticker symbol with the overall `df_backtest` DataFrame.
10. Calculates the total profit and loss (`pnl_total`) and the number of trades (`trades`) for the current ticker symbol.
11. Plots a histogram of the profit and loss values for the overall backtest results (`df_backtest`).

After the loop through all ticker symbols, the function prints a completion message and returns the `df_backtest` DataFrame.

## Sample Backtest Results

The `sample_backtest` function is used to perform statistical sampling on the backtest results. This helps estimate the distribution of possible outcomes (worst case to best case) and helps establish statistical significance if considering a reasonable confidence interval. It accepts the following parameters:
- `df_backtest`: The DataFrame containing the backtest results.
- `iterations`: The number of iterations to perform (default: `1000`).
- `samples`: The number of samples to take in each iteration (default: `100`). -- How many trades within a given time period?

The function performs the following steps:

1. Initializes an empty DataFrame (`df_backtest_sample`) to store the sampled results.
2. Iterates through the specified number of iterations.
3. In each iteration, randomly samples the specified number of samples from the `df_backtest` DataFrame with replacement.
4. Calculates descriptive statistics (mean) for the sampled data and appends it to the `df_backtest_sample` DataFrame.
5. Plots a histogram of the sampled profit and loss values from the `df_backtest_sample` DataFrame.

The function returns the `df_backtest_sample` DataFrame.

## Example of Use - Backtesting the TTM Squeeze

The provided example demonstrates the usage of the backtesting framework.

1. It obtains a list of ticker symbols from a CSV file. This uses the stock screener CSV format from nasdaq.com.
2. It runs the backtest using the specified trade size and maximum loss.
3. It filters out trades with zero profit and loss. (optional)
4. It adjusts the recorded maximum losses to match the requested maximum loss. (optional)
5. It plots the sampled backtest results.

### Configuration
Adapting the backtest for TTM Squeeze requires additional parameters such as a Keltner Channel and Bollinger Bands. In this specific case, a squeeze is defined as "Upper BB < Upper KC" or " Lower BB > Lower KC", denoted by red dots in the Squeeze indicator below.
![image](https://github.com/pbatoon0316/event-driven-backtester/assets/118654860/023db304-6233-4b47-91c2-c7b9259663d3)

```
### Calculate parameters and indicators
df['%change'] = df['close'].pct_change()
df['atr'] = TA.ATR(df, period=14)
df['ema_stop'] = TA.EMA(df, period=ema_stop)
df['ema20'] = TA.EMA(df, period=20)
df['ema100'] = TA.EMA(df, period=100)
df['ema200'] = TA.EMA(df, period=200)
df['RSI'] = TA.RSI(df, period=14)
df['smooth RSI'] = df['RSI'].rolling(5).mean()

## TTM Squeeze Indicators - KC and BB
df['kc_upper'] = df['ema20'] + 1.5*df['atr']
df['kc_lower'] = df['ema20'] - 1.5*df['atr']
df[['bb_upper','bb_mid','bb_lower']] = TA.BBANDS(df,  20)
df['squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])

### Entry and exit conditions
# Entry
c_squeeze = (df['squeeze']==False) & (df['squeeze'].shift(1)==True) & (df['squeeze'].shift(2)==True)
c_up = df['%change'] > 0
c_rsi = df['smooth RSI'] > 50
c_entry = c_squeeze & c_up & c_rsi

# Exit
c_exit = df['low'] < df['ema_stop']
c_max_loss = trade_size * max_loss
```
Positions are opened by purchasing at the `close` of each daily bar 
```
### Loop through dates
for i in  range(len(df)):
  if deploy_capital == True:
    if  (c_entry[i] == True) & (df['close'].iloc[i]<1000):
      price = df['close'].iloc[i]
```
  


### Running the backtest
```
#%% Obtain list of tickers from database (extracted from NASDAQ) and download stock data
url = 'https://gist.githubusercontent.com/pbatoon0316/1b45f69402cf56e8174ad2034b62db2a/raw/a78376ab57e85d288bfcb3e832ca766ab81aa4ff/nasdaq_nyse_amex_tickers_20242801.csv'
stocks = pd.read_csv(url)
tickers = stocks['Symbol'].tolist()

#%% Run backtest to generate population of trades
size = 1000       # $1000 if capital deployed
max_loss = 0.05   # $50 of risk before stop loss hit

df_backtest = backtest(data, trade_size=size, max_loss=max_loss)
df_backtest = df_backtest.loc[df_backtest['pnl'] != 0].copy()                  # For exploratory data analysis I like studying the exits. This line is optional
df_backtest.loc[df_backtest['pnl'] < -max_loss*size,'pnl'] = -max_loss*size    # Sometimes the stoploss executes imprecisely, so recorded max losses are larger than requested. This line is optional

#%% Plot sampled backtest
df_backtest_sample = sample_backtest(df_backtest, iterations=1000, samples=100)
```
### Results
The results indicate that over a 5-year backtest period, it was able to execute 22,887 unique trade events with a 41.1% winrate. Since we're able to control our max loss via a Stop Loss cutoff, the results are highly "right skewed" with a positive pnl. In other words, certain trades may have outsized risk:reward

By sampling the data 1000 times over 100 potential trades, you can see a distribution of possible outcomes. While the `pnl` expectation value is positive, there is still a realm of possibility that this strategy results in negative returns.
```
-Backtest Complete-
# Trades  = 22887
Winrate = 41.1%
Total P&L = $85024.63
Mean P&L  = $3.72
Median P&L  = $-4.93
```
![image](https://github.com/pbatoon0316/event-driven-backtester/assets/118654860/72f0f2ac-c404-4276-b26b-4391abd8b2ff)
![image](https://github.com/pbatoon0316/event-driven-backtester/assets/118654860/a9b39369-7c78-4c53-8c9a-b8498b02cd63)

