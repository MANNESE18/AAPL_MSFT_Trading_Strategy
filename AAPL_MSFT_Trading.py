import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# bring in necessary files and set them up to use
# renanmes columns in MSFT dataframe to match AAPL Dataframe
AAPL_file = input("Enter Apple file:").strip()
MSFT_file = input("Enter Microsoft file:").strip()
AAPL = pd.read_csv(AAPL_file)
MSFT = pd.read_csv(MSFT_file)

# used this to check data format (adj close was float64)
# print(AAPL.info())
# print(MSFT.info())
AAPL['Date'] = pd.to_datetime(AAPL['Date'])
AAPL = AAPL[AAPL['Date'].dt.year.between(2014,2021)]                             
MSFT['Date'] = pd.to_datetime(MSFT['Date'])
MSFT = MSFT[MSFT['Date'].dt.year.between(2014,2021)]

AAPL_MSFT = pd.merge(AAPL, MSFT, on='Date', how='inner', suffixes=('_AAPL', '_MSFT'))

# creates new tables using a copy of original dataframe
AM_20_1 = AAPL_MSFT.copy()
AM_60_2 = AAPL_MSFT.copy()

# calc three parts of z-score and overall z-score based on spread of two stock prices
# added extra lines to calc spread b/c I want the spread number that results from -> (larger/smaller)
AM_20_1['AM_Comp'] = AAPL_MSFT['Adj Close_AAPL']/AAPL_MSFT['Adj Close_MSFT']
AM_20_1['MA_Comp'] = AAPL_MSFT['Adj Close_MSFT']/AAPL_MSFT['Adj Close_AAPL']
AM_20_1['Spread'] = np.maximum(AM_20_1['AM_Comp'], AM_20_1['MA_Comp'])
AM_20_1['Spread_mean'] = AM_20_1['Spread'].ewm(span=20).mean()
AM_20_1['Spread_std'] = AM_20_1['Spread'].ewm(span=20).std()
AM_20_1['Z-Score']= (AM_20_1['Spread'] - AM_20_1['Spread_mean'])/(AM_20_1['Spread_std'])

# calc blank variables to show which of stock prices is higher when the signal shows a '1'
AM_20_1['Higher'] = np.maximum(AM_20_1['Adj Close_AAPL'], AM_20_1['Adj Close_MSFT'])
AM_20_1['Lower'] = np.minimum(AM_20_1['Adj Close_AAPL'], AM_20_1['Adj Close_MSFT'])
AM_20_1['Signal'] = np.where(AM_20_1['Z-Score'] >= 1,1, 0)

# finds weights for investment based on std. dev. of stocks
AM_20_1['Daily_Return_for_Higher'] = AM_20_1['Higher'].pct_change()
AM_20_1['Daily_Return_for_Lower'] = AM_20_1['Lower'].pct_change()
AM_20_1['std_for_Higher'] = AM_20_1['Daily_Return_for_Higher'].ewm(span=20).std()
AM_20_1['std_for_Lower'] = AM_20_1['Daily_Return_for_Lower'].ewm(span=20).std()
AM_20_1['inv_std_sum'] = (1/AM_20_1['std_for_Higher']) + (1/AM_20_1['std_for_Lower'])
AM_20_1['Weight_for_Higher'] = (1/AM_20_1['std_for_Higher']) / (AM_20_1['inv_std_sum'])
AM_20_1['Weight_for_Lower'] = (1/AM_20_1['std_for_Lower']) / (AM_20_1['inv_std_sum'])

# calc to "long" the lower stock and "short" the higher stock
AM_20_1['Strategy'] = ((AM_20_1['Higher'].pct_change())*(AM_20_1['Signal']).shift(1))*(AM_20_1['Weight_for_Higher'].shift(1))*(-1) + ((AM_20_1['Lower'].pct_change())*(AM_20_1['Signal']).shift(1))*(AM_20_1['Weight_for_Lower'].shift(1))

# calculates Sharpe Ratio
Strategy_Daily = AM_20_1['Strategy'].fillna(0)
Strategy_Daily_Mean = Strategy_Daily.mean()
Strategy_Daily_Std = Strategy_Daily.std()
if Strategy_Daily_Std != 0:
    Sharpe_20_1 = (Strategy_Daily_Mean/Strategy_Daily_Std)*np.sqrt(252)
else:
    Sharpe_20_1 = 0
Sharpe_text_20_1 = f'Annualized Sharpe: {Sharpe_20_1: .2f}'

# uses "1+" to allow returns to compound, and then take it out at end w/ "-1" to leave the investment return
AM_20_1['Total_Return'] = ((1 + AM_20_1['Strategy'].fillna(0)).cumprod()-1)*100

# filters through 2014-2021 data and excludes 2014 (used 2014 to generate data so there was no lapse to start 2015)
AM_20_1_Chart = AM_20_1[AM_20_1['Date'].dt.year >= 2015].copy()

# same steps for AM_60_2 (adjust span and z score threshold)
AM_60_2['AM_Comp'] = AAPL_MSFT['Adj Close_AAPL']/AAPL_MSFT['Adj Close_MSFT']
AM_60_2['MA_Comp'] = AAPL_MSFT['Adj Close_MSFT']/AAPL_MSFT['Adj Close_AAPL']
AM_60_2['Spread'] = np.maximum(AM_60_2['AM_Comp'], AM_60_2['MA_Comp'])                            
AM_60_2['Spread_mean'] = AM_60_2['Spread'].ewm(span=60).mean()
AM_60_2['Spread_std'] = AM_60_2['Spread'].ewm(span=60).std()
AM_60_2['Z-Score']= (AM_60_2['Spread'] - AM_60_2['Spread_mean'])/(AM_60_2['Spread_std'])

AM_60_2['Higher'] = np.maximum(AM_60_2['Adj Close_AAPL'], AM_60_2['Adj Close_MSFT'])
AM_60_2['Lower'] = np.minimum(AM_60_2['Adj Close_AAPL'], AM_60_2['Adj Close_MSFT'])

AM_60_2['Signal'] = np.where(AM_60_2['Z-Score'] >= 2,1, 0)
                       
AM_60_2['Daily_Return_for_Higher'] = AM_60_2['Higher'].pct_change()
AM_60_2['Daily_Return_for_Lower'] = AM_60_2['Lower'].pct_change()
AM_60_2['std_for_Higher'] = AM_60_2['Daily_Return_for_Higher'].ewm(span=60).std()
AM_60_2['std_for_Lower'] = AM_60_2['Daily_Return_for_Lower'].ewm(span=60).std()
AM_60_2['inv_std_sum'] = (1/AM_60_2['std_for_Higher']) + (1/AM_60_2['std_for_Lower'])
AM_60_2['Weight_for_Higher'] = (1/AM_60_2['std_for_Higher']) / (AM_60_2['inv_std_sum'])
AM_60_2['Weight_for_Lower'] = (1/AM_60_2['std_for_Lower']) / (AM_60_2['inv_std_sum'])
                       
AM_60_2['Strategy'] = ((AM_60_2['Higher'].pct_change())*(AM_60_2['Signal']).shift(1))*(AM_60_2['Weight_for_Higher'].shift(1))*(-1) + ((AM_60_2['Lower'].pct_change())*(AM_60_2['Signal']).shift(1))*(AM_60_2['Weight_for_Lower'].shift(1))
Strategy_Daily = AM_60_2['Strategy'].fillna(0)
Strategy_Daily_Mean = Strategy_Daily.mean()
Strategy_Daily_Std = Strategy_Daily.std()
if Strategy_Daily_Std != 0:
    Sharpe_60_2 = (Strategy_Daily_Mean/Strategy_Daily_Std)*np.sqrt(252)
else:
    Sharpe_60_2 = 0
Sharpe_text_60_2 = f'Annualized Sharpe: {Sharpe_60_2: .2f}'

AM_60_2['Total_Return'] = ((1 + AM_60_2['Strategy'].fillna(0)).cumprod()-1)*100
AM_60_2_Chart = AM_60_2[AM_60_2['Date'].dt.year >= 2015].copy()

# set up a chart for each trading strategy
# added big title above all four charts
fig, ((a,b),(c,d)) = plt.subplots(2,2, figsize = (12,10), sharex=True)
                                                  
fig.suptitle('Z-Score Strategy Investment Returns', fontsize=16)

# set up chart layout / info for Z-Score and each trading strategy
# added Sharpe Ratio to trading charts
AM_20_1_Chart.plot(ax=a, x = 'Date', y='Z-Score')
a.set_title('Z-Score Signal: 20-Day Tactical Window')
a.set_xlabel('Year')
a.set_ylabel('Z-Score')
a.axhline(1, color='red', linestyle='--', linewidth=1.5, label='Sell/Short Threshold (2σ)')
a.axhline(-1, color='green', linestyle='--', linewidth=1.5, label='Sell/Short Threshold (2σ)')
a.axhline(0, color='black', linewidth=1)
a.grid(True, alpha=.5)

AM_60_2_Chart.plot(ax=b, x= 'Date', y='Z-Score')
b.set_title('Z-Score Signal: 60-Day Structural Window')
b.set_xlabel('Year')
b.set_ylabel('Z-Score')
b.axhline(2, color='red', linestyle='--', linewidth=1.5, label='Sell/Short Threshold (2σ)')
b.axhline(-2, color='green', linestyle='--', linewidth=1.5, label='Sell/Short Threshold (2σ)')
b.axhline(0, color='black', linewidth=1)
b.grid(True, alpha=.5)

AM_20_1_Chart.plot(ax=c, x='Date', y='Total_Return')
c.set_title('Performance: Aggressive Mean Reversion (Z=1)')
c.set_xlabel('Year')
c.set_ylabel('Total Return (%)')
c.text(0.02, 0.95, Sharpe_text_20_1, transform = c.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
c.fill_between(AM_20_1['Date'], c.get_ylim()[0], c.get_ylim()[1], where = (AM_20_1['Signal'] !=0), color='gray', alpha=0.15, label='In Market')
c.axhline(0, color='black', linewidth=1, alpha=0.5)
c.grid(True, alpha=.5)

AM_60_2_Chart.plot(ax=d, x='Date', y='Total_Return')
d.set_title('Performance: Institutional Sniper (Z=2)')
d.set_xlabel('Year')
d.set_ylabel('Total Return (%)')
d.text(0.02, 0.95, Sharpe_text_60_2, transform = d.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
d.fill_between(AM_60_2['Date'], d.get_ylim()[0], d.get_ylim()[1], where = (AM_60_2['Signal'] !=0), color='gray', alpha=0.15, label='In Market')
d.axhline(0, color='black', linewidth=1, alpha=0.5)
d.grid(True, alpha=.5)

plt.tight_layout()
plt.show()




                              
