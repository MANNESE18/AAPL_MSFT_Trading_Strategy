# README: AAPL-MSFT Pairs Trading & Z-Score Strategy

This Python project implements a **Pairs Trading Strategy** between Apple (AAPL) and Microsoft (MSFT). By analyzing the price spread between these two tech giants, the script identifies mean-reversion opportunities. It utilizes two distinct trading windows—a tactical 20-day "Aggressive" approach and a structural 60-day "Institutional" approach—to calculate Z-scores, generate signals, and backtest total returns including annualized Sharpe Ratios.

## Features

* **Dual-Window Analysis**: Compares a fast-moving 20-day window (Z≥1) against a conservative 60-day window (Z≥2) to evaluate different market regimes.

* **Dynamic Spread Calculation**: Automatically identifies the "Higher" and "Lower" valued stock in the pair to calculate a consistent relative spread.

* **Volatility-Adjusted Weighting**: Instead of a simple 50/50 split, the strategy weights the long and short positions based on the inverse of their rolling standard deviation (volatility).

* **Performance Visualization**: Generates a 2x2 dashboard using Matplotlib that displays Z-score fluctuations, entry signals (market overlays), and cumulative total returns.

* **Risk Metrics**: Calculates the **Annualized Sharpe Ratio** for both strategies to assess risk-adjusted performance.

## Built With

* **Python 3.x**: The core programming language.

* **Pandas**: Used for extensive data manipulation, time-series alignment, and merging CSV datasets.

* **NumPy**: Utilized for vectorized mathematical operations, specifically for calculating spreads and signal logic.

* **Matplotlib**: Used to create the professional-grade 4-panel visualization dashboard.

* **Exponential Weighted Functions (EWM)**: Used for calculating rolling means and standard deviations to give more weight to recent price action.

## Key Achievements in Code

**1. Robust Signal Logic**

The code uses `np.maximum` and `np.minimum` to handle "Higher" and "Lower" stock designations dynamically. This ensures the spread is always calculated relative to the dominant price, preventing negative spreads and simplifying the Z-score logic.

**2. Risk-Parity Weighting**

The implementation of "Inverse Volatility Weighting" is a sophisticated touch:

```
Python

AM_20_1['inv_std_sum'] = (1/AM_20_1['std_for_Higher']) + (1/AM_20_1['std_for_Lower'])
AM_20_1['Weight_for_Higher'] = (1/AM_20_1['std_for_Higher']) / (AM_20_1['inv_std_sum'])
This ensures that the more volatile stock in the pair receives a smaller allocation, balancing the risk of the overall trade.
```


**3. Realistic Backtesting**

The strategy applies a `.shift(1)` to the signals and weights before calculating returns. This is a critical best practice in quantitative finance to avoid look-ahead bias, ensuring the trade is executed on the day after the signal is generated.

**4. Automated Visualization Dashboard**

The script doesn't just output numbers; it creates a visual narrative. By using `ax.fill_between` with a `where` clause, the charts highlight exactly when the strategy was "In Market," allowing for immediate visual correlation between Z-score spikes and performance.
