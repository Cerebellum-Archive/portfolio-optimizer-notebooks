import os
import warnings
from datetime import datetime
import time
import pytz
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import hashlib
import quantstats as qs

# Scikit-learn and statsmodels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as rmse, mean_absolute_error as mae, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import sys
sys.path.append('/home/drew/projects/bwmacro/portfolio-optimizer-notebooks/utils')

import PortfolioOptimizer

# Utility functions from our custom library
from utils_simulate import (
    simplify_teos, log_returns, generate_train_predict_calender,
    StatsModelsWrapper_with_OLS, p_by_year, EWMTransformer
)



from BW_Multi_Target_Simulator_v2 import (
    load_and_prepare_multi_target_data,
    L_func_multi_target_equal_weight,
    L_func_multi_target_confidence_weighted,
    L_func_multi_target_long_short
)



# Read BWM data
ds  = xr.load_dataset('./data/rm_demo_ds_20250627.nc')

fm_opt = PortfolioOptimizer.FactorModelOptimizer(ds, optimizer_config=fm_optimizer_kwargs)

classic_opt = PortfolioOptimizer.ClassicOptimizer(ds, optimizer_config=classic_optimizer_kwargs)

# Simulation test args
test_asset_universe = ['NFLX', 'NVDA', 'JNJ', 'AMZN', 'JPM', 'XOM', 'PG', 'GOOG']
rebalance_freq = 'weekly'
window_size = 30
window_type='expanding'


# Optimizer configs
# classic
classic_optimizer_kwargs = {
    "model": "Classic",               # Classic Markowitz optimizer
    "rm": "MV",                       # Risk Measure: MV = mean variance
    "rf": 0,                          # Risk-free rate (in decimal form)
    "l": 0,                           # Risk aversion (used in some models)
    "method_mu": "ewma",              # Method to estimate expected returns
    "method_cov": "ewma",             # Method to estimate covariance matrix
    "halflife": 30,                   # EWMA halflife in trading days
    "hist": False,                    # sigma & mu are computed from factor data
    "obj": "Sharpe",                  # Optimization objective
    "sht": True,                      # Allow shorts
    "budget": 1.0,                    # Absolute sum of weights
    "uppersht": 1.0,                  # Maximum sum of absolute values of short
    "upperlng": 1.0,                  # Maximum of the sum of long
    "returns_var": "residual_return", # Returns variable name: 'return' or 'residual_return'
    "mu_scalar": None                 # Scalar applied to expected returns [0,1]

}

classic_optimizer_kwargs = {
    "model": "Classic",               # Classic Markowitz optimizer
    "rm": "MV",                       # Risk Measure: MV = mean variance
    "rf": 0,                          # Risk-free rate (in decimal form)
    "l": 0,                           # Risk aversion (used in some models)
    "obj": "Sharpe",                  # Optimization objective
    "sht": True,                      # Allow shorts
    "budget": 1.0,                    # Absolute sum of weights
    "uppersht": 1.0,                  # Maximum sum of absolute values of short
    "upperlng": 1.0,                  # Maximum of the sum of long
    "returns_var": "residual_return", # Returns variable name: 'return' or 'residual_return'
}




# factor
fm_optimizer_kwargs = {
    "model": "Classic",          # Classic Markowitz optimizer
    "rm": "MV",                  # Risk Measure: MV = mean variance
    "rf": 0,                     # Risk-free rate (in decimal form)
    "l": 0,                      # Risk aversion (used in some models)
    "method_f": "ewma",          # Method to estimate expected returns
    "method_F": "ewma",          # Method to estimate covariance matrix
    "halflife": 30,              # EWMA halflife in trading days
    "hist": False,               # sigma & mu are computed from factor data
    "obj": "Sharpe",             # Optimization objective
    "sht": True,                 # Allow shorts
    "budget": 1.0,               # Absolute sum of weights
    "uppersht": 1.0,             # Maximum sum of absolute values of short
    "upperlng": 1.0,             # Maximum of the sum of long
    "returns_var": "return",     # Returns variable name: 'return' or 'residual_return'
    "f_scalar": None             # Scalar applied to expected factor returns [0,1]
}



def L_func_optimized_weight(returns_window_ds:xr.Dataset, ):
    """
    Riskfolio-optimized weight position across universe assets.

    args:
        returns_window_ds: Dataset

    kwargs:
        PortfolioOptimizer configuration

    Returns:
        Series with portfolio leverage for each date
    """
    base_leverage = params[0] if params else 1.0
    n_targets = len(predictions_df.columns)

    # Simple equal weight: average prediction across targets
    avg_prediction = predictions_df.mean(axis=1)

    # Binary position: long if avg > 0, short if avg < 0
    leverage = np.where(avg_prediction > 0, base_leverage, -base_leverage)

    return pd.Series(leverage, index=predictions_df.index)


def Simulate_Optimizer_Backtest(ds, optimizer_class, optimizer_kwargs=None,
                                asset_universe=None, window_size=60,
                                window_type='expanding', rebalance_freq='monthly',
                                tag=None, use_cache=True):
    """
    Backtest loop for RiskfolioOptimizer using historical returns from an BW Xarray dataset.

    Args:
        ds: xarray.Dataset with dimensions ('date', 'ticker')
        optimizer_class: class or callable that returns a Riskfolio-compatible optimizer
        optimizer_kwargs: dict of kwargs to pass to optimizer_class
        asset_universe: list of tickers to include (optional)
        window_size: lookback window for optimizer
        window_type: 'expanding' or 'rolling'
        rebalance_freq: 'daily', 'weekly', 'monthly'
        tag: string label for the run
        use_cache: whether to enable caching

    Returns:
        pd.DataFrame with portfolio returns and weights
    """

    
    print(f"Starting backtest simulation: {tag}")
    print(f"Universe: {asset_universe}")
    print(f"Rebalance frequency: {rebalance_freq}, Window: {window_type} ({window_size})")

    # Subset asset universe
    if asset_universe is not None:
        ds = ds.sel(ticker=asset_universe)

    # Filter valid rebalancing dates from actual trading dates
    trading_dates = ds.date.to_index()
    rebalance_dates = None

    if rebalance_freq == 'daily':
        rebalance_dates = trading_dates[window_size:]

    elif rebalance_freq == 'weekly':
        # Convert to DataFrame to allow grouping by week
        df_dates = pd.DataFrame(index=trading_dates)
        df_dates['week'] = df_dates.index.to_period('W')  # ISO weeks
        rebalance_dates = df_dates.groupby('week').tail(1).index  # Last trading day per week
        rebalance_dates = rebalance_dates[rebalance_dates >= trading_dates[window_size]]

    elif rebalance_freq == 'monthly':
        df_dates = pd.DataFrame(index=trading_dates)
        df_dates['month'] = df_dates.index.to_period('M')
        rebalance_dates = df_dates.groupby('month').tail(1).index  # Last trading day per month
        rebalance_dates = rebalance_dates[rebalance_dates >= trading_dates[window_size]]

    else:
        raise ValueError(f"Unsupported rebalance frequency: {rebalance_freq}")


    portfolio_returns = []
    portfolio_weights = []
    weight_dates = []

    for d in rebalance_dates:
        if window_type == 'expanding':
            ds_window = ds.sel(date=slice(None, d))
        else:
            ds_window = ds.sel(date=slice(d - pd.Timedelta(days=window_size*2), d)).isel(date=-window_size)

        # Initialize optimizer and run optimization
        opt = optimizer_class(ds_window, optimizer_config=optimizer_kwargs)
        opt.optimize(returns_col=returns_var)

        w = opt.w
        if w is None:
            print(f"[{d.date()}] No weights returned, skipping...")
            continue

        # Store weights
        w_series = w.squeeze() if hasattr(w, 'squeeze') else w
        w_series.name = d
        portfolio_weights.append(w_series)
        weight_dates.append(d)

        # Compute next-period return
        next_date_idx = all_dates.get_loc(d) + 1
        if next_date_idx >= len(all_dates):
            break

        next_date = all_dates[next_date_idx]
        returns = ds[returns_var].sel(date=next_date).to_series()

        aligned_returns = returns.reindex(w_series.index).fillna(0)
        port_ret = (w_series * aligned_returns).sum()
        portfolio_returns.append((next_date, port_ret))

    # Format results
    ret_df = pd.DataFrame(portfolio_returns, columns=['date', 'portfolio_ret']).set_index('date')
    w_df = pd.DataFrame(portfolio_weights, index=weight_dates)
    results = ret_df.join(w_df, how='left')

    print(f"Backtest complete: {len(results)} periods evaluated.")
    return results



"""Old Test for BW-MultiTarget"""

# Creating inputs
feature_etfs = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU']
target_etfs = ['SPY']
all_etfs = feature_etfs + target_etfs


# Extract market factor returns
market_df = (
    ds[['market_factor_return']]
    .to_dataframe()
    .reset_index()
    .drop_duplicates(subset=['date'])  # Only keep one per date
    .assign(ticker='SPY')              # Label the market return with 'SPY'
    .set_index(['date', 'ticker'])
    .rename(columns={'market_factor_return': 'return'})
)

# Step 3: Drop duplicates: one sector_factor_return per sector per date
sector_df = (
    ds[['sector_factor_return', 'sector_etf']]
    .to_dataframe()
    .reset_index()
    .dropna(subset=['sector_etf'])  # Make sure ETF labels are present
    .groupby(['date', 'sector_etf'])  # One row per sector ETF per date
    .first()  # or .mean() if aggregation makes sense
    .reset_index()
    .drop(columns='ticker')
    .rename(columns={'sector_etf': 'ticker', 'sector_factor_return': 'return'})
    .set_index(['date', 'ticker'])
    .sort_index()
)


# Concatenate market and sector
factor_df = pd.concat([market_df, sector_df]).sort_index()

# Wide format
factor_df = factor_df.unstack('ticker')

# Convert to xarray
factor_ds = factor_df.to_xarray()
