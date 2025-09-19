import numpy as np
import pandas as pd
import xarray as xr

from utils.PortfolioOptimizer import ClassicOptimizer, FactorModelOptimizer, BlackLittermanOptimizer

def load_data(file_path="data/rm_demo_ds_20250627.nc"):
    """Loads and prepares the data."""
    try:
        ds = xr.load_dataset(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        print("Please ensure you have the demo dataset in the 'data/' directory.")
        return None
    return ds

def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*80)
    print(f"### {title} ###")
    print("="*80)

def print_weights(weights):
    """Prints formatted portfolio weights."""
    if weights is not None:
        print("\n--- Optimized Weights ---")
        print(weights.to_string(float_format="{:.4f}".format))
    else:
        print("\n--- Weights not calculated ---")

def print_stats(optimizer):
    """Prints key portfolio statistics."""
    if hasattr(optimizer, 'sharpe_'):
        print("\n--- In-Sample Portfolio Stats ---")
        stats = optimizer.get_portfolio_stats()
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
        print(f"Annualized Return: {stats['portfolio_return']:.4f}")
        print(f"Annualized Volatility: {stats['portfolio_volatility']:.4f}")
        print(f"Net Exposure: {stats['net_exposure']:.2%}")
        print(f"Gross Exposure: {stats['gross_exposure'].item():.2%}")

def demonstrate_classic_optimizer(data):
    """Demonstrates the features of the ClassicOptimizer."""
    print_header("ClassicOptimizer Demonstration")

    # 1. Basic Mean-Variance Optimization (Historical)
    print("\n1. Basic Sharpe Ratio Maximization (Historical mu and cov)")
    classic_hist = ClassicOptimizer(obj='Sharpe', sht=True)
    classic_hist.fit(data)
    print_weights(classic_hist.get_weights())
    print_stats(classic_hist)

    # 2. EWMA for mu and cov
    print("\n2. Sharpe Ratio Maximization (EWMA mu and cov)")
    classic_ewma = ClassicOptimizer(
        method_mu='ewma1',
        method_cov='ewma1',
        ewma_mu_halflife=30,
        ewma_cov_halflife=30,
        obj='Sharpe',
        sht=True
    )
    classic_ewma.fit(data)
    print_weights(classic_ewma.get_weights())
    print_stats(classic_ewma)

    # 3. Custom mu input
    print("\n3. Using a custom mu vector (e.g., from a predictive model)")
    custom_mu = pd.Series(np.random.randn(len(data.ticker)) * 0.01, index=data.ticker.values)
    print("\n--- Custom Mu Vector ---")
    print(custom_mu)
    classic_custom_mu = ClassicOptimizer(
        method_mu='custom',
        user_input_mu=custom_mu,
        method_cov='hist',
        obj='Sharpe',
        sht=True
    )
    classic_custom_mu.fit(data)
    print_weights(classic_custom_mu.get_weights())
    print_stats(classic_custom_mu)

def demonstrate_factor_model_optimizer(data):
    """Demonstrates the features of the FactorModelOptimizer."""
    print_header("FactorModelOptimizer Demonstration")

    # 1. Standard Factor Model Optimization
    print("\n1. Sharpe Ratio Maximization using calculated factor returns (f) and covariance (F)")
    factor_model = FactorModelOptimizer(obj='Sharpe', sht=True)
    factor_model.fit(data)
    print_weights(factor_model.get_weights())
    print_stats(factor_model)
    print("\n--- Calculated Factor Returns (f) ---")
    print(factor_model.f_)

    # 2. Custom Factor Inputs (f and F)
    print("\n2. Using custom factor returns (f) and factor covariance (F)")
    f_in = (factor_model.f_['f'].copy()) * 1.5 # Example: more bullish factor view
    F_in = factor_model.F_.copy() * 1.1 # Example: slightly higher factor volatility
    print("\n--- Custom Factor Returns (f_in) ---")
    print(f_in)
    
    factor_custom_f = FactorModelOptimizer(
        use_custom_factor_inputs=True,
        user_input_f=f_in,
        user_input_F=F_in,
        obj='Sharpe',
        sht=True
    )
    factor_custom_f.fit(data)
    print_weights(factor_custom_f.get_weights())
    print_stats(factor_custom_f)

def demonstrate_black_litterman_optimizer(data):
    """Demonstrates the features of the BlackLittermanOptimizer."""
    print_header("BlackLittermanOptimizer Demonstration")

    # 1. Equilibrium Portfolio (No Views)
    print("\n1. Black-Litterman with no views (produces equilibrium portfolio)")
    bl_no_views = BlackLittermanOptimizer(sht=True)
    bl_no_views.fit(data)
    print("\n--- Implied Equilibrium Returns (Pi) ---")
    print(bl_no_views.get_equilibrium_returns())
    print_weights(bl_no_views.get_weights())
    print_stats(bl_no_views)

    # 2. Black-Litterman with Custom Views
    print("\n2. Black-Litterman with custom investor views")
    tickers = data.ticker.values
    
    # Create P matrix robustly by name
    P = pd.DataFrame(0, index=['View1', 'View2'], columns=tickers)
    # View 1: GOOG will outperform AMZN by 2%
    P.loc['View1', 'GOOG'] = 1
    P.loc['View1', 'AMZN'] = -1
    # View 2: NVDA will have an absolute return of 15%
    P.loc['View2', 'NVDA'] = 1

    Q = pd.Series([0.02, 0.15], index=['View1', 'View2'])
    
    print("\n--- Investor Views (P Matrix) ---")
    print(P)
    print("\n--- View Expected Returns (Q Vector) ---")
    print(Q)

    bl_with_views = BlackLittermanOptimizer(
        P=P,
        Q=Q,
        sht=True
    )
    bl_with_views.fit(data)
    
    print("\n--- BL-Adjusted Expected Returns (mu_bl) ---")
    print(bl_with_views.mu_bl_)
    print_weights(bl_with_views.get_weights())
    print_stats(bl_with_views)

def main():
    """Main function to run all demonstrations."""
    data = load_data()
    if data is None:
        return

    demonstrate_classic_optimizer(data)
    demonstrate_factor_model_optimizer(data)
    demonstrate_black_litterman_optimizer(data)

if __name__ == "__main__":
    main()