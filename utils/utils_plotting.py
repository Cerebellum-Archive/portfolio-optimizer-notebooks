# utils/utils_plotting.py
"""
Utility functions for plotting and analyzing portfolio backtest results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_optimizer_performance(results_dict, labels=None, rolling_window=30, save_plt=False):
    """
    Plots performance comparison for different optimizers.

    Args:
        results_dict (dict): Dictionary of {label: results_list} from different optimizers.
        labels (list, optional): List of labels for the plot. Defaults to keys in results_dict.
        rolling_window (int): Window for rolling Sharpe ratio.
        save_plt (bool): If True, saves the plot to the 'reports/' directory.
    """
    os.makedirs('reports', exist_ok=True)

    if labels is None:
        labels = list(results_dict.keys())

    all_dfs = {k: pd.DataFrame(v).set_index("prediction_date") for k, v in results_dict.items()}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # --- 1. Cumulative Returns ---
    ax1.set_title('Cumulative Portfolio Returns')
    for label in labels:
        df = all_dfs[label]
        cum_ret = (1 + df['portfolio_ret']).cumprod()
        cum_ret.plot(ax=ax1, label=label, lw=2)

        if 'benchmark_ret' in df.columns:
            cum_bm_ret = (1 + df['benchmark_ret']).cumprod()
            bm_label = f"{label} Benchmark (Equal-Weight)"
            cum_bm_ret.plot(ax=ax1, label=bm_label, color='gray', linestyle='--', alpha=0.8)

    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # --- 2. Rolling Sharpe Ratio ---
    ax2.set_title(f'Rolling Sharpe Ratio ({rolling_window}-day)')
    for label in labels:
        df = all_dfs[label]
        rolling_ret = df['portfolio_ret'].rolling(rolling_window)
        rolling_sharpe = (rolling_ret.mean() / rolling_ret.std()) * np.sqrt(252)
        rolling_sharpe.plot(ax=ax2, label=label, lw=2)

    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel("Annualized Rolling Sharpe")
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    # --- 3. Drawdown ---
    ax3.set_title('Portfolio Drawdown')
    for label in labels:
        df = all_dfs[label]
        cum_ret = (1 + df['portfolio_ret']).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret / running_max) - 1
        drawdown.plot(ax=ax3, label=label, lw=2)

    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_ylabel("Drawdown")
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.5)

    plt.tight_layout()
    if save_plt:
        plt.savefig('reports/optimizer_performance_comparison.png', dpi=300)
        print("Saved plot to: reports/optimizer_performance_comparison.png")
    plt.show()

def create_optimizer_summary_table(results_dict, save_csv=False):
    """
    Creates a performance summary DataFrame for each optimizer strategy.

    Args:
        results_dict (dict): Dictionary of {label: results_list}.
        save_csv (bool): If True, saves the table to the 'reports/' directory.

    Returns:
        pd.DataFrame: Performance summary table.
    """
    all_dfs = {k: pd.DataFrame(v).set_index("prediction_date") for k, v in results_dict.items()}
    summary_data = []

    for tag, df in all_dfs.items():
        port_ret = df['portfolio_ret']
        bench_ret = df.get('benchmark_ret')

        # Portfolio Metrics
        port_ann_ret = port_ret.mean() * 252
        port_ann_vol = port_ret.std() * np.sqrt(252)
        port_sharpe = port_ann_ret / port_ann_vol if port_ann_vol != 0 else 0
        port_cum_ret = (1 + port_ret).prod() - 1
        port_dd = ((1 + port_ret).cumprod().div((1 + port_ret).cumprod().cummax()) - 1).min()

        row = {
            'Strategy': tag,
            'Annualized Return': f"{port_ann_ret:.2%}",
            'Annualized Volatility': f"{port_ann_vol:.2%}",
            'Sharpe Ratio': f"{port_sharpe:.2f}",
            'Max Drawdown': f"{port_dd:.2%}",
            'Cumulative Return': f"{port_cum_ret:.2%}"
        }

        # Benchmark Metrics
        if bench_ret is not None:
            bench_ann_ret = bench_ret.mean() * 252
            bench_ann_vol = bench_ret.std() * np.sqrt(252)
            bench_sharpe = bench_ann_ret / bench_ann_vol if bench_ann_vol != 0 else 0
            row['Benchmark Sharpe'] = f"{bench_sharpe:.2f}"
            row['Excess Return'] = f"{(port_ann_ret - bench_ann_ret):.2%}"

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    
    if save_csv:
        os.makedirs('reports', exist_ok=True)
        summary_df.to_csv('reports/strategy_summary_table.csv', index=False)
        print("Saved summary table to: reports/strategy_summary_table.csv")
        
    return summary_df




def plot_static_frontier(optimizer, realized_return=None, realized_vol=None, title='Efficient Frontier'):
    """
    Plots the efficient frontier for a single fitted optimizer instance.

    Args:
        optimizer: A fitted instance of a PortfolioOptimizer class.
        realized_return (float, optional): The ex-post realized return for the period.
        realized_vol (float, optional): The ex-post realized volatility for the period.
        title (str): The title for the plot.
    """
    if not hasattr(optimizer, 'port_') or not hasattr(optimizer.port_, 'frontier'):
        print("Optimizer has not been fitted or frontier not calculated.")
        return

    frontier = optimizer.port_.frontier

    fig = go.Figure()

    # Plot the efficient frontier line
    fig.add_trace(go.Scatter(
        x=frontier['Volatility'],
        y=frontier['Returns'],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=2)
    ))

    # Plot the optimized portfolio (in-sample)
    fig.add_trace(go.Scatter(
        x=[optimizer.port_vol_],
        y=[optimizer.port_return_],
        mode='markers',
        name='Optimized Portfolio (In-Sample)',
        marker=dict(color='red', size=12, symbol='star')
    ))

    # Plot the realized portfolio point (ex-post)
    if realized_return is not None and realized_vol is not None:
        fig.add_trace(go.Scatter(
            x=[realized_vol],
            y=[realized_return],
            mode='markers',
            name='Realized Portfolio (Ex-Post)',
            marker=dict(color='green', size=12, symbol='circle')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return',
        legend_title='Portfolio',
        template='plotly_white'
    )
    fig.show()


def animate_frontier_over_time(results_list, title='Animated Efficient Frontier Over Time'):
    """
    Creates an animated plot of the efficient frontier over a backtest period.

    Args:
        results_list (list): The list of dictionaries from a backtest run.
        title (str): The title for the plot.
    """
    fig = go.Figure()

    # Add frames for each date in the backtest
    for i, result in enumerate(results_list):
        optimizer = result['optimizer_instance'] # Assuming you store the instance
        frontier = optimizer.port_.frontier

        fig.add_trace(go.Scatter(
            x=frontier['Volatility'],
            y=frontier['Returns'],
            mode='lines',
            name=f"Frontier {result['prediction_date'].strftime('%Y-%m-%d')}",
            visible=(i == 0) # Only the first trace is visible initially
        ))
        fig.add_trace(go.Scatter(
            x=[result['realized_vol']],
            y=[result['realized_ret']],
            mode='markers',
            name=f"Realized {result['prediction_date'].strftime('%Y-%m-%d')}",
            marker=dict(color='green', size=10),
            visible=(i == 0)
        ))

    # Create slider
    steps = []
    for i in range(0, len(fig.data), 2):
        step = dict(
            method="restyle",
            args=[{"visible": [False] * len(fig.data)}],
            label=fig.data[i].name.replace("Frontier ", "")
        )
        step["args"][0]["visible"][i] = True  # Toggle frontier
        step["args"][0]["visible"][i+1] = True # Toggle realized point
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Date: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=title,
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return',
        template='plotly_white'
    )
    fig.show()


def plot_expost_frontier(results_list, strategy_name='Strategy'):
    """
    Plots the ex-post realized risk/return points from a backtest.

    Args:
        results_list (list): The list of dictionaries from a backtest run.
        strategy_name (str): The name of the strategy for the plot legend.
    """
    df = pd.DataFrame(results_list)

    # Calculate rolling realized volatility and returns
    # This is a simplification; a more rigorous approach would use a consistent window
    realized_vol = df['portfolio_ret'].rolling(window=30).std() * np.sqrt(252)
    realized_ret = df['portfolio_ret'].rolling(window=30).mean() * 252

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=realized_vol,
        y=realized_ret,
        mode='markers',
        name=f'Realized Risk/Return ({strategy_name})',
        marker=dict(
            size=8,
            color=df['prediction_date'], # Color by date
            colorscale='Viridis',
            showscale=True,
            colorbar_title='Date'
        )
    ))

    fig.update_layout(
        title='Ex-Post Realized Frontier',
        xaxis_title='Realized Annualized Volatility (30-day rolling)',
        yaxis_title='Realized Annualized Return (30-day rolling)',
        template='plotly_white'
    )
    fig.show()


def plot_realized_vs_perfect_frontier(results_list, full_history_returns, strategy_name='Strategy'):
    """
    Plots the strategy's final realized risk/return point against the
    ex-post optimal ("perfect foresight") efficient frontier.

    Args:
        results_list (list): The list of dictionaries from a backtest run.
        full_history_returns (pd.DataFrame): DataFrame of returns for the entire backtest period.
        strategy_name (str): The name of the strategy for the plot legend.
    """
    # 1. Calculate the single realized risk/return point for the strategy
    df = pd.DataFrame(results_list)
    realized_return = df['portfolio_ret'].mean() * 252
    realized_vol = df['portfolio_ret'].std() * np.sqrt(252)

    # 2. Calculate the ex-post optimal ("perfect foresight") frontier
    # We use a ClassicOptimizer on the full history to find the theoretical best frontier
    perfect_optimizer = ClassicOptimizer(obj='Sharpe')
    perfect_optimizer.fit(full_history_returns)
    frontier = perfect_optimizer.port_.frontier

    # 3. Plot both
    fig = go.Figure()

    # Add the "perfect" frontier
    fig.add_trace(go.Scatter(
        x=frontier['Volatility'],
        y=frontier['Returns'],
        mode='lines',
        name='Ex-Post Optimal Frontier (Perfect Foresight)',
        line=dict(color='blue', width=2, dash='dash')
    ))

    # Add the strategy's realized performance point
    fig.add_trace(go.Scatter(
        x=[realized_vol],
        y=[realized_return],
        mode='markers',
        name=f'Realized Performance ({strategy_name})',
        marker=dict(
            color='red',
            size=16,
            symbol='star',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        hovertemplate=(
            f"<b>{strategy_name}</b><br>"
            f"Realized Return: {realized_return:.2%}<br>"
            f"Realized Volatility: {realized_vol:.2%}<extra></extra>"
        )
    ))

    fig.update_layout(
        title='Strategy Performance vs. Perfect Foresight Frontier',
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return',
        template='plotly_white',
        legend_title='Portfolio'
    )
    fig.show()
