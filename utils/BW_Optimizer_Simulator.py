import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_optimizer_performance(results_dict, labels=None, rolling_window=30, save_plt=False):
    """
    Plots performance comparison for different optimizers.

    Args:
        results_dict: dict of {label: results_list} from different optimizers
        labels: optional list of labels (same order as keys in results_dict)
        rolling_window: int, window for rolling Sharpe
    """

    if labels is None:
        labels = list(results_dict.keys())

    # Convert to DataFrames
    all_dfs = {k: pd.DataFrame(v).set_index("prediction_date") for k, v in results_dict.items()}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

    # --- Subplot 1: Cumulative Returns ---
    ax1.set_title('Cumulative Portfolio Returns')
    for label in labels:
        df = all_dfs[label]
        cum_ret = (1 + df['portfolio_ret']).cumprod()
        cum_ret.plot(ax=ax1, label=label)

        # Grab the last color used
        last_color = ax1.get_lines()[-1].get_color()

        cum_bm_ret = (1 + df['benchmark_ret']).cumprod()
        bm_label = label + '_benchmark'
        cum_bm_ret.plot(ax=ax1,
                        label=bm_label,
                        color=last_color,
                        linestyle='--',
                        alpha=0.7)

    ax1.set_ylabel("Cumulative Return")
    ax1.set_xlabel('')
    ax1.legend()
    ax1.grid(True)

    # --- Subplot 2: Rolling Sharpe Ratio ---
    ax2.set_title(f'Rolling Sharpe Ratio ({rolling_window}-day)')
    for label in labels:
        df = all_dfs[label]
        rolling_ret = df['portfolio_ret'].rolling(rolling_window)
        rolling_sharpe = rolling_ret.mean() / rolling_ret.std()
        rolling_sharpe.plot(ax=ax2, label=label)

        # Grab the last color used
        last_color = ax2.get_lines()[-1].get_color()

        rolling_bm_ret = df['benchmark_ret'].rolling(rolling_window)
        rolling_bm_sharpe = rolling_bm_ret.mean() / rolling_bm_ret.std()
        bm_label = label + '_benchmark'
        rolling_bm_sharpe.plot(ax=ax2,
                               label=bm_label,
                               color=last_color,
                               linestyle='--',
                               alpha=0.7)

    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_ylabel("Rolling Sharpe")
    ax2.set_xlabel('')
    ax2.legend()
    ax2.grid(True)

    # --- Subplot 3: Drawdown ---
    ax3.set_title('Portfolio Drawdown')
    for label in labels:
        df = all_dfs[label]
        cum_ret = (1 + df['portfolio_ret']).cumprod()
        running_max = cum_ret.cummax()
        drawdown = cum_ret / running_max - 1
        drawdown.plot(ax=ax3, label=label)

        # Grab the last color used
        last_color = ax3.get_lines()[-1].get_color()

        cum_bm_ret = (1 + df['benchmark_ret']).cumprod()
        running_bm_max = cum_bm_ret.cummax()
        drawdown_bm = cum_bm_ret / running_bm_max - 1
        bm_label = label + '_benchmark'
        drawdown_bm.plot(ax=ax3, label=bm_label, color=last_color, linestyle='--')
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_ylabel("Drawdown")
    ax3.set_xlabel('')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    if save_plt:
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/optimizer_performance_comparison.png', dpi=300)
        print("Saved: reports/optimizer_performance_comparison.png")
    plt.show()


def create_optimizer_summary_table(results_dict, save_csv=False):
    """
    Creates a performance summary DataFrame for each optimizer strategy.

    Parameters
    ----------
    results_dict : dict
        Keys are strategy names (e.g., 'Classic', 'FactorModel').
        Values are list of dicts with keys: 'portfolio_ret', 'benchmark_ret'.

    Returns
    -------
    summary_df : pd.DataFrame
        Performance summary table.
    """
    # Convert to DataFrames
    all_dfs = {k: pd.DataFrame(v).set_index("prediction_date") for k, v in results_dict.items()}

    summary_data = []

    for tag, df in all_dfs.items():
        df = df.copy()

        port_ret = df['portfolio_ret']
        bench_ret = df['benchmark_ret']

        # Annualized metrics
        port_ann_ret = port_ret.mean() * 252
        port_ann_vol = port_ret.std() * np.sqrt(252)
        port_sharpe = port_ann_ret / port_ann_vol if port_ann_vol != 0 else np.nan

        bench_ann_ret = bench_ret.mean() * 252
        bench_ann_vol = bench_ret.std() * np.sqrt(252)
        bench_sharpe = bench_ann_ret / bench_ann_vol if bench_ann_vol != 0 else np.nan

        # Max drawdown
        port_dd = (port_ret.cumsum() - port_ret.cumsum().expanding().max()).min()
        bench_dd = (bench_ret.cumsum() - bench_ret.cumsum().expanding().max()).min()

        summary_data.append({
            'Strategy': tag,
            'Portfolio_Annual_Return': port_ann_ret,
            'Portfolio_Annual_Vol': port_ann_vol,
            'Portfolio_Sharpe': port_sharpe,
            'Portfolio_Max_DD': port_dd,
            'Benchmark_Annual_Return': bench_ann_ret,
            'Benchmark_Annual_Vol': bench_ann_vol,
            'Benchmark_Sharpe': bench_sharpe,
            'Benchmark_Max_DD': bench_dd,
            'Excess_Return': port_ann_ret - bench_ann_ret,
            'Sharpe_Improvement': port_sharpe - bench_sharpe
        })

    summary_df = pd.DataFrame(summary_data)
    if save_csv:
        os.makedirs('reports', exist_ok=True)
        summary_df.to_csv('reports/strategy_summary_table.csv', index=False)
        print("Saved strategy performance summary: reports/strategy_summary_table.csv")
    return summary_df
