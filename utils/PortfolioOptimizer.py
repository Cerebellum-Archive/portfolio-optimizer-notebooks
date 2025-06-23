import pandas as pd
import numpy as np
import riskfolio as rp
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class RiskfolioOptimizer:
    def __init__(self, daily_ds:xr.Dataset, optimizer_config:dict=None):
        """
        Initialize the RiskfolioOptimizer with a pre-sliced xarray dataset.

        Parameters
        ----------
        daily_ds : xr.Dataset
            A sliced xarray dataset with dimensions ['date', 'ticker'] and containing
            variables named in Blue Water conventions (return, variance, etc.).
            Expected to be in percentage terms.

        optimizer_config : dict, optional
            Configuration dictionary to specify optimization model parameters such as
            risk measure, objective function, constraints, etc.

        Notes
        -----
        This initializer will:
        - Infer start/end dates and tickers from the dataset's coordinates.
        - Convert the dataset to a tidy DataFrame (self.bw_daily_df).
        - Convert return/variance columns from percentage to decimal format.
        """

        # Store metadata
        self.start_date = daily_ds.coords["date"].values[0]
        self.end_date = daily_ds.coords["date"].values[-1]
        self.tickers = list(daily_ds.coords["ticker"].values)
        self.optimizer_config = optimizer_config

        # Convert xarray dataset to DataFrame
        df = daily_ds.to_dataframe().reset_index()
        df.set_index(["date", "ticker"], inplace=True)

        # Convert returns/variances from percentage to decimal
        for col in df.columns:
            if "return" in col.lower() or "variance" in col.lower():
                df[col] = df[col] / 100.0

        self.bw_daily_df = df

        # Placeholders for optimization components
        self.mu = None      # Expected returns
        self.sigma = None   # Covariance matrix
        self.w = None       # Optimized weights
        self.port = None    # Riskfolio portfolio object


    def optimize(self):
        raise NotImplementedError("Subclasses must implement the optimize() method.")

    def plot_weights(self):
        if self.w is not None:
            has_negative = (self.w < 0).any().item()

        weights = self.w.iloc[:,0] # TODO: convert type try to clean up later
        if has_negative:
            weights.sort_values(ascending=True).plot(kind='barh', title='Optimized Portfolio Weights')
            plt.xlabel("Weight")
            plt.ylabel("Ticker")
        else:
            weights.sort_values(ascending=False).plot(kind='barh', title='Optimized Portfolio Weights')
            plt.ylabel("Weight")
            plt.xlabel("Ticker")
            plt.axvline(0, color='black', linewidth=0.8) if has_negative else None
            plt.tight_layout()
            plt.show()


    def plot_weights_plotly(self):
        if self.w is not None:
            weights = self.w.sort_values(by='weights')
            tickers = weights.index
            values = weights['weights'].values.flatten()

            colors = np.where(values >= 0, '#1f77b4', '#ff7f0e')
            text_positions = ['middle right' if v >= 0 else 'middle left' for v in values]
            text_labels = [f"{v:.1%}  {ticker}" if v >= 0 else f"{v:.1%}  {ticker}"
                           for v, ticker in zip(values, tickers)]

            fig = go.Figure()

            # Lollipop sticks
            for val, i in zip(values, range(len(tickers))):
                fig.add_trace(go.Scatter(
                    x=[0, val], y=[i, i],
                    mode='lines',
                    line=dict(color='lightgray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Lollipop heads
            fig.add_trace(go.Scatter(
                x=values,
                y=list(range(len(tickers))),
                mode='markers+text',
                marker=dict(size=10, color=colors),
                text=text_labels,
                textposition=text_positions,
                textfont=dict(size=10, color=colors),
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ))

            # Layout settings
            fig.update_layout(
                title=dict(
                    text='Optimized Portfolio Weights',
                    x=0.5, xanchor='center',
                    font=dict(size=16, family='Arial', color='black')
                ),
                xaxis_title='Weight',
                xaxis_tickformat='.0%',
                plot_bgcolor='white',
                height=max(400, 20 * len(weights)),
                margin=dict(l=120, r=120, t=60, b=40)
            )

            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='black')
            fig.show(config={'responsive': True})


    def plot_frontier(self, points=30, current_weights=None):
        if self.port is not None:
            # Generate efficient frontier
            frontier = self.port.efficient_frontier(points=points)

            # Plot optimized portfolio
            ax = rp.plot_frontier(w_frontier=frontier,
                                  mu=self.mu,
                                  cov=self.port.cov,
                                  returns=self.port.returns,
                                  w=self.w,
                                  label='Optimized',
                                  marker='*',
                                  s=16,
                                  c='r')

            # Plot current portfolio if provided
            if current_weights is not None:
                current_weights = current_weights.reindex(self.w.index).fillna(0)
                port_return = np.dot(current_weights, self.mu.values.flatten())
                port_risk = np.sqrt(np.dot(current_weights.T, np.dot(self.port.cov.values, current_weights)))
                ax.scatter(port_risk, port_return, marker='o', s=16, color='red', label='Current Position')

            fig = ax.get_figure()
            fig.set_constrained_layout(True)

            plt.show()


class ClassicOptimizer(RiskfolioOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate_config(self):
        # Check that returns are provided
        if self.bw_daily_df is None or self.bw_daily_df.empty:
            raise ValueError("Return data (bw_daily_df) is empty or not provided.")

        # Check for NaNs
        if self.bw_daily_df.isnull().any().any():
            raise ValueError("bw_daily_df contains NaN values.")

        if self.optimizer_config is not None:
            # Check model
            if self.optimizer_config.get('model') != 'Classic':
                msg = ("ClassicOptimizer requires optimizer_config['model'] == 'Classic'.")
                raise ValueError(msg)

            # Check config keys
            valid_objs = {"Sharpe", "MinRisk", "MaxRet", "Utility"}
            obj = self.optimizer_config.get("obj", "Sharpe")
            if obj not in valid_objs:
                raise ValueError(f"Invalid objective '{obj}'. Must be one of {valid_objs}.")

        else:
            # Set default config
            self.optimizer_config = {}
            self.optimizer_config.setdefault("model", "Classic")
            self.optimizer_config.setdefault("rm", "MV")     # Risk measure: MV = variance
            self.optimizer_config.setdefault("rf", 0)        # Risk-free rate
            self.optimizer_config.setdefault("l", 0)         # Risk aversion
            self.optimizer_config.setdefault("hist", True)   # Use historical data


    def optimize(self, returns_col:str='residual_return', LongOnly=True) -> None:
        """Creates a portfolio object from optimizer_config
        and calculates optimized weights from historical returns."""

        self.validate_config()
        returns = self.bw_daily_df[returns_col].unstack(level='ticker')

        # Create portfolio object
        self.port = rp.Portfolio(returns=returns)
        self.port.assets_stats(method_mu='hist', method_cov='hist')

        # Check constraints
        if LongOnly==False:
            self.port.sht = True # enable shorts
            self.port.budget = 1.0 # absolute sum of weights
            self.port.uppersht = 1.0 # maximum of the sum of absolute values of short
            self.port.upperlng = 1.0 # maximum of the sum of long

        self.sigma = self.port.cov
        self.mu = self.port.mu
        self.w = self.port.optimization(**self.optimizer_config)


class FactorModelOptimizer(RiskfolioOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = None  # factor exposure matrix
        self.F = None  # factor covariance matrix
        self.D = None  # residual matrix
        self.f = None  # factor returns vector

        self.returns_window = None  # trailing window gross returns used for plotting
        self.factor_returns_window = None  # trailing window factor returns


    def validate_config(self):
        # Check that factor returns are provided
        if self.bw_daily_df is None or self.bw_daily_df.empty:
            raise ValueError("Return data (bw_daily_df) is empty or not provided.")

        # Check for NaNs
        if self.bw_daily_df.isnull().any().any():
            raise ValueError("bw_daily_df contains NaN values.")

        if self.optimizer_config is not None:
            # Check model and hist
            if self.optimizer_config.get('model') != 'Classic' or self.optimizer_config.get('hist') is True:
                msg = ("FactorModelOptimizer requires optimizer_config['model'] == 'Classic' "
                       "and optimizer_config['hist'] == False\n"
                       "FactorModelOptimizer precomputes factor covariance matrix (sigma).\n"
                       "FactorModelOptimizer model='Classic' to use precomputed sigma.")
                raise ValueError(msg)

            # Check config keys
            valid_objs = {"Sharpe", "MinRisk", "MaxRet", "Utility"}
            obj = self.optimizer_config.get("obj", "Sharpe")
            if obj not in valid_objs:
                raise ValueError(f"Invalid objective '{obj}'. Must be one of {valid_objs}.")

        else:
            # Set default config
            self.optimizer_config = {}
            self.optimizer_config.setdefault("model", "Classic")
            self.optimizer_config.setdefault("rm", "MV")     # Risk measure: MV = variance
            self.optimizer_config.setdefault("rf", 0)        # Risk-free rate
            self.optimizer_config.setdefault("l", 0)         # Risk aversion
            self.optimizer_config.setdefault("hist", False)  # Estimate from market returns


    def calculate_fm_cov(self, n_window:int=None, end_window:pd.Timestamp=None, return_sigma=False):
        """
        Compute the factor model covariance matrix Σ = B F Bᵀ + D,
        where:
            - B: factor exposure matrix (market + sector)
            - F: factor covariance matrix
            - D: diagonal matrix of residual variances
        """

        # --- 1. Resolve end date ---
        all_dates = self.bw_daily_df.index.get_level_values('date').unique()

        if end_window is None:
            end_date = all_dates[-1]
        elif end_window in all_dates:
            end_date = end_window
        else:
            msg = (f"End window {end_window} not found in dataset. "
                   f"Available range: {all_dates.min()} to {all_dates.max()}"
                   )
            raise ValueError(msg)

        # --- 2. Determine time window for factor return estimation ---
        if n_window is None:
            trailing_dates = all_dates[all_dates <= end_date]
        else:
            idx = all_dates.get_loc(end_date)
            if idx - n_window + 1 < 0:
                msg = (f"Requested window of {n_window} days before "
                       f"{end_date} exceeds data bounds."
                       )
                raise ValueError(msg)
            trailing_dates = all_dates[idx - n_window + 1 : idx + 1]

        # --- 3. Slice factor-level DataFrame over trailing window ---
        mask = self.bw_daily_df.index.get_level_values('date').isin(trailing_dates)
        trailing_df = self.bw_daily_df[mask]

        self.returns_window = trailing_df['return'].unstack(level='ticker')

        end_df = trailing_df.xs(end_date, level='date')

        # --- 4. Construct B matrix: Market and Sector exposures ---
        market_beta = end_df[['market_beta']]  # shape: [n_assets x 1]
        sector_dummies = pd.get_dummies(end_df['bw_sector_name'])  # [n_assets x k_sectors]
        sector_beta = sector_dummies.mul(end_df['sector_beta'], axis=0)  # [n_assets x k]
        self.B = pd.concat([market_beta, sector_beta], axis=1)  # [n_assets x (1 + k)]

        # --- 5. Residual variance matrix D (diagonal) ---
        self.D = pd.DataFrame(np.diag(end_df['residual_variance']),
                              index=self.B.index,
                              columns=self.B.index)

        # --- 6. Factor returns matrix f: mean returns over trailing window ---
        mkt_factor_ret = trailing_df.groupby('date')['market_factor_return'].first().to_frame(name='Market')
        sctr_factor_ret = trailing_df.groupby(['bw_sector_name', 'date'])['sector_factor_return'].first().unstack().T
        factor_returns = pd.concat([mkt_factor_ret, sctr_factor_ret], axis=1)
        self.factor_returns_window = factor_returns

        self.f = factor_returns.mean(axis=0).to_frame(name='f')  # shape: (1 + k, 1)

        # --- 7. Factor covariance matrix F ---
        self.F = factor_returns.cov()

        # --- 8. Expected returns μ = B @ f ---
        mu = self.B.values @ self.f.values
        mu = mu.flatten()
        self.mu = pd.DataFrame([mu], columns=self.B.index, dtype=np.float64)

        # --- 9. Covariance matrix Σ = B F Bᵗ + D ---
        sigma_vals = self.B.values @ self.F.values @ self.B.values.T + self.D.values
        self.sigma = pd.DataFrame(sigma_vals,
                                  index=self.B.index,
                                  columns=self.B.index,
                                  dtype=np.float64
                                  )

        if return_sigma:
            return self.sigma
        else:
            return None


    def optimize(self, LongOnly=True) -> None:
        """
        Optimizes portfolio using precomputed mu and sigma from factor model.
        """

        # Check inputs
        self.validate_config()
        if self.sigma is None:
            msg = ("Sigma and mu must be computed prior to optimization. "
                   "Run calculate_fm_cov()."
                   )
            print(msg)
            return

        # Step 1: Create Riskfolio-Lib Portfolio object
        self.port = rp.Portfolio(returns=self.returns_window)

        # Step 2: Set constraints if needed TODO: Expand constraints capability
        if LongOnly==False:
            self.port.sht = True # enable shorts
            self.port.budget = 1.0 # absolute sum of weights
            self.port.uppersht = 1.0 # maximum of the sum of absolute values of short
            self.port.upperlng = 1.0 # maximum of the sum of long

        # Step 3: Portfolio assignment
        self.port.mu = self.mu
        self.port.cov = self.sigma

        # Step 5: Run optimization
        self.w = self.port.optimization(**self.optimizer_config)


class BlackLittermanOptimizer(RiskfolioOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.views = None  # investor views to blend with factor model
        self.asset_classes = None
        self.P = None
        self.Q = None

    def validate_config(self):
        pass

    def optimize(self) -> None:
        pass
