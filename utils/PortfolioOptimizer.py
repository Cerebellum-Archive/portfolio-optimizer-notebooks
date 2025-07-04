import pandas as pd
import numpy as np
import warnings
import riskfolio as rp
import xarray as xr
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
        self.input_config = optimizer_config

        # Convert xarray dataset to DataFrame
        df = daily_ds.to_dataframe().reset_index()
        df.set_index(["date", "ticker"], inplace=True)
        df.sort_index(level=['date', 'ticker'], inplace=True)

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


    def configure_optimizer(self):
        raise NotImplementedError("Subclasses must implement the configure_optimizer() method.")

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
        self.configure_optimizer()

        self.returns = None
        self.scaled_mu = None

    def configure_optimizer(self):
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
            self.optimizer_config = {
                "model": "Classic",          # Classic Markowitz optimizer
                "rm": "MV",                  # Risk Measure: MV = Variance
                "rf": 0,                     # Risk-free rate (in decimal form)
                "l": 0,                      # Risk aversion (used in some models)
                "method_mu": "ewma",         # Method to estimate expected returns
                "method_cov": "ewma",        # Method to estimate covariance matrix
                "halflife": 30,              # EWMA halflife in trading days
                "obj": "Sharpe",             # Optimization objective
                "hist": False,               # sigma & mu are computed from asset_stats
                "sht": True,                 # Allow shorts
                "budget": 1.0,               # Absolute sum of weights
                "uppersht": 1.0,             # Maximum sum of absolute values of short
                "upperlng": 1.0,             # Maximum of the sum of long
                "returns_var": "return",     # Returns variable name: 'return' or 'residual_return'
                "mu_scalar": None            # Scalar applied to expected returns [0,1]
            }

        returns_var = self.optimizer_config.pop('returns_var')
        self.returns = self.bw_daily_df[returns_var].unstack(level='ticker')

        # Create portfolio
        self.port = rp.Portfolio(returns=self.returns)

        # Determine mean/covariance method for stats
        method_mu = self.optimizer_config.pop('method_mu', 'hist')
        method_cov = self.optimizer_config.pop('method_cov', 'hist')

        # Handle EWMA case with fallback halflife
        if method_mu == 'ewma' or method_cov == 'ewma':
            halflife = self.optimizer_config.pop('halflife', None)
            if halflife is None:
                halflife = 30  # Default fallback
                warnings.warn(
                    "Optimizer config did not specify a halflife for EWMA. "
                    "Defaulting to halflife=30."
                )
            mu = self.returns.ewm(halflife=halflife).mean().iloc[-1]
            cov = self.returns.ewm(halflife=halflife).cov().iloc[-self.returns.shape[1]:]

            self.port.mu = mu
            self.port.cov = cov.droplevel('date')

        else:
            # Apply simple 'hist'
            self.port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    def optimize(self):
        """Calculates optimized weights from historical returns."""

        # Assign sigma, mu used for optimization
        self.sigma = self.port.cov
        self.mu = self.port.mu

        # Scaled mu if not confident in estimate
        mu_scalar = self.optimizer_config.pop('mu_scalar', None)
        if mu_scalar is not None:
            self.scaled_mu = self.port.mu * mu_scalar
            self.port.mu = self.scaled_mu

        # Assign portfolio leverage and long/short budget
        self.port.sht = self.optimizer_config.pop('sht', False)
        self.port.budget = self.optimizer_config.pop('budget', 1.0)
        self.port.uppersht = self.optimizer_config.pop('uppersht', 1.0)
        self.port.upperlng = self.optimizer_config.pop('upperlng', 1.0)


        self.w = self.port.optimization(**self.optimizer_config)

        return self.w


class FactorModelOptimizer(RiskfolioOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_optimizer()

        self.B = None  # factor exposure matrix
        self.F = None  # factor covariance matrix
        self.D = None  # residual matrix
        self.f = None  # factor returns vector
        self.f_scaled = None  # scaled factor returns vector

        self.returns = None  # gross returns used for plotting
        self.factor_returns = None  # factor returns


    def configure_optimizer(self):
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
                       "FactorModelOptimizer model=='Classic' to use precomputed sigma.")
                raise ValueError(msg)

            # Check config keys
            valid_objs = {"Sharpe", "MinRisk", "MaxRet", "Utility"}
            obj = self.optimizer_config.get("obj", "Sharpe")
            if obj not in valid_objs:
                raise ValueError(f"Invalid objective '{obj}'. Must be one of {valid_objs}.")

        else:
            # Set default config
            self.optimizer_config = {
                "model": "Classic",          # Classic Markowitz optimizer
                "rm": "MV",                  # Risk Measure: MV = Variance
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

        returns_var = self.optimizer_config.pop('returns_var')
        self.returns = self.bw_daily_df[returns_var].unstack(level='ticker')

        # Only used for number of assets and tickers
        self.port = rp.Portfolio(returns=self.returns)



    def compute_mu_sigma(self):
        """
        Compute the factor model covariance matrix Σ = B F Bᵀ + D,
        where:
            - B: factor exposure matrix (market + sector)
            - F: factor covariance matrix
            - D: diagonal matrix of residual variances
        """

        # --- 1. Factor-level DataFrame retrieve exposures on final date ---
        window_df = self.bw_daily_df

        end_df = self.bw_daily_df.xs(self.end_date, level='date')

        # --- 2. Construct B matrix: Market and Sector exposures ---
        market_beta = end_df[['market_beta']]  # shape: [n_assets x 1]
        sector_dummies = pd.get_dummies(end_df['bw_sector_name'])  # [n_assets x k_sectors]
        sector_beta = sector_dummies.mul(end_df['sector_beta'], axis=0)  # [n_assets x k]
        self.B = pd.concat([market_beta, sector_beta], axis=1)  # [n_assets x (1 + k)]

        # --- 3. Residual variance matrix D (diagonal) ---
        self.D = pd.DataFrame(np.diag(end_df['residual_variance']),
                              index=self.B.index,
                              columns=self.B.index)

        # --- 4. Expected factor returns matrix f: mean or EWM ---
        mkt_factor_ret = window_df.groupby('date')['market_factor_return'].first().to_frame(name='Market')
        sctr_factor_ret = window_df.groupby(['bw_sector_name', 'date'])['sector_factor_return'].first().unstack().T
        factor_returns = pd.concat([mkt_factor_ret, sctr_factor_ret], axis=1)
        self.factor_returns = factor_returns

        method_f = self.optimizer_config.pop('method_f', None)

        if method_f == 'ewma':
            halflife = self.optimizer_config.pop('halflife')
            self.f = factor_returns.ewm(halflife=halflife).mean().iloc[-1].to_frame(name='f')
        elif method_f == 'hist':
            self.f = factor_returns.mean(axis=0).to_frame(name='f')
        else:
            warnings.warn("method_f not read; using simple mean as default.")
            self.f = factor_returns.mean(axis=0).to_frame(name='f')

        # --- 5. Factor covariance matrix F ---
        method_F = self.optimizer_config.pop('method_F')

        if method_F == 'ewma':
            self.F = factor_returns.ewm(halflife=halflife).cov().iloc[-factor_returns.shape[1]:]  # last cov matrix
        elif method_F == 'hist':
            self.F = factor_returns.cov()
        else:
            warnings.warn("method_F not read; using simple covariance as default.")
            self.F = factor_returns.cov()

        # --- 6. Expected returns μ = B @ f ---

        # Scaled f if not confident in estimate
        f_scalar = self.optimizer_config.pop('f_scalar', None)
        if f_scalar is not None:
            self.scaled_f = self.f * f_scalar
            mu = self.B.values @ self.scaled_f.values
            mu = mu.flatten()
            self.mu = pd.DataFrame([mu], columns=self.B.index, dtype=np.float64)

        else:
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

    def optimize(self):
        """
        Optimizes portfolio using computed mu and sigma from factor model.
        """

        # Check for computed sigma, mu
        if self.sigma is None:
            self.compute_mu_sigma()

        # Assign factor sigma, mu
        self.port.mu = self.mu
        self.port.cov = self.sigma

        # Assign portfolio leverage and short/long budget
        self.port.sht = self.optimizer_config.pop('sht', False)
        self.port.budget = self.optimizer_config.pop('budget', 1.0)
        self.port.uppersht = self.optimizer_config.pop('uppersht', 1.0)
        self.port.upperlng = self.optimizer_config.pop('upperlng', 1.0)


        # Run optimization
        self.w = self.port.optimization(**self.optimizer_config)

        return self.w


"""
class BlackLittermanOptimizer(RiskfolioOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.views = None  # investor views to blend with factor model
        self.asset_classes = None
        self.P = None
        self.Q = None

    def configure_optimizer(self):
        pass

    def optimize(self) -> None:
        pass
"""
