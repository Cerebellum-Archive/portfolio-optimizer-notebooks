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
        self._config = optimizer_config
        if optimizer_config is not None:
            self.input_config = optimizer_config.copy()
        else:
            self.input_config = None

        # Convert xarray dataset to DataFrame
        self.bw_daily_df = self._convert_bw_ds_to_df(daily_ds)

        # Placeholders for optimization components
        self.mu_ = None                    # Expected returns
        self.sigma_ = None                 # Covariance matrix
        self.weights_ = None               # Optimized weights
        self.sharpe_ = None                # Portfolio in-sample Sharpe
        self.port_vol_ = None              # Portfolio volatility
        self.port_return_ = None           # in-sample returns
        self.predicted_return_ = None      # out-of-sample returns
        self.port = None                   # Riskfolio portfolio object


    def configure_optimizer(self):
        raise NotImplementedError("Subclasses must implement the configure_optimizer() method.")

    def fit(self):
        raise NotImplementedError("Subclasses must implement the fit() method.")

    def predict(self, returns_next: xr.Dataset) -> float:
        """
        Predicts realized portfolio return using actual asset returns from the next day.
        These are multiplied by the optimized weights (self.weights_).

        Parameters
        ----------
        returns_next : xr.Dataset
            Slice of the dataset with one date and asset-level returns.
            Expected to be in percentage units (0.52 = 0.52%).
        """
        returns_var = self._config.get("returns_var", "return")

        # Convert to Series (e.g. ticker → return)
        returns_next = returns_next[returns_var].to_series() * 0.01
        returns_next.index.name = 'ticker'

        # Align with optimized weights
        returns_next = returns_next.reindex(self.weights_.index)

        # Predict portfolio return
        self.predicted_return_ = (self.weights_.T @ returns_next).values.item()
        return self.predicted_return_


    def _convert_bw_ds_to_df(self, ds:xr.Dataset) -> pd.DataFrame:
        # Convert xarray dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        df.set_index(["date", "ticker"], inplace=True)
        df.sort_index(level=['date', 'ticker'], inplace=True)

        # Convert returns/variances from percentage to decimal
        for col in df.columns:
            if "return" in col.lower() or "variance" in col.lower():
                df[col] = df[col] * 0.01
        return df

    def _calculate_in_sample_sharpe(self) -> float:
        # Calculate portfolio in-sample Sharpe and volatility
        self.port_return_ = (self.weights_.T @ self.mu_).values.item()
        self.port_vol_ = (np.sqrt(self.weights_.T @ self.sigma_ @ self.weights_)
                          .values.item()
                          )
        rf = self._config.get('rf', 0)
        self.sharpe_ = (self.port_return_ - rf) / self.port_vol_
        return self.sharpe_

    def plot_weights(self):
        if self.weights_ is not None:
            has_negative = (self.weights_ < 0).any().item()

        weights = self.weights_.iloc[:,0] # TODO: convert type try to clean up later
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
        if self.weights_ is not None:
            weights = self.weights_.sort_values(by='weights')
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
                                  mu=self.mu_,
                                  cov=self.port.cov,
                                  returns=self.port.returns,
                                  w=self.weights_,
                                  label='Optimized',
                                  marker='*',
                                  s=16,
                                  c='r')

            # Plot current portfolio if provided
            if current_weights is not None:
                current_weights = current_weights.reindex(self.weights_.index).fillna(0)
                port_return = np.dot(current_weights, self.mu_.values.flatten())
                port_risk = np.sqrt(np.dot(current_weights.T, np.dot(self.port.cov.values, current_weights)))
                ax.scatter(port_risk, port_return, marker='o', s=16, color='red', label='Current Position')

            fig = ax.get_figure()
            fig.set_constrained_layout(True)

            plt.show()


class ClassicOptimizer(RiskfolioOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_optimizer()

    def configure_optimizer(self):
        # Check that returns are provided
        if self.bw_daily_df is None or self.bw_daily_df.empty:
            raise ValueError("Return data (bw_daily_df) is empty or not provided.")

        # Check for NaNs
        if self.bw_daily_df.isnull().any().any():
            raise ValueError("bw_daily_df contains NaN values.")

        if self._config is not None:
            # Check model
            if self._config.get('model') != 'Classic':
                msg = ("ClassicOptimizer requires optimizer_config['model'] == 'Classic'.")
                raise ValueError(msg)

            # Check config keys
            valid_objs = {"Sharpe", "MinRisk", "MaxRet", "Utility"}
            obj = self._config.get("obj", "Sharpe")
            if obj not in valid_objs:
                raise ValueError(f"Invalid objective '{obj}'. Must be one of {valid_objs}.")

        else:
            # Set default config
            self._config = {
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

        if self.input_config is None:
            self.input_config = self._config.copy()

        # Load historical returns into optimizer
        returns_var = self._config.pop('returns_var')
        returns = self.bw_daily_df[returns_var].unstack(level='ticker')

        # Create portfolio
        self.port = rp.Portfolio(returns=returns)

        # Determine mean/covariance method for stats
        method_mu = self._config.pop('method_mu', 'hist')
        method_cov = self._config.pop('method_cov', 'hist')

        # Handle EWMA case with fallback halflife
        if method_mu == 'ewma' or method_cov == 'ewma':
            if self._config.get('hist', None):
                warnings.warn(
                    "The optimizer configuration specified EWMA."
                    "Preventing accidental override: setting 'hist' to False."
                )
                self._config['hist'] = False
                self.input_config['hist'] = False

            halflife = self._config.pop('halflife', None)
            if halflife is None:
                halflife = 30  # Default fallback
                warnings.warn(
                    "Optimizer config did not specify a halflife for EWMA. "
                    "Defaulting to halflife=30."
                )
            mu = returns.ewm(halflife=halflife).mean().iloc[-1]
            cov = returns.ewm(halflife=halflife).cov().iloc[-returns.shape[1]:]

            self.port.mu = mu
            self.port.cov = cov.droplevel('date')

        else:
            # Apply simple 'hist'
            self._config.pop('halflife', None) # Remove 'halflife'
            self.port.assets_stats(method_mu=method_mu, method_cov=method_cov)


    def fit(self):
        """
        Calculates optimized weights from historical returns.
        """

        # Assign sigma, mu used for optimization
        self.sigma_ = self.port.cov

        # Scaled mu if not confident in estimate
        mu_scalar = self._config.pop('mu_scalar', None)
        if mu_scalar is not None:
            scaled_mu = self.port.mu * mu_scalar
            self.port.mu = scaled_mu
            self.mu_ = self.port.mu
        else:
            self.mu_ = self.port.mu
        self.mu_ = self.mu_.squeeze()
        self.mu_.index.name = 'ticker'

        # Assign portfolio leverage and long/short budget
        self.port.sht = self._config.pop('sht', False)
        self.port.budget = self._config.pop('budget', 1.0)
        self.port.uppersht = self._config.pop('uppersht', 1.0)
        self.port.upperlng = self._config.pop('upperlng', 1.0)

        # Run optimization
        self.weights_ = self.port.optimization(**self._config)

        # Reset optimizer_config to input
        self._config = self.input_config

        # Calculate portfolio in-sample Sharpe and volatility
        self._calculate_in_sample_sharpe()


class FactorModelOptimizer(RiskfolioOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_optimizer()

        self.B = None  # factor exposure matrix
        self.F = None  # factor covariance matrix
        self.D = None  # residual matrix
        self.f = None  # expected factor returns vector
        self.f_scaled = None  # scaled factor returns vector


    def configure_optimizer(self):
        # Check that factor returns are provided
        if self.bw_daily_df is None or self.bw_daily_df.empty:
            raise ValueError("Return data (bw_daily_df) is empty or not provided.")

        # Check for NaNs
        if self.bw_daily_df.isnull().any().any():
            raise ValueError("bw_daily_df contains NaN values.")

        if self._config is not None:
            # Check model and hist
            if self._config.get('model') != 'Classic' or self._config.get('hist') is True:
                msg = ("FactorModelOptimizer requires optimizer_config['model'] == 'Classic' "
                       "and optimizer_config['hist'] == False\n"
                       "FactorModelOptimizer precomputes factor covariance matrix (sigma).\n"
                       "FactorModelOptimizer model=='Classic' is required to use precomputed sigma.")
                raise ValueError(msg)

            # Check config keys
            valid_objs = {"Sharpe", "MinRisk", "MaxRet", "Utility"}
            obj = self._config.get("obj", "Sharpe")
            if obj not in valid_objs:
                raise ValueError(f"Invalid objective '{obj}'. Must be one of {valid_objs}.")

        else:
            # Set default config
            self._config = {
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

        if self.input_config is None:
            self.input_config = self._config.copy()

        returns_var = self._config.pop('returns_var')
        returns = self.bw_daily_df[returns_var].unstack(level='ticker')

        # Only used for number of assets and tickers
        self.port = rp.Portfolio(returns=returns)


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

        method_f = self._config.pop('method_f', None)
        halflife = self._config.pop('halflife', None)

        if method_f == 'ewma':
            self.f = factor_returns.ewm(halflife=halflife).mean().iloc[-1].to_frame(name='f')
        elif method_f == 'hist':
            self.f = factor_returns.mean(axis=0).to_frame(name='f')
        else:
            warnings.warn("method_f not read; using simple mean as default.")
            self.f = factor_returns.mean(axis=0).to_frame(name='f')

        # --- 5. Factor covariance matrix F ---
        method_F = self._config.pop('method_F')

        if method_F == 'ewma':
            self.F = factor_returns.ewm(halflife=halflife).cov().iloc[-factor_returns.shape[1]:]  # last cov matrix
        elif method_F == 'hist':
            self.F = factor_returns.cov()
        else:
            warnings.warn("method_F not read; using simple covariance as default.")
            self.F = factor_returns.cov()

        # --- 6. Expected returns μ = B @ f ---

        # Scaled f if not confident in estimate
        f_scalar = self._config.pop('f_scalar', None)
        if f_scalar is not None:
            self.scaled_f = self.f * f_scalar
            mu = self.B.values @ self.scaled_f.values
        else:
            mu = self.B.values @ self.f.values

        self.mu_ = pd.Series(mu.flatten(), index=self.B.index, dtype=np.float64)

        # --- 9. Covariance matrix Σ = B F Bᵗ + D ---
        sigma_vals = self.B.values @ self.F.values @ self.B.values.T + self.D.values
        self.sigma_ = pd.DataFrame(sigma_vals,
                                  index=self.B.index,
                                  columns=self.B.index,
                                  dtype=np.float64
                                  )

    def fit(self):
        """
        Optimizes portfolio using computed mu and sigma from factor model.
        """

        # Check for computed sigma, mu
        if self.sigma_ is None:
            self.compute_mu_sigma()

        # Assign factor sigma, mu
        self.port.mu = self.mu_
        self.port.cov = self.sigma_

        # Assign portfolio leverage and short/long budget
        self.port.sht = self._config.pop('sht', False)
        self.port.budget = self._config.pop('budget', 1.0)
        self.port.uppersht = self._config.pop('uppersht', 1.0)
        self.port.upperlng = self._config.pop('upperlng', 1.0)


        # Run optimization
        self.weights_ = self.port.optimization(**self._config)

        # Reset optimizer_config to input
        self._config = self.input_config

        # Calculate portfolio in-sample Sharpe and volatility
        self._calculate_in_sample_sharpe()


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

    def fit(self) -> None:
        pass
"""
