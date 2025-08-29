import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
import riskfolio as rp
import xarray as xr

from utils.logging_config import setup_logger
logger = setup_logger(__name__)

class BasePortfolioOptimizer(BaseEstimator, TransformerMixin, RegressorMixin):
    """Base scikit-learn compatible portfolio optimizer."""

    def __init__(self):
        """
        Initialize the optimizer with configuration parameters.
        """

        # Internal attributes
        self._fitted_attrs = [
            'mu_', 'sigma_', 'weights_', 'gross_exposure_',
            'sharpe_', 'port_vol_', 'port_return_', 'port_'
        ]

    def _reset_fitted_attrs(self):
        """Reset all fitted attributes."""
        for attr in self._fitted_attrs:
            setattr(self, attr, None)

    def _get_config(self):
        """Get current configuration as dict."""
        config = {}
        for param in self.get_params():
            config[param] = getattr(self, param)
        return config

    def _validate_input(self, X, y=None, require_targets=False):
        """
        Validate input data format.

        Parameters
        ----------
        X : xr.Dataset or pd.DataFrame
            Input features (returns data)
        y : array-like, optional
            Target values (for compatibility, usually None)
        require_targets : bool
            Whether targets are required

        Returns
        -------
        bw_daily_df : pd.DataFrame
            Processed DataFrame in Blue Water format
        """
        if isinstance(X, xr.Dataset):
            # Convert xarray dataset to DataFrame
            df = X.to_dataframe().reset_index()
            df.set_index(["date", "ticker"], inplace=True)
            df.sort_index(level=['date', 'ticker'], inplace=True)

            # Convert returns/variances from percentage to decimal
            for col in df.columns:
                if "return" in col.lower() or "variance" in col.lower():
                    df[col] = df[col] * 0.01

        elif isinstance(X, pd.DataFrame):
            df = X.copy()
            # Assume already in correct format
        else:
            raise ValueError("X must be xr.Dataset or pd.DataFrame")

        # Check for NaNs
        if df.isnull().any().any():
            raise ValueError("Input data contains NaN values")

        return df

    def _calculate_in_sample_stats(self):
        """Calculate portfolio in-sample statistics."""
        if self.weights_ is None or self.mu_ is None or self.sigma_ is None:
            return

        self.port_return_ = float((self.weights_.T @ self.mu_).values.item())
        self.port_vol_ = float(np.sqrt(self.weights_.T @ self.sigma_ @ self.weights_).values.item())

        rf = getattr(self, 'rf', 0.0)
        self.sharpe_ = (self.port_return_ - rf) / self.port_vol_ if self.port_vol_ > 0 else 0.0

    def fit(self, X, y=None):
        """
        Fit the portfolio optimizer.

        Parameters
        ----------
        X : xr.Dataset or pd.DataFrame
            Input returns data
        y : array-like, optional
            Ignored, for sklearn compatibility

        Returns
        -------
        self : object
            Returns self
        """
        # Reset fitted attributes
        self._reset_fitted_attrs()

        # Validate and process input
        self.bw_daily_df_ = self._validate_input(X, y)

        # Store metadata
        dates = self.bw_daily_df_.index.get_level_values('date').unique()
        tickers = self.bw_daily_df_.index.get_level_values('ticker').unique()

        self.start_date_ = dates[0]
        self.end_date_ = dates[-1]
        self.tickers_ = list(tickers)

        # Fit the specific optimizer
        self._fit_optimizer()

        return self

    def _fit_optimizer(self):
        """Override in subclasses to implement specific fitting logic."""
        raise NotImplementedError("Subclasses must implement _fit_optimizer")

    def transform(self, X):
        """
        Transform returns the optimized weights.

        Parameters
        ----------
        X : xr.Dataset or pd.DataFrame
            Input data (can be different from fit data for validation)

        Returns
        -------
        weights : np.ndarray
            Optimized portfolio weights
        """
        check_is_fitted(self)
        return self.weights_.values

    def predict(self, X):
        """
        Predict portfolio returns using optimized weights.
        This method delegates to subclass-specific prediction logic.

        Parameters
        ----------
        X : xr.Dataset or pd.DataFrame
            Future returns/factor data for prediction

        Returns
        -------
        predicted_returns : np.ndarray
            Predicted portfolio returns
        """
        check_is_fitted(self)
        predictions = np.atleast_1d(self._predict_optimizer(X))
        return predictions

    def _predict_optimizer(self, X):
        """Override in subclasses to implement specific prediction logic."""
        raise NotImplementedError("Subclasses must implement _predict_optimizer")

    def get_weights(self):
        """Get optimized weights as pandas Series."""
        check_is_fitted(self)
        return self.weights_.copy().squeeze()

    def get_portfolio_stats(self, active_threshold=0.001):
        """Get portfolio statistics."""
        check_is_fitted(self)
        active_positions = np.abs(self.weights_) > active_threshold
        return {
            'portfolio_return': self.port_return_,
            'portfolio_volatility': self.port_vol_,
            'sharpe_ratio': self.sharpe_,
            'gross_exposure': self.gross_exposure_,
            'num_assets': active_positions.sum().values[0],
            'long_exposure': self.weights_[self.weights_ > 0].sum().values[0],
            'short_exposure': self.weights_[self.weights_ < 0].sum().values[0],
            'net_exposure': self.weights_.sum().values[0]
        }


class ClassicOptimizer(BasePortfolioOptimizer):
    """Scikit-learn compatible ClassicOptimizer.

    Parameters
    ----------
    method_mu : str, default='hist'
        Method for expected returns estimation
    method_cov : str, default='hist'
        Method for covariance estimation
    ewma_mu_halflife : float, optional
        EWMA halflife for returns
    ewma_cov_halflife : float, optional
        EWMA halflife for covariance
    returns_var : str, default='return'
        Column name for returns
    rm : str, default='MV'
        Risk measure
    obj : str, default='Sharpe'
        Optimization objective
    rf : float, default=0.0
        Risk-free rate
    l : float, default=0.0
        Risk aversion parameter
    sht : bool, default=False
        Allow short positions
    budget : float, default=1.0
        Total budget constraint
    budgetsht : float, default=0.2
        Short budget constraint
    uppersht : float, default=0.2
        Upper bound for short positions
    upperlng : float, default=1.0
        Upper bound for long positions
    user_input_mu : pd.Series, optional
        Custom expected returns
    user_input_cov : pd.DataFrame, optional
        Custom covariance matrix
    """

    def __init__(self,
                 method_mu='hist',
                 method_cov='hist',
                 ewma_mu_halflife=None,
                 ewma_cov_halflife=None,
                 returns_var='return',
                 rm='MV',
                 obj='Sharpe',
                 rf=0.0,
                 l=0.0,
                 sht=False,
                 budget=1.0,
                 budgetsht=0.2,
                 uppersht=0.2,
                 upperlng=1.0,
                 user_input_mu=None,
                 user_input_cov=None):

        # Store all parameters as instance attributes (required for sklearn)
        self.method_mu = method_mu
        self.method_cov = method_cov
        self.ewma_mu_halflife = ewma_mu_halflife
        self.ewma_cov_halflife = ewma_cov_halflife
        self.returns_var = returns_var
        self.rm = rm
        self.obj = obj
        self.rf = rf
        self.l = l
        self.sht = sht
        self.budget = budget
        self.budgetsht = budgetsht
        self.uppersht = uppersht
        self.upperlng = upperlng
        self.user_input_mu = user_input_mu
        self.user_input_cov = user_input_cov

        # Call parent constructor
        super().__init__()
        super()._reset_fitted_attrs()

    def _halflife_to_d(self, halflife):
        """Convert halflife to decay parameter."""
        return 2 ** (-1 / halflife)

    def _d_to_halflife(self, d):
        """Convert decay parameter to halflife."""
        return -1 / np.log2(d)

    def _fit_optimizer(self):
        """Fit the Classic optimizer."""
        # Create returns matrix
        returns = self.bw_daily_df_[self.returns_var].unstack(level='ticker')

        # Create portfolio object
        self.port_ = rp.Portfolio(returns=returns)

        # Handle custom inputs
        mu = self.user_input_mu
        sigma = self.user_input_cov

        # Validate custom inputs if provided
        if mu is not None and self.method_mu == 'custom':
            mu = mu.sort_index()
            mu.index.name = "ticker"
            if mu.shape[0] != returns.shape[1]:
                raise ValueError(f"mu length {mu.shape[0]} does not match number of assets {returns.shape[1]}")
            mu = mu.reindex(returns.columns)

        if sigma is not None and self.method_cov == 'custom':
            sigma = sigma.sort_index().sort_index(axis=1)
            sigma.index.name = "ticker"
            sigma.columns.name = "ticker"
            if sigma.shape != (returns.shape[1], returns.shape[1]):
                raise ValueError(f"Sigma shape {sigma.shape} does not match number of assets")
            sigma = sigma.reindex(index=returns.columns, columns=returns.columns)

        # Prepare method parameters
        dict_mu = {}
        dict_cov = {}

        # EWMA conversions if needed
        if self.method_mu in ('ewma1', 'ewma2'):
            if self.ewma_mu_halflife is not None:
                dict_mu['d'] = self._halflife_to_d(self.ewma_mu_halflife)

        if self.method_cov in ('ewma1', 'ewma2'):
            if self.ewma_cov_halflife is not None:
                dict_cov['d'] = self._halflife_to_d(self.ewma_cov_halflife)

        # Calculate statistics based on method combination
        if self.method_mu != 'custom' and self.method_cov != 'custom':
            self.port_.assets_stats(method_mu=self.method_mu, method_cov=self.method_cov,
                                   dict_mu=dict_mu, dict_cov=dict_cov)
        elif self.method_mu == 'custom' and self.method_cov != 'custom':
            self.port_.assets_stats(method_cov=self.method_cov, dict_cov=dict_cov)
            self.port_.mu = mu
        elif self.method_cov == 'custom' and self.method_mu != 'custom':
            self.port_.assets_stats(method_mu=self.method_mu, dict_mu=dict_mu)
            self.port_.cov = sigma
        else:
            self.port_.mu = mu
            self.port_.cov = sigma

        # Store computed statistics
        self.sigma_ = self.port_.cov.copy()
        self.sigma_.index.name = 'ticker'
        self.sigma_.columns.name = 'ticker'
        self.mu_ = self.port_.mu.squeeze().copy()
        if np.isscalar(self.mu_):
            self.mu_ = pd.Series([self.mu_], index=self.sigma_.index)
        self.mu_.index.name = 'ticker'
        self.mu_.name = None

        # Set portfolio constraints
        self.port_.sht = self.sht
        self.port_.budget = self.budget
        self.port_.budgetsht = self.budgetsht
        self.port_.uppersht = self.uppersht
        self.port_.upperlng = self.upperlng

        # Prepare optimization config
        opt_config = {
            'rm': self.rm,
            'obj': self.obj,
            'rf': self.rf,
            'l': self.l
        }

        # Run optimization
        self.weights_ = self.port_.optimization(**opt_config)

        # Calculate derived statistics
        self.gross_exposure_ = self.weights_.abs().sum()
        self._calculate_in_sample_stats()

    def _predict_optimizer(self, X):
        """
        Classic optimizer prediction: w^T * r
        Requires asset-level returns (tickers × time).
        """
        if isinstance(X, xr.Dataset):
            returns_var = getattr(self, 'returns_var', 'return')
            returns = X[returns_var].sel(ticker=self.weights_.index) * 0.01

            # Convert weights to xarray
            w = self.get_weights()
            w = xr.DataArray(
                w.values,
                dims=["ticker"],
                coords={"ticker": w.index},
            )

            # Portfolio returns: w^T * r
            predictions = xr.dot(returns, w, dim='ticker').values
            return predictions

        elif isinstance(X, pd.DataFrame):
            # Assume X is already in returns format (tickers as columns)
            aligned_returns = X.reindex(columns=self.weights_.index)
            predictions = (aligned_returns @ self.weights_.values).values
            return predictions

        else:
            raise ValueError("X must be xr.Dataset or pd.DataFrame")


class FactorModelOptimizer(BasePortfolioOptimizer):
    """Scikit-learn compatible Factor Model optimizer.

    Parameters
    ----------
    method_f : str, default='ewma1'
        Method for factor return estimation
    method_F : str, default='ewma1'
        Method for factor covariance estimation
    halflife : int, default=30
        EWMA halflife in days
    returns_var : str, default='return'
        Column name for returns
    rm : str, default='MV'
        Risk measure
    obj : str, default='Sharpe'
        Optimization objective
    rf : float, default=0.0
        Risk-free rate
    l : float, default=0.0
        Risk aversion parameter
    sht : bool, default=True
        Allow short positions
    budget : float, default=1.0
        Total budget constraint
    budgetsht : float, default=0.2
        Short budget constraint
    uppersht : float, default=0.2
        Upper bound for short positions
    upperlng : float, default=1.0
        Upper bound for long positions
    user_input_mu : pd.Series, optional
        Custom expected returns
    user_input_cov : pd.DataFrame, optional
        Custom covariance matrix
    use_custom_inputs : bool, default=False
        Whether to use custom mu/sigma instead of factor model computation
    user_input_f : pd.Series, optional
        Custom expected factor returns
    user_input_F : pd.DataFrame, optional
        Custom factor covariance matrix
    use_custom_factor_inputs : bool, default=False
        Whether to use custom f/F instead of factor model computation
    """

    def __init__(self,
                 method_f='ewma1',
                 method_F='ewma1',
                 halflife=30,
                 returns_var='return',
                 rm='MV',
                 obj='Sharpe',
                 rf=0.0,
                 l=0.0,
                 sht=True,
                 budget=1.0,
                 budgetsht=0.2,
                 uppersht=0.2,
                 upperlng=1.0,
                 user_input_mu=None,
                 user_input_cov=None,
                 use_custom_inputs=False,
                 user_input_f=None,
                 user_input_F=None,
                 use_custom_factor_inputs=False):

        # Store all parameters as instance attributes (required for sklearn)
        self.method_f = method_f
        self.method_F = method_F
        self.halflife = halflife
        self.returns_var = returns_var
        self.rm = rm
        self.obj = obj
        self.rf = rf
        self.l = l
        self.sht = sht
        self.budget = budget
        self.budgetsht = budgetsht
        self.uppersht = uppersht
        self.upperlng = upperlng
        self.user_input_mu = user_input_mu
        self.user_input_cov = user_input_cov
        self.use_custom_inputs = use_custom_inputs
        self.user_input_f = user_input_f
        self.user_input_F = user_input_F
        self.use_custom_factor_inputs = use_custom_factor_inputs

        # Factor model specific attributes
        self._factor_attrs = ['B_', 'F_', 'D_', 'f_', 'mu_residual_']

        # Call parent constructor
        super().__init__()
        self._reset_fitted_attrs()

    def _reset_fitted_attrs(self):
        """Reset fitted attributes including factor-specific ones."""
        super()._reset_fitted_attrs()
        for attr in self._factor_attrs:
            setattr(self, attr, None)

    def _validate_factor_data(self, df):
        """Validate factor model data requirements."""
        if self.use_custom_inputs:
            # Skip validation if using custom inputs for BOTH mu and sigma
            if self.user_input_mu is not None and self.user_input_cov is not None:
                return

        required_cols = [
            'market_beta', 'sector_beta', 'bw_sector_name',
            'residual_variance', 'market_factor_return',
            'sector_factor_return', 'residual_return'
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required factor model columns: {missing_cols}")

    def _validate_custom_mu_sigma_inputs(self, returns):
        """Validate custom mu and sigma inputs."""
        mu = self.user_input_mu
        sigma = self.user_input_cov

        # Must provide at least one
        if mu is None and sigma is None:
            raise ValueError("At least one of user_input_mu or user_input_cov must be provided when use_custom_inputs=True")

        if mu is not None:
            if not isinstance(mu, (pd.Series, np.ndarray)):
                raise ValueError("user_input_mu must be a pandas Series or numpy array")
            if isinstance(mu, np.ndarray):
                mu = pd.Series(mu.flatten(), index=returns.columns)
            else:
                mu = mu.sort_index()
                mu.index.name = "ticker"
                if mu.shape[0] != returns.shape[1]:
                    raise ValueError(f"mu length {mu.shape[0]} does not match number of assets {returns.shape[1]}")
                # Align with returns columns
                mu = mu.reindex(returns.columns)
            self.user_input_mu = mu

        if sigma is not None:
            if not isinstance(sigma, (pd.DataFrame, np.ndarray)):
                raise ValueError("user_input_cov must be a pandas DataFrame or numpy array")
            if isinstance(sigma, np.ndarray):
                sigma = pd.DataFrame(sigma, index=returns.columns, columns=returns.columns)
            else:
                sigma = sigma.sort_index().sort_index(axis=1)
                sigma.index.name = "ticker"
                sigma.columns.name = "ticker"
                if sigma.shape != (returns.shape[1], returns.shape[1]):
                    raise ValueError(f"Sigma shape {sigma.shape} does not match number of assets")
                # Align with returns columns
                sigma = sigma.reindex(index=returns.columns, columns=returns.columns)
            self.user_input_cov = sigma

    def _validate_custom_factor_inputs(self, B_matrix):
        """Validate custom f and F inputs.

        - f (expected factor returns) is optional, like mu.
        - F (factor covariance) is optional, like sigma.
        - At least one of them must be provided if use_custom_factor_inputs=True.
        """
        f = self.user_input_f
        F = self.user_input_F

        # Must provide at least one
        if f is None and F is None:
            raise ValueError("At least one of user_input_f or user_input_F must be provided when use_custom_factor_inputs=True")

        # Validate f
        if f is not None:
            if not isinstance(f, (pd.Series, np.ndarray)):
                raise ValueError("user_input_f must be a pandas Series or numpy array")
            if isinstance(f, np.ndarray):
                f = pd.Series(f.flatten(), index=B_matrix.columns)
            else:
                f = f.reindex(B_matrix.columns)
            if f.shape[0] != B_matrix.shape[1]:
                raise ValueError(f"user_input_f length {f.shape[0]} does not match number of factors {B_matrix.shape[1]}")
            self.user_input_f = f

        # Validate F
        if F is not None:
            if not isinstance(F, (pd.DataFrame, np.ndarray)):
                raise ValueError("user_input_F must be a pandas DataFrame or numpy array")
            if isinstance(F, np.ndarray):
                F = pd.DataFrame(F, index=B_matrix.columns, columns=B_matrix.columns)
            else:
                F = F.reindex(index=B_matrix.columns, columns=B_matrix.columns)
            if F.shape != (B_matrix.shape[1], B_matrix.shape[1]):
                raise ValueError(f"user_input_F shape {F.shape} does not match number of factors")
            self.user_input_F = F


    def _fit_optimizer(self):
        """Fit the Factor Model optimizer."""
        # Create basic portfolio object for optimization
        returns = self.bw_daily_df_[self.returns_var].unstack(level='ticker')
        self.port_ = rp.Portfolio(returns=returns)

        # Construct factor exposure matrix B and residual variance D from input data X
        # Only construct B and D if we need factor model computation
        if not (self.use_custom_inputs and
                self.user_input_mu is not None and
                self.user_input_cov is not None):
            self._construct_B_and_D()

        if self.use_custom_inputs:
            # Use custom mu/sigma inputs
            self._fit_with_custom_mu_sigma_inputs(returns)
        elif self.use_custom_factor_inputs:
            # Use custom f/F inputs
            self._fit_with_custom_factor_inputs()
        else:
            # Validate factor model data
            self._validate_factor_data(self.bw_daily_df_)
            # Compute factor model statistics
            self._compute_factor_model_stats()

        # Assign computed statistics to portfolio
        self.port_.mu = self.mu_
        self.port_.cov = self.sigma_

        # Set portfolio constraints
        self.port_.sht = self.sht
        self.port_.budget = self.budget
        self.port_.budgetsht = self.budgetsht
        self.port_.uppersht = self.uppersht
        self.port_.upperlng = self.upperlng

        # Prepare optimization config
        opt_config = {
            'model': 'Classic',  # Required for factor models
            'rm': self.rm,
            'obj': self.obj,
            'rf': self.rf,
            'l': self.l,
            'hist': False
        }

        # Run optimization
        self.weights_ = self.port_.optimization(**opt_config)

        # Calculate derived statistics
        self.gross_exposure_ = self.weights_.abs().sum()
        self._calculate_in_sample_stats()

    def _fit_with_custom_mu_sigma_inputs(self, returns):
        """Fit optimizer using custom mu and/or sigma inputs."""
        logger.info("Using custom mu and/or sigma inputs instead of factor model computation...")

        # Validate custom inputs
        self._validate_custom_mu_sigma_inputs(returns)

        # Initialize with None
        mu = None
        sigma = None

        # Handle custom mu
        if self.user_input_mu is not None:
            mu = self.user_input_mu.copy()
            mu.index.name = 'ticker'
            mu.name = None
            logger.info(f"Using custom mu with shape: {mu.shape}")

        # Handle custom sigma
        if self.user_input_cov is not None:
            sigma = self.user_input_cov.copy()
            sigma.index.name = 'ticker'
            sigma.columns.name = 'ticker'

            # Ensure positive definite for sigma
            eigenvals = np.linalg.eigvals(sigma.values)
            if np.any(eigenvals <= 0):
                logger.warning("Custom covariance matrix is not positive definite. "
                              "Adding small regularization.")
                sigma += np.eye(sigma.shape[0]) * 1e-8

            logger.info(f"Using custom sigma with shape: {sigma.shape}")

        # Compute missing components using factor model
        if mu is None or sigma is None:
            # Compute mu or sigma from factor model
            logger.info("Computing missing factor model stats from factor data.")
            self._compute_factor_model_stats()  # This computes mu_, sigma_, f_, etc.

        if sigma is not None:
            # Assign portfolio sigma_ to user_input_cov
            logger.info("Assigning portfolio sigma to user input sigma.")
            self.sigma_ = sigma

        if mu is not None:
            # Assign portfolio sigma_ to user_input_cov
            logger.info("Assigning portfolio mu to user input mu.")
            self.mu_ = mu

        logger.info(f"Final mu shape: {self.mu_.shape}, sigma shape: {self.sigma_.shape}")

    def _fit_with_custom_factor_inputs(self):
        """Fit optimizer using custom f and/or F inputs with flexible combinations."""
        logger.info("Using custom f and/or F inputs...")

        # Validate custom f and F
        self._validate_custom_factor_inputs(self.B_)

        # Initialize with None
        f = None
        F = None

        # Handle custom f
        if self.user_input_f is not None:
            f = self.user_input_f.copy()
            logger.info(f"Using custom f with shape: {f.shape}")

        # Handle custom F
        if self.user_input_F is not None:
            F = self.user_input_F.copy()
            logger.info(f"Using custom F with shape: {F.shape}")

        # Compute missing components from data
        if f is None or F is None:
            logger.info("Computing missing factor statistics from data")
            self._compute_factor_model_stats()  # This computes both f_ and F_

        # Use custom inputs where provided
        if f is not None:
            self.f_ = f
        # self.f_ already set by _compute_factor_model_stats() if f is None

        if F is not None:
            self.F_ = F
            # Ensure positive definite for F
            eigenvals = np.linalg.eigvals(self.F_.values)
            if np.any(eigenvals <= 0):
                logger.warning("Custom factor covariance matrix is not positive definite. "
                              "Adding small regularization.")
                self.F_ += np.eye(self.F_.shape[0]) * 1e-8
        # self.F_ already set by _compute_factor_model_stats() if F is None

        # Now compute asset-level statistics using final f_ and F_
        # Expected asset returns μ = B @ f
        try:
            mu_vals = self.B_.values @ self.f_.values.flatten()
            self.mu_ = pd.Series(mu_vals, index=self.B_.index, name='mu', dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Error computing expected asset returns: {e}")

        # Asset covariance matrix Σ = B F B^T + D
        try:
            BFBt = self.B_.values @ self.F_.values @ self.B_.values.T
            sigma_vals = BFBt + self.D_.values

            self.sigma_ = pd.DataFrame(
                sigma_vals,
                index=self.B_.index,
                columns=self.B_.index,
                dtype=np.float64
            )

            # Ensure positive definite
            eigenvals = np.linalg.eigvals(self.sigma_.values)
            if np.any(eigenvals <= 0):
                logger.warning("Asset covariance matrix is not positive definite. "
                              "Adding small regularization.")
                self.sigma_ += np.eye(self.sigma_.shape[0]) * 1e-8

            logger.info(f"Final asset covariance matrix Σ shape: {self.sigma_.shape}")

        except Exception as e:
            raise ValueError(f"Error computing asset covariance matrix: {e}")

        # Handle residual returns (only if computed from data)
        if f is None:
            # self.mu_residual_ already computed in _compute_factor_model_stats()
            pass
        else:
            # With custom f, residual returns aren't directly available
            self.mu_residual_ = None

        logger.info(f"Final mu shape: {self.mu_.shape}, sigma shape: {self.sigma_.shape}")

    def _construct_B_and_D(self):
        """Construct factor exposure matrix B and residual variance D from input data."""
        logger.info("Constructing factor exposure matrix B and residual variance D...")
        # Get exposures from final date
        try:
            end_df = self.bw_daily_df_.xs(self.end_date_, level='date')
        except KeyError:
            raise ValueError(f"End date {self.end_date_} not found in data")

        # Construct factor exposure matrix B
        try:
            market_beta = end_df[['market_beta']].copy()
            market_beta.columns = ['Market']

            sector_dummies = pd.get_dummies(end_df['bw_sector_name'])
            sector_beta = sector_dummies.mul(end_df['sector_beta'], axis=0)

            self.B_ = pd.concat([market_beta, sector_beta], axis=1).astype(np.float64)

            logger.info(f"Factor exposure matrix B shape: {self.B_.shape}")

        except KeyError as e:
            raise ValueError(f"Error constructing factor exposure matrix: {e}")

        # Residual variance matrix D (diagonal)
        try:
            residual_var = end_df['residual_variance'].values
            if np.any(residual_var <= 0):
                raise ValueError("Residual variances must be positive")

            self.D_ = pd.DataFrame(
                np.diag(residual_var),
                index=self.B_.index,
                columns=self.B_.index,
                dtype=np.float64
            )
        except Exception as e:
            raise ValueError(f"Error constructing residual variance matrix: {e}")
        logger.info("B and D matrices constructed.")

    def _compute_factor_model_stats(self):
        """Compute factor model mu and sigma."""
        logger.info("Computing factor model mu and sigma...")

        # Construct factor returns
        try:
            mkt_factor_ret = (self.bw_daily_df_.groupby('date')['market_factor_return']
                             .first().to_frame(name='Market'))

            sector_data = (self.bw_daily_df_.groupby(['bw_sector_name', 'date'])
                          ['sector_factor_return'].first().unstack(level=0))

            factor_returns = pd.concat([mkt_factor_ret, sector_data], axis=1)
            factor_returns = factor_returns.reindex(columns=self.B_.columns)

            if factor_returns.isnull().any().any():
                raise ValueError("Factor returns contain NaN values after alignment")

        except Exception as e:
            raise ValueError(f"Error constructing factor returns: {e}")

        # Expected factor returns
        try:
            if self.method_f == 'ewma1':
                self.f_ = (factor_returns.ewm(halflife=self.halflife, adjust=True)
                          .mean().iloc[-1]
                          .to_frame(name='f')
                          )
            elif self.method_f == 'ewma2':
                self.f_ = (factor_returns.ewm(halflife=self.halflife, adjust=False)
                          .mean().iloc[-1]
                          .to_frame(name='f')
                          )
            elif self.method_f == 'hist':
                self.f_ = factor_returns.mean(axis=0).to_frame(name='f')
            else:
                raise ValueError(f"Unknown method_f: {self.method_f}")

            logger.info(f"Expected factor returns method: {self.method_f}")

        except Exception as e:
            raise ValueError(f"Error computing expected factor returns: {e}")

        # Factor covariance matrix
        try:
            if self.method_F == 'ewma1':
                ewm_cov = factor_returns.ewm(halflife=self.halflife, adjust=True).cov()
                n_factors = factor_returns.shape[1]
                self.F_ = ewm_cov.iloc[-n_factors:].copy()
            elif self.method_F == 'ewma2':
                ewm_cov = factor_returns.ewm(halflife=self.halflife, adjust=False).cov()
                n_factors = factor_returns.shape[1]
                self.F_ = ewm_cov.iloc[-n_factors:].copy()
            elif self.method_F == 'hist':
                self.F_ = factor_returns.cov()
            else:
                raise ValueError(f"Unknown method_F: {self.method_F}")

            # Ensure positive definite
            eigenvals = np.linalg.eigvals(self.F_.values)
            if np.any(eigenvals <= 0):
                logger.warning("Factor covariance matrix is not positive definite. "
                              "Adding small regularization.")
                self.F_ += np.eye(self.F_.shape[0]) * 1e-8

            logger.info(f"Factor covariance method: {self.method_F}")

        except Exception as e:
            raise ValueError(f"Error computing factor covariance matrix: {e}")

        # Expected asset returns μ = B @ f
        try:
            mu_vals = self.B_.values @ self.f_.values.flatten()
            self.mu_ = pd.Series(mu_vals, index=self.B_.index, name='mu', dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Error computing expected asset returns: {e}")

        # Residual returns (for analysis)
        try:
            res_ret = self.bw_daily_df_['residual_return'].unstack(level='ticker')
            res_ret = res_ret.reindex(columns=self.B_.index)

            if self.method_f in ('ewma1', 'ewma2'):
                if self.method_f == 'ewma1':
                    self.mu_residual_ = res_ret.ewm(halflife=self.halflife, adjust=True).mean().iloc[-1]
                else:
                    self.mu_residual_ = res_ret.ewm(halflife=self.halflife, adjust=False).mean().iloc[-1]
            elif self.method_f == 'hist':
                self.mu_residual_ = res_ret.mean(axis=0)
            else:
                self.mu_residual_ = res_ret.mean(axis=0)

        except Exception as e:
            raise ValueError(f"Error computing residual returns: {e}")

        # Asset covariance matrix Σ = B F B^T + D
        try:
            BFBt = self.B_.values @ self.F_.values @ self.B_.values.T
            sigma_vals = BFBt + self.D_.values

            self.sigma_ = pd.DataFrame(
                sigma_vals,
                index=self.B_.index,
                columns=self.B_.index,
                dtype=np.float64
            )

            # Ensure positive definite
            eigenvals = np.linalg.eigvals(self.sigma_.values)
            if np.any(eigenvals <= 0):
                logger.warning("Asset covariance matrix is not positive definite. "
                              "Adding small regularization.")
                self.sigma_ += np.eye(self.sigma_.shape[0]) * 1e-8

            logger.info(f"Asset covariance matrix Σ shape: {self.sigma_.shape}")

        except Exception as e:
            raise ValueError(f"Error computing asset covariance matrix: {e}")

    def _predict_optimizer(self, X):
        """
        Factor model prediction: w^T * B * f
        Can work with either factor returns directly or compute from asset returns.
        """
        if self.use_custom_inputs or self.B_ is None:
            # Fall back to classic prediction if no factor structure
            logger.info("Using classic prediction fallback for factor model")
            predictions = self._predict_classic_fallback(X)
        elif self.use_custom_factor_inputs:
            # If custom f and F were provided, use them for prediction
            # This assumes X contains the assets for which we want to predict returns
            # and that B_ is already constructed from the training data.
            # The prediction is w^T * (B * f_ + residual_returns)
            # Since residual_returns are not available with custom f/F, we use B*f_
            # This is a simplification, as a full prediction would require future residual returns.
            # For now, we'll assume the prediction is based purely on factor returns.
            if self.f_ is None:
                raise ValueError("Custom factor returns (f_) not set for prediction.")
            if self.B_ is None:
                raise ValueError("Factor exposure matrix (B_) not set for prediction.")

            # Portfolio factor exposures: w^T * B
            portfolio_exposures = (self.weights_.values.T @ self.B_.values).flatten()

            # Predicted portfolio return = portfolio_exposures @ f_
            predictions = (portfolio_exposures @ self.f_.values).flatten()
            # If X is provided, we might want to align it, but for now, this is a simple prediction.
            # This part might need refinement based on how prediction is expected to work with custom f/F.
            # For now, it's a single predicted return based on the fitted f_.
            return predictions
        else:
            if isinstance(X, xr.Dataset):
                # Try to get factor returns directly
                factor_vars = ['market_factor_return', 'sector_factor_return']
                if all(var in X.data_vars for var in factor_vars):
                    # Use factor returns directly
                    predictions = self._predict_from_factor_returns_xarray(X)
                else:
                    # Fall back to asset-level prediction
                    predictions = self._predict_classic_fallback(X)

            elif isinstance(X, pd.DataFrame):
                # Check if X contains factor returns or asset returns
                if self.B_ is not None and all(factor in X.columns for factor in self.B_.columns):
                    # X contains factor returns
                    predictions = self._predict_from_factor_returns_pandas(X)
                else:
                    # Fall back to asset-level prediction
                    predictions = self._predict_classic_fallback(X)

            else:
                raise ValueError("X must be xr.Dataset or pd.DataFrame")

        return predictions

    def _predict_from_factor_returns_xarray(self, X):
        """Predict from factor returns using xarray."""
        # Get portfolio factor exposures: w^T * B
        portfolio_exposures = (self.weights_.values.T @ self.B_.values).flatten()

        # Construct factor returns array
        market_ret = X['market_factor_return'] * 0.01
        sector_rets = []

        for sector in self.B_.columns[1:]:  # Skip 'Market'
            if f'sector_factor_return_{sector}' in X.data_vars:
                sector_rets.append(X[f'sector_factor_return_{sector}'] * 0.01)
            else:
                # Use generic sector return if specific sector not found
                sector_rets.append(X['sector_factor_return'] * 0.01)

        # Stack factor returns
        factor_returns = xr.concat([market_ret] + sector_rets, dim='factor')
        factor_returns = factor_returns.assign_coords(factor=self.B_.columns)

        # Portfolio return: portfolio_exposures^T * factor_returns
        exposures_da = xr.DataArray(portfolio_exposures, dims=['factor'],
                                   coords={'factor': self.B_.columns})

        predictions = xr.dot(exposures_da, factor_returns, dim='factor').values
        return predictions

    def _predict_from_factor_returns_pandas(self, X):
        """Predict from factor returns using pandas DataFrame."""
        # Get portfolio factor exposures: w^T * B
        portfolio_exposures = (self.weights_.values.T @ self.B_.values).flatten()

        # Align factor returns with portfolio exposures
        factor_returns = X[self.B_.columns]

        # Portfolio return: portfolio_exposures^T * factor_returns
        predictions = (factor_returns.values @ portfolio_exposures).flatten()
        return predictions

    def _predict_classic_fallback(self, X):
        """Fallback to classic asset-level prediction."""
        if isinstance(X, xr.Dataset):
            returns_var = getattr(self, 'returns_var', 'return')
            returns = X[returns_var].sel(ticker=self.weights_.index) * 0.01

            # Convert weights to xarray
            w = self.get_weights()
            w = xr.DataArray(w.values, dims=["ticker"], coords={"ticker": w.index})

            predictions = xr.dot(returns, w, dim='ticker').values
            return predictions

        elif isinstance(X, pd.DataFrame):
            aligned_returns = X.reindex(columns=self.weights_.index)
            predictions = (aligned_returns @ self.weights_.values).values
            return predictions

    def get_factor_exposures(self):
        """Get portfolio factor exposures."""
        check_is_fitted(self)

        # Only available if factor model was computed (not custom inputs)
        if self.use_custom_inputs or self.B_ is None:
            logger.warning("Factor exposures not available when using custom inputs")
            return None

        w = self.weights_.values.flatten()
        factor_exposures = self.B_.values.T @ w

        return pd.Series(
            factor_exposures,
            index=self.B_.columns,
            name='portfolio_exposure'
        )

    def decompose_returns(self):
        """Decompose expected returns into factor and residual components."""
        check_is_fitted(self)

        # Only available if factor model was computed (not custom inputs)
        if self.use_custom_inputs or self.B_ is None or self.f_ is None:
            logger.warning("Return decomposition not available when using custom inputs")
            return {
                'factor_contribution': None,
                'residual_contribution': None,
                'total_expected_return': self.port_return_,
                'portfolio_expected_return': self.port_return_
            }

        w = self.weights_.values.flatten()

        # Factor contribution
        factor_returns = (self.B_.values @ self.f_.values.flatten())
        factor_contrib = np.dot(factor_returns, w)

        # Residual contribution
        resid_contrib = np.dot(self.mu_residual_.values, w)

        return {
            'factor_contribution': factor_contrib,
            'residual_contribution': resid_contrib,
            'total_expected_return': factor_contrib + resid_contrib,
            'portfolio_expected_return': self.port_return_
        }

    def get_portfolio_stats(self, active_threshold=0.001):
        """Get extended portfolio statistics including factor exposures."""
        stats = super().get_portfolio_stats(active_threshold=active_threshold)

        # Add factor-specific stats only if factor model was computed
        if not self.use_custom_inputs and hasattr(self, 'B_') and self.B_ is not None:
            stats['num_factors'] = self.B_.shape[1]
            factor_exp = self.get_factor_exposures()
            if factor_exp is not None:
                stats['factor_exposures'] = factor_exp.to_dict()

            # Add return decomposition
            try:
                decomp = self.decompose_returns()
                stats.update(decomp)
            except:
                pass
        else:
            stats['using_custom_inputs'] = self.use_custom_inputs

        return stats


class BlackLittermanOptimizer(BasePortfolioOptimizer):
    """Scikit-learn compatible Black-Litterman optimizer.

    The Black-Litterman model combines market equilibrium returns with
    investor views to generate expected returns and optimal portfolios.

    Parameters
    ----------
    method_mu : str, default='hist'
        Method for market returns estimation
    method_cov : str, default='hist'
        Method for covariance estimation
    ewma_mu_halflife : float, optional
        EWMA halflife for returns
    ewma_cov_halflife : float, optional
        EWMA halflife for covariance
    returns_var : str, default='return'
        Column name for returns
    rm : str, default='MV'
        Risk measure
    obj : str, default='Sharpe'
        Optimization objective
    rf : float, default=0.0
        Risk-free rate
    l : float, default=0.0
        Risk aversion parameter
    sht : bool, default=False
        Allow short positions
    budget : float, default=1.0
        Total budget constraint
    budgetsht : float, default=0.2
        Short budget constraint
    uppersht : float, default=0.2
        Upper bound for short positions
    upperlng : float, default=1.0
        Upper bound for long positions
    delta : float, optional
        Risk aversion coefficient. If None, estimated from market cap weights
    equilibrium : bool, default=True
        Whether to use equilibrium returns as prior
    P : pd.DataFrame, optional
        Picking matrix for views (n_views x n_assets)
    Q : pd.Series, optional
        View portfolio expected returns (n_views,)
    Omega : pd.DataFrame, optional
        Uncertainty matrix for views (n_views x n_views)
    tau : float, default=1.0
        Scales the uncertainty of the prior
    market_cap : pd.Series, optional
        Market capitalizations for equilibrium calculation
    """

    def __init__(self,
                 method_mu='hist',
                 method_cov='hist',
                 ewma_mu_halflife=None,
                 ewma_cov_halflife=None,
                 returns_var='return',
                 rm='MV',
                 obj='Sharpe',
                 rf=0.0,
                 l=0.0,
                 sht=False,
                 budget=1.0,
                 budgetsht=0.2,
                 uppersht=0.2,
                 upperlng=1.0,
                 delta=None,
                 equilibrium=True,
                 P=None,
                 Q=None,
                 Omega=None,
                 tau=1.0,
                 market_cap=None):

        # Store all parameters as instance attributes
        self.method_mu = method_mu
        self.method_cov = method_cov
        self.ewma_mu_halflife = ewma_mu_halflife
        self.ewma_cov_halflife = ewma_cov_halflife
        self.returns_var = returns_var
        self.rm = rm
        self.obj = obj
        self.rf = rf
        self.l = l
        self.sht = sht
        self.budget = budget
        self.budgetsht = budgetsht
        self.uppersht = uppersht
        self.upperlng = upperlng
        self.delta = delta
        self.equilibrium = equilibrium
        self.P = P
        self.Q = Q
        self.Omega = Omega
        self.tau = tau
        self.market_cap = market_cap

        # Black-Litterman specific attributes
        self._bl_attrs = ['Pi_', 'mu_bl_', 'sigma_bl_', 'w_eq_', 'delta_']

        # Call parent constructor
        super().__init__()
        self._reset_fitted_attrs()

    def _reset_fitted_attrs(self):
        """Reset fitted attributes including BL-specific ones."""
        super()._reset_fitted_attrs()
        for attr in self._bl_attrs:
            setattr(self, attr, None)

    def _halflife_to_d(self, halflife):
        """Convert halflife to decay parameter."""
        return 2 ** (-1 / halflife)

    def _validate_bl_inputs(self, n_assets):
        """Validate Black-Litterman specific inputs."""
        if self.P is not None:
            if not isinstance(self.P, pd.DataFrame):
                raise ValueError("P must be a pandas DataFrame")
            if self.P.shape[1] != n_assets:
                raise ValueError(f"P must have {n_assets} columns (one per asset)")

        if self.Q is not None:
            if not isinstance(self.Q, (pd.Series, np.ndarray)):
                raise ValueError("Q must be a pandas Series or numpy array")
            if self.P is not None and len(self.Q) != self.P.shape[0]:
                raise ValueError("Q length must match number of rows in P")

        if self.Omega is not None:
            if not isinstance(self.Omega, pd.DataFrame):
                raise ValueError("Omega must be a pandas DataFrame")
            if self.P is not None:
                n_views = self.P.shape[0]
                if self.Omega.shape != (n_views, n_views):
                    raise ValueError(f"Omega must be {n_views}x{n_views} matrix")

        # Check that views are specified together
        if any([self.P is not None, self.Q is not None, self.Omega is not None]):
            if not all([self.P is not None, self.Q is not None]):
                raise ValueError("P and Q must both be specified for views")

    def _estimate_delta(self, returns, market_cap=None):
        """Estimate risk aversion coefficient delta."""
        if self.delta is not None:
            return self.delta

        if market_cap is not None:
            # Use market cap weighted portfolio
            w_market = market_cap / market_cap.sum()
            market_ret = (returns @ w_market).mean()
            market_var = (returns @ w_market).var()
            delta_est = market_ret / market_var
        else:
            # Use equal weighted portfolio as proxy
            w_eq = pd.Series(1/len(returns.columns), index=returns.columns)
            market_ret = (returns @ w_eq).mean()
            market_var = (returns @ w_eq).var()
            delta_est = market_ret / market_var

        return max(delta_est, 0.1)  # Ensure positive and reasonable

    def _fit_optimizer(self):
        """Fit the Black-Litterman optimizer."""
        # Create returns matrix
        returns = self.bw_daily_df_[self.returns_var].unstack(level='ticker')
        n_assets = returns.shape[1]

        # Validate BL inputs
        self._validate_bl_inputs(n_assets)

        # Create portfolio object for basic stats
        self.port_ = rp.Portfolio(returns=returns)

        # Prepare method parameters
        dict_mu = {}
        dict_cov = {}

        # EWMA conversions if needed
        if self.method_mu in ('ewma1', 'ewma2'):
            if self.ewma_mu_halflife is not None:
                dict_mu['d'] = self._halflife_to_d(self.ewma_mu_halflife)

        if self.method_cov in ('ewma1', 'ewma2'):
            if self.ewma_cov_halflife is not None:
                dict_cov['d'] = self._halflife_to_d(self.ewma_cov_halflife)

        # Calculate basic statistics
        self.port_.assets_stats(method_mu=self.method_mu, method_cov=self.method_cov,
                               dict_mu=dict_mu, dict_cov=dict_cov)

        # Store basic statistics
        sigma_hist = self.port_.cov.copy()
        mu_hist = self.port_.mu.squeeze().copy()

        # Estimate or use provided delta
        market_cap = None
        if self.market_cap is not None:
            market_cap = self.market_cap.reindex(returns.columns)

        self.delta_ = self._estimate_delta(returns, market_cap)

        # Calculate equilibrium portfolio weights
        if self.equilibrium and market_cap is not None:
            # Use market cap weights as equilibrium
            self.w_eq_ = market_cap / market_cap.sum()
        else:
            # Use reverse optimization: w_eq = (1/delta) * Sigma^-1 * mu_hist
            try:
                sigma_inv = pd.DataFrame(
                    np.linalg.inv(sigma_hist.values),
                    index=sigma_hist.index,
                    columns=sigma_hist.columns
                )
                self.w_eq_ = sigma_inv @ mu_hist / self.delta_
                self.w_eq_ = self.w_eq_ / self.w_eq_.sum()  # Normalize to sum to 1
            except np.linalg.LinAlgError:
                # If covariance is not invertible, use equal weights
                logger.warning("Covariance matrix not invertible, using equal weights")
                self.w_eq_ = pd.Series(1/n_assets, index=returns.columns)

        # Calculate equilibrium returns: Pi = delta * Sigma * w_eq
        self.Pi_ = self.delta_ * (sigma_hist @ self.w_eq_)

        # Apply Black-Litterman if views are provided
        if self.P is not None and self.Q is not None:
            self._apply_black_litterman(sigma_hist)
        else:
            # No views, use equilibrium returns and historical covariance
            self.mu_bl_ = self.Pi_.copy()
            self.sigma_bl_ = sigma_hist.copy()

        # Store final statistics
        self.mu_ = self.mu_bl_.copy()
        self.mu_.index.name = 'ticker'
        self.mu_.name = None

        self.sigma_ = self.sigma_bl_.copy()
        self.sigma_.index.name = 'ticker'
        self.sigma_.columns.name = 'ticker'

        # Assign to portfolio object for optimization
        self.port_.mu = self.mu_
        self.port_.cov = self.sigma_

        # Set portfolio constraints
        self.port_.sht = self.sht
        self.port_.budget = self.budget
        self.port_.budgetsht = self.budgetsht
        self.port_.uppersht = self.uppersht
        self.port_.upperlng = self.upperlng

        # Prepare optimization config
        opt_config = {
            'model': 'Classic',
            'rm': self.rm,
            'obj': self.obj,
            'rf': self.rf,
            'l': self.l,
            'hist': False
        }

        # Run optimization
        self.weights_ = self.port_.optimization(**opt_config)

        # Calculate derived statistics
        self.gross_exposure_ = self.weights_.abs().sum()
        self._calculate_in_sample_stats()

    def _apply_black_litterman(self, sigma_hist):
        """Apply Black-Litterman formula to combine prior and views."""
        # Align P matrix with asset ordering
        P = self.P.reindex(columns=sigma_hist.columns)

        # Convert Q to numpy array if it's a Series
        Q = np.array(self.Q).reshape(-1, 1)

        # Default Omega if not provided (proportional to view portfolio variance)
        if self.Omega is None:
            omega_diag = np.diag(P.values @ sigma_hist.values @ P.values.T)
            self.Omega = pd.DataFrame(
                np.diag(omega_diag),
                index=P.index,
                columns=P.index
            )

        # Align Omega with P
        Omega = self.Omega.reindex(index=P.index, columns=P.index)

        # Black-Litterman formulas
        try:
            # tau * Sigma
            tau_sigma = self.tau * sigma_hist.values

            # Sigma^-1
            sigma_inv = np.linalg.inv(sigma_hist.values)

            # P^T * Omega^-1 * P
            omega_inv = np.linalg.inv(Omega.values)
            PtOmegaP = P.values.T @ omega_inv @ P.values

            # P^T * Omega^-1 * Q
            PtOmegaQ = P.values.T @ omega_inv @ Q.flatten()

            # New covariance: Sigma_BL = [(tau*Sigma)^-1 + P^T*Omega^-1*P]^-1
            M1_inv = np.linalg.inv(tau_sigma) + PtOmegaP
            self.sigma_bl_ = pd.DataFrame(
                np.linalg.inv(M1_inv),
                index=sigma_hist.index,
                columns=sigma_hist.columns
            )

            # New mean: mu_BL = Sigma_BL * [(tau*Sigma)^-1*Pi + P^T*Omega^-1*Q]
            term1 = np.linalg.inv(tau_sigma) @ self.Pi_.values
            term2 = PtOmegaQ
            mu_bl_vals = self.sigma_bl_.values @ (term1 + term2)

            self.mu_bl_ = pd.Series(
                mu_bl_vals.flatten(),
                index=sigma_hist.index,
                name='mu_bl'
            )

            logger.info(f"Applied Black-Litterman with {P.shape[0]} views")

        except np.linalg.LinAlgError as e:
            logger.error(f"Black-Litterman calculation failed: {e}")
            # Fall back to equilibrium
            self.mu_bl_ = self.Pi_.copy()
            self.sigma_bl_ = sigma_hist.copy()

    def _predict_optimizer(self, X):
        """
        Black-Litterman prediction: w^T * r
        Similar to classic optimizer but using BL-adjusted parameters.
        """
        if isinstance(X, xr.Dataset):
            returns_var = getattr(self, 'returns_var', 'return')
            returns = X[returns_var].sel(ticker=self.weights_.index) * 0.01

            # Convert weights to xarray
            w = self.get_weights()
            w = xr.DataArray(
                w.values,
                dims=["ticker"],
                coords={"ticker": w.index},
            )

            predictions = xr.dot(returns, w, dim='ticker').values
            return predictions

        elif isinstance(X, pd.DataFrame):
            aligned_returns = X.reindex(columns=self.weights_.index)
            predictions = (aligned_returns @ self.weights_.values).values
            return predictions

        else:
            raise ValueError("X must be xr.Dataset or pd.DataFrame")

    def get_bl_stats(self):
        """Get Black-Litterman specific statistics."""
        check_is_fitted(self)

        stats = {
            'delta': self.delta_,
            'tau': self.tau,
            'equilibrium_used': self.equilibrium,
            'has_views': self.P is not None,
            'equilibrium_portfolio_return': (self.w_eq_.T @ self.Pi_) if self.w_eq_ is not None else None,
        }

        if self.P is not None:
            stats['num_views'] = self.P.shape[0]
            stats['views_matrix_shape'] = self.P.shape

        return stats

    def get_portfolio_stats(self, active_threshold=0.001):
        """Get extended portfolio statistics including BL-specific stats."""
        stats = super().get_portfolio_stats(active_threshold=active_threshold)

        # Add Black-Litterman specific stats
        bl_stats = self.get_bl_stats()
        stats.update(bl_stats)

        return stats

    def get_equilibrium_weights(self):
        """Get equilibrium portfolio weights."""
        check_is_fitted(self)
        return self.w_eq_.copy() if self.w_eq_ is not None else None

    def get_equilibrium_returns(self):
        """Get equilibrium (implied) returns."""
        check_is_fitted(self)
        return self.Pi_.copy() if self.Pi_ is not None else None


# Pipeline utilities and examples
class PortfolioPipeline:
    """Utility class for creating portfolio optimization pipelines."""

    @staticmethod
    def create_classic_pipeline(**optimizer_params):
        """Create a basic classic optimization pipeline."""
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ('optimizer', ClassicOptimizer(**optimizer_params))
        ])

    @staticmethod
    def create_factor_pipeline(**optimizer_params):
        """Create a factor model optimization pipeline."""
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ('optimizer', FactorModelOptimizer(**optimizer_params))
        ])

    @staticmethod
    def create_bl_pipeline(**optimizer_params):
        """Create a Black-Litterman optimization pipeline."""
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ('optimizer', BlackLittermanOptimizer(**optimizer_params))
        ])

    @staticmethod
    def create_ensemble_pipeline(optimizers_config):
        """Create an ensemble of optimizers with different configurations."""
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import VotingRegressor

        estimators = []
        for name, (optimizer_class, params) in optimizers_config.items():
            estimators.append((name, optimizer_class(**params)))

        return Pipeline([
            ('ensemble', VotingRegressor(estimators))
        ])


# Example usage functions
def create_bl_views_example():
    """Example function showing how to create Black-Litterman views."""

    # Example: 3 assets, 2 views
    assets = ['AAPL', 'MSFT', 'GOOGL']

    # View 1: AAPL will outperform MSFT by 2%
    # View 2: GOOGL will have absolute return of 8%
    P = pd.DataFrame([
        [1, -1, 0],    # AAPL - MSFT
        [0, 0, 1]      # GOOGL
    ], columns=assets, index=['View1', 'View2'])

    Q = pd.Series([0.02, 0.08], index=['View1', 'View2'])

    # Omega: uncertainty in views (higher = less confident)
    Omega = pd.DataFrame([
        [0.001, 0.0],
        [0.0, 0.002]
    ], index=['View1', 'View2'], columns=['View1', 'View2'])

    return P, Q, Omega
