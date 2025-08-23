import unittest
import numpy as np
import pandas as pd
import xarray as xr
import sys


from utils.PortfolioOptimizer import ClassicOptimizer, FactorModelOptimizer, BlackLittermanOptimizer
from sklearn.pipeline import Pipeline


class TestPortfolioOptimizerBase(unittest.TestCase):
    """Base test class with common setup and utilities."""

    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests."""
        try:
            cls.ds = xr.load_dataset('./data/rm_demo_ds_20250627.nc')
            cls.ds_train = cls.ds.sel(date=slice("2024-06-27", "2025-06-26"))
            cls.ds_predict = cls.ds.sel(date="2025-06-27")
        except FileNotFoundError:
            # Create mock data if file doesn't exist
            cls.ds_train, cls.ds_predict = cls._create_mock_data()

    @classmethod
    def _create_mock_data(cls):
        """Create mock xarray dataset for testing."""
        dates = pd.date_range('2024-06-27', '2025-06-26', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                   'META', 'NVDA', 'NFLX', 'CRM', 'ADBE']

        # Create random returns data
        np.random.seed(42)  # For reproducible tests
        returns = np.random.normal(0.05, 0.15, (len(dates), len(tickers)))

        # Create factor model data
        market_beta = np.random.uniform(0.8, 1.2, len(tickers))
        sector_beta = np.random.uniform(0.5, 1.5, len(tickers))
        sectors = ['Tech', 'Finance', 'Healthcare']
        bw_sector_name = np.random.choice(sectors, len(tickers))
        residual_variance = np.random.uniform(0.01, 0.05, len(tickers))

        # Create factor returns
        market_factor_return = np.random.normal(0.03, 0.10, len(dates))
        sector_factor_return = np.random.normal(0.02, 0.08, len(dates))
        residual_return = np.random.normal(0.0, 0.05, (len(dates), len(tickers)))

        # Build dataset
        data_vars = {
            'return': (['date', 'ticker'], returns),
            'market_beta': (['date', 'ticker'],
                          np.tile(market_beta, (len(dates), 1))),
            'sector_beta': (['date', 'ticker'],
                          np.tile(sector_beta, (len(dates), 1))),
            'bw_sector_name': (['date', 'ticker'],
                             np.tile(bw_sector_name, (len(dates), 1))),
            'residual_variance': (['date', 'ticker'],
                                np.tile(residual_variance, (len(dates), 1))),
            'market_factor_return': (['date'], market_factor_return),
            'sector_factor_return': (['date'], sector_factor_return),
            'residual_return': (['date', 'ticker'], residual_return)
        }

        coords = {
            'date': dates,
            'ticker': tickers
        }

        ds = xr.Dataset(data_vars, coords=coords)
        ds_train = ds.sel(date=slice("2024-06-27", "2025-06-25"))
        ds_predict = ds.sel(date="2025-06-26")

        return ds_train, ds_predict


class TestClassicOptimizer(TestPortfolioOptimizerBase):
    """Test cases for ClassicOptimizer."""

    def test_basic_initialization(self):
        """Test basic optimizer initialization."""
        optimizer = ClassicOptimizer()
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.method_mu, 'hist')
        self.assertEqual(optimizer.method_cov, 'hist')
        self.assertFalse(optimizer.sht)

    def test_fit_basic(self):
        """Test basic fit functionality."""
        optimizer = ClassicOptimizer()
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        self.assertIsNotNone(optimizer.mu_)
        self.assertIsNotNone(optimizer.sigma_)
        self.assertTrue(hasattr(optimizer, 'port_return_'))
        self.assertTrue(hasattr(optimizer, 'port_vol_'))
        self.assertTrue(hasattr(optimizer, 'sharpe_'))

    def test_ewma1_configuration(self):
        """Test EWMA1 method configuration."""
        config = {
            "method_mu": "ewma1",
            "method_cov": "ewma1",
            "ewma_mu_halflife": None,
            "ewma_cov_halflife": None,
            "sht": True,
            "budget": 1.0,
            "budgetsht": 0.2,
            "uppersht": 0.2,
            "upperlng": 1.0
        }

        optimizer = ClassicOptimizer(**config)
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        # Check that shorts are allowed
        self.assertTrue((optimizer.weights_ < 0).any().any())
        # Check budget constraints
        self.assertAlmostEqual(optimizer.weights_.sum().iloc[0], 1.0, places=3)

    def test_ewma2_with_halflife(self):
        """Test EWMA2 method with custom halflife."""
        config = {
            "method_mu": "ewma2",
            "method_cov": "ewma2",
            "ewma_mu_halflife": 77,
            "ewma_cov_halflife": 77,
            "sht": True
        }

        optimizer = ClassicOptimizer(**config)
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        self.assertEqual(optimizer.ewma_mu_halflife, 77)
        self.assertEqual(optimizer.ewma_cov_halflife, 77)

    def test_mixed_methods(self):
        """Test mixed mu and covariance methods."""
        config = {
            "method_mu": "ewma2",
            "method_cov": "hist",
            "sht": True
        }

        optimizer = ClassicOptimizer(**config)
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        self.assertEqual(optimizer.method_mu, "ewma2")
        self.assertEqual(optimizer.method_cov, "hist")

    def test_custom_mu_input(self):
        """Test custom mu input."""
        # First fit to get reference data structure
        ref_optimizer = ClassicOptimizer()
        ref_optimizer.fit(self.ds_train)

        # Create custom mu
        mu_in = pd.Series(
            np.ones(len(ref_optimizer.mu_)) * ref_optimizer.mu_.mean(),
            index=ref_optimizer.mu_.index
        )

        config = {
            "method_mu": "custom",
            "method_cov": "hist",
            "user_input_mu": mu_in,
            "sht": True
        }

        optimizer = ClassicOptimizer(**config)
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        # Check that custom mu was used
        np.testing.assert_array_almost_equal(
            optimizer.mu_.values, mu_in.values, decimal=6
        )

    def test_custom_mu_and_sigma_input(self):
        """Test custom mu and sigma input."""
        # Get reference data
        ref_optimizer = ClassicOptimizer()
        ref_optimizer.fit(self.ds_train)

        mu_in = pd.Series(
            np.ones(len(ref_optimizer.mu_)) * ref_optimizer.mu_.mean(),
            index=ref_optimizer.mu_.index
        )
        sigma_in = ref_optimizer.sigma_.copy()

        config = {
            "method_mu": "custom",
            "method_cov": "custom",
            "user_input_mu": mu_in,
            "user_input_cov": sigma_in,
            "sht": True
        }

        optimizer = ClassicOptimizer(**config)
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        np.testing.assert_array_almost_equal(
            optimizer.mu_.values, mu_in.values, decimal=6
        )
        np.testing.assert_array_almost_equal(
            optimizer.sigma_.values, sigma_in.values, decimal=6
        )

    def test_predict_functionality(self):
        """Test prediction functionality."""
        optimizer = ClassicOptimizer(sht=True)
        optimizer.fit(self.ds_train)

        # Test prediction
        predictions = optimizer.predict(self.ds_predict)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 1)  # Single date prediction

    def test_get_weights(self):
        """Test weight retrieval."""
        optimizer = ClassicOptimizer()
        optimizer.fit(self.ds_train)

        weights = optimizer.get_weights()
        self.assertIsInstance(weights, pd.Series)
        self.assertEqual(len(weights), len(optimizer.tickers_))

    def test_get_portfolio_stats(self):
        """Test portfolio statistics."""
        optimizer = ClassicOptimizer(sht=True)
        optimizer.fit(self.ds_train)

        stats = optimizer.get_portfolio_stats()
        required_keys = [
            'portfolio_return', 'portfolio_volatility', 'sharpe_ratio',
            'gross_exposure', 'num_assets', 'long_exposure',
            'short_exposure', 'net_exposure'
        ]

        for key in required_keys:
            self.assertIn(key, stats)
            self.assertIsNotNone(stats[key])

    def test_sklearn_pipeline_integration(self):
        """Test sklearn pipeline integration."""
        pipe = Pipeline([('optimizer', ClassicOptimizer())])

        # Test fit and predict through pipeline
        pipe.fit(self.ds_train)
        predictions = pipe.predict(self.ds_predict)

        self.assertIsNotNone(predictions)


class TestFactorModelOptimizer(TestPortfolioOptimizerBase):
    """Test cases for FactorModelOptimizer."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        optimizer = FactorModelOptimizer()
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.method_f, 'ewma1')
        self.assertEqual(optimizer.method_F, 'ewma1')
        self.assertEqual(optimizer.halflife, 30)
        self.assertTrue(optimizer.sht)

    def test_factor_model_fit(self):
        """Test factor model fitting."""
        optimizer = FactorModelOptimizer(halflife=30)
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        self.assertIsNotNone(optimizer.mu_)
        self.assertIsNotNone(optimizer.sigma_)

        # Factor-specific attributes
        if not optimizer.use_custom_inputs:
            self.assertIsNotNone(optimizer.B_)  # Factor loadings
            self.assertIsNotNone(optimizer.F_)  # Factor covariance
            self.assertIsNotNone(optimizer.D_)  # Residual variance
            self.assertIsNotNone(optimizer.f_)  # Factor returns

    def test_custom_inputs_mode(self):
        """Test factor model with custom inputs."""
        # Get reference data
        ref_optimizer = FactorModelOptimizer()
        ref_optimizer.fit(self.ds_train)

        mu_in = ref_optimizer.mu_.copy()
        sigma_in = ref_optimizer.sigma_.copy()

        optimizer = FactorModelOptimizer(
            use_custom_inputs=True,
            user_input_mu=mu_in,
            user_input_cov=sigma_in
        )
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        self.assertTrue(optimizer.use_custom_inputs)
        # Factor attributes should be None when using custom inputs
        self.assertIsNone(optimizer.B_)

    def test_factor_exposures(self):
        """Test factor exposure calculation."""
        optimizer = FactorModelOptimizer()
        optimizer.fit(self.ds_train)

        exposures = optimizer.get_factor_exposures()
        if not optimizer.use_custom_inputs:
            self.assertIsNotNone(exposures)
            self.assertIsInstance(exposures, pd.Series)

    def test_return_decomposition(self):
        """Test return decomposition."""
        optimizer = FactorModelOptimizer()
        optimizer.fit(self.ds_train)

        decomp = optimizer.decompose_returns()
        self.assertIsInstance(decomp, dict)

        expected_keys = [
            'factor_contribution', 'residual_contribution',
            'total_expected_return', 'portfolio_expected_return'
        ]
        for key in expected_keys:
            self.assertIn(key, decomp)

    def test_factor_prediction(self):
        """Test factor model prediction."""
        optimizer = FactorModelOptimizer()
        optimizer.fit(self.ds_train)

        predictions = optimizer.predict(self.ds_predict)
        self.assertIsNotNone(predictions)


class TestBlackLittermanOptimizer(TestPortfolioOptimizerBase):
    """Test cases for BlackLittermanOptimizer."""

    def setUp(self):
        """Set up Black-Litterman views for testing."""
        # Create sample views
        view_tickers = self.ds_train.ticker.values[:3]  # Use first 3 tickers
        tickers = self.ds_train.ticker.values

        # Compact view definitions (just the relevant tickers)
        compact_views = {
            "View1": {view_tickers[0]:1, view_tickers[1]:-1}, # Ticker1 outperforms Ticker2
            "View2": {view_tickers[2]:1},                     # Absolute view on Ticker3
        }

        Q_values = {"View1": 0.02, "View2": 0.08}
        Omega_diag = {"View1": 0.001, "View2": 0.002}

        # Expand compact views to full P matrix
        P = pd.DataFrame(0, index=compact_views.keys(), columns=tickers)
        for view, weights in compact_views.items():
            for asset, weight in weights.items():
                if asset in P.columns:
                    P.loc[view, asset] = weight

        # Build Q and Omega
        Q = pd.Series(Q_values)
        Omega = pd.DataFrame(0, index=Q.index, columns=Q.index, dtype=float)
        for view, var in Omega_diag.items():
            Omega.loc[view, view] = var

        self.P, self.Q, self.Omega = P, Q, Omega

    def test_basic_initialization(self):
        """Test BL optimizer initialization."""
        optimizer = BlackLittermanOptimizer()
        self.assertIsNotNone(optimizer)
        self.assertTrue(optimizer.equilibrium)
        self.assertEqual(optimizer.tau, 1.0)

    def test_bl_without_views(self):
        """Test BL optimizer without views (equilibrium only)."""
        optimizer = BlackLittermanOptimizer()
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        self.assertIsNotNone(optimizer.Pi_)  # Equilibrium returns
        self.assertIsNotNone(optimizer.delta_)  # Risk aversion

    def test_bl_with_views(self):
        """Test BL optimizer with views."""
        optimizer = BlackLittermanOptimizer(
            P=self.P,
            Q=self.Q,
            Omega=self.Omega,
            tau=0.5
        )
        optimizer.fit(self.ds_train)

        self.assertIsNotNone(optimizer.weights_)
        self.assertIsNotNone(optimizer.mu_bl_)  # BL adjusted returns
        self.assertIsNotNone(optimizer.sigma_bl_)  # BL adjusted covariance

    def test_bl_stats(self):
        """Test BL-specific statistics."""
        optimizer = BlackLittermanOptimizer(
            P=self.P,
            Q=self.Q,
            Omega=self.Omega
        )
        optimizer.fit(self.ds_train)

        stats = optimizer.get_bl_stats()
        expected_keys = [
            'delta', 'tau', 'equilibrium_used', 'has_views',
            'num_views', 'views_matrix_shape'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

        self.assertTrue(stats['has_views'])
        self.assertEqual(stats['num_views'], 2)

    def test_equilibrium_weights(self):
        """Test equilibrium weight calculation."""
        optimizer = BlackLittermanOptimizer()
        optimizer.fit(self.ds_train)

        eq_weights = optimizer.get_equilibrium_weights()
        eq_returns = optimizer.get_equilibrium_returns()

        self.assertIsNotNone(eq_weights)
        self.assertIsNotNone(eq_returns)
        self.assertAlmostEqual(eq_weights.sum(), 1.0, places=3)


class TestInputValidation(TestPortfolioOptimizerBase):
    """Test input validation and error handling."""

    def test_invalid_input_type(self):
        """Test invalid input type handling."""
        optimizer = ClassicOptimizer()

        with self.assertRaises(ValueError):
            optimizer.fit("invalid_input")

    def test_nan_input_handling(self):
        """Test NaN input handling."""
        # Create dataset with NaNs
        ds_with_nans = self.ds_train.copy()
        ds_with_nans['return'][0, 0] = np.nan

        optimizer = ClassicOptimizer()

        with self.assertRaises(ValueError):
            optimizer.fit(ds_with_nans)

    def test_mismatched_custom_inputs(self):
        """Test mismatched custom input dimensions."""
        optimizer = ClassicOptimizer()
        optimizer.fit(self.ds_train)

        # Wrong size mu
        wrong_mu = pd.Series([1, 2, 3], index=['A', 'B', 'C'])

        config = {
            "method_mu": "custom",
            "user_input_mu": wrong_mu
        }

        optimizer_bad = ClassicOptimizer(**config)
        with self.assertRaises(ValueError):
            optimizer_bad.fit(self.ds_train)

    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        optimizer = ClassicOptimizer()

        with self.assertRaises(Exception):  # sklearn raises NotFittedError
            optimizer.predict(self.ds_predict)


class TestPerformanceAndEdgeCases(TestPortfolioOptimizerBase):
    """Test performance and edge cases."""

    def test_single_asset_case(self):
        """Test optimization with single asset."""
        single_asset_ds = self.ds_train.sel(ticker=self.ds_train.ticker.values[0])

        optimizer = ClassicOptimizer()
        optimizer.fit(single_asset_ds)

        self.assertIsNotNone(optimizer.weights_)
        self.assertEqual(len(optimizer.weights_), 1)
        self.assertAlmostEqual(optimizer.weights_.iloc[0, 0], 1.0, places=3)

    def test_very_short_time_series(self):
        """Test with very short time series."""
        short_ds = self.ds_train.sel(date=self.ds_train.date.values[:5])

        optimizer = ClassicOptimizer()
        # This might fail or succeed depending on implementation
        # Just test that it doesn't crash unexpectedly
        try:
            optimizer.fit(short_ds)
        except Exception as e:
            # Log the exception but don't fail the test
            print(f"Short time series test raised: {e}")

    def test_extreme_risk_aversion(self):
        """Test with extreme risk aversion settings."""
        config = {
            "obj": "Utility",
            "l": 100.0  # Very high risk aversion
        }

        optimizer = ClassicOptimizer(**config)
        optimizer.fit(self.ds_train)

        # Should produce very conservative weights
        self.assertIsNotNone(optimizer.weights_)


def create_test_suite():
    """Create a test suite with all test cases."""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestClassicOptimizer,
        TestFactorModelOptimizer,
        TestBlackLittermanOptimizer,
        TestInputValidation,
        TestPerformanceAndEdgeCases
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    return test_suite


if __name__ == '__main__':
    # Run tests with different verbosity levels
    import argparse

    parser = argparse.ArgumentParser(description='Run portfolio optimizer tests')
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        help='Increase test verbosity')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Run only tests matching pattern')

    args = parser.parse_args()

    if args.pattern:
        # Run specific test pattern
        suite = unittest.TestLoader().discover('.', pattern=f'*{args.pattern}*')
    else:
        # Run all tests
        suite = create_test_suite()

    runner = unittest.TextTestRunner(verbosity=args.verbose)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
