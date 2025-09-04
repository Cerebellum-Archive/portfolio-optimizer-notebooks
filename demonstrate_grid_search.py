import pandas as pd
import numpy as np
import xarray as xr
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer

from utils.PortfolioOptimizer import ClassicOptimizer

def load_data(file_path="data/rm_demo_ds_20250627.nc"):
    """Loads and prepares the data into a returns DataFrame."""
    ds = xr.open_dataset(file_path)

    # Convert xarray dataset to DataFrame
    df = ds.to_dataframe().reset_index()
    df.set_index(["date", "ticker"], inplace=True)
    df.sort_index(level=['date', 'ticker'], inplace=True)

    return df.dropna()

def sharpe_scorer(estimator, X, y=None):
    """
    Custom scorer that returns the in-sample Sharpe ratio
    calculated by the optimizer after it has been fitted by GridSearchCV.
    """
    # The `fit` method is called by GridSearchCV before the scorer.
    # We just need to access the resulting 'sharpe_' attribute.
    return getattr(estimator, 'sharpe_', -np.inf)

def main():
    """Main function to run the GridSearchCV demonstration on the optimizer."""
    print("Loading data...")
    returns_df = load_data()

    # 1. Define the optimizer we want to tune
    optimizer = ClassicOptimizer()

    # 2. Define the parameter grid for the optimizer's parameters
    param_grid = {
        'method_mu': ['ewma1'],
        'method_cov': ['ewma1'],
        'ewma_mu_halflife': [5, 10, 30],
        'ewma_cov_halflife': [5, 10, 30]
    }

    # 3. Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # 4. Set up and run GridSearchCV
    print("Running GridSearchCV on optimizer parameters...")
    grid_search = GridSearchCV(
        estimator=optimizer,
        param_grid=param_grid,
        cv=tscv,
        scoring=sharpe_scorer,
        error_score='raise'
    )

    # GridSearchCV's fit method will pass slices of the returns_df to the optimizer's fit method
    grid_search.fit(returns_df)

    # 5. Print the results
    print("\n--- GridSearchCV Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best score (in-sample Sharpe): {grid_search.best_score_}")

    print("\n--- Detailed Results ---")
    results_df = pd.DataFrame(grid_search.cv_results_)
    print(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

    print("\n--- Optimized Weights with Best Estimator ---")
    # The best_estimator is already fitted on the last and largest training split.
    # We can inspect its weights directly.
    best_estimator = grid_search.best_estimator_
    print(best_estimator.get_weights())


if __name__ == "__main__":
    main()
