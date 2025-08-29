import pandas as pd
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Assuming PortfolioOptimizer is in the utils directory and in the python path
from utils.PortfolioOptimizer import ClassicOptimizer
from utils.utils_simulate import EWMTransformer

def load_data(file_path="data/rm_demo_ds_20250627.nc"):
    """Loads and prepares the data."""
    # Load the dataset
    rm_demo_ds = xr.open_dataset(file_path)

    # Convert to DataFrame
    df = rm_demo_ds.to_dataframe()
    df = df.reset_index().set_index(['date', 'ticker'])
    
    # Get returns
    returns = df['return'].unstack() * 0.01
    
    # Create features (lagged returns) and target
    X = returns.shift(1).dropna()
    y = returns.reindex(X.index)
    
    return X, y

class PredictiveOptimizer(BaseEstimator, RegressorMixin):
    """
    A custom meta-estimator that first runs a predictor to get mu,
    then runs an optimizer with that mu.
    """
    def __init__(self, predictor, optimizer):
        self.predictor = predictor
        self.optimizer = optimizer

    def fit(self, X, y):
        """
        Fit the predictor and then the optimizer.
        """
        # 1. Fit the predictor
        self.predictor.fit(X, y)
        
        # 2. Predict mu for the entire dataset to be used in optimization
        # In a real backtest, this would be handled on a rolling basis.
        # For this demonstration, we'll use the last predicted value as the mu.
        predicted_mus = self.predictor.predict(X)
        last_mu = pd.Series(predicted_mus[-1], index=y.columns)
        
        # 3. Set the predicted mu on the optimizer
        self.optimizer.set_params(user_input_mu=last_mu, method_mu='custom')
        
        # 4. Fit the optimizer using the full history for covariance
        self.optimizer.fit(X)
        
        self.weights_ = self.optimizer.get_weights()
        
        return self

    def predict(self, X):
        """
        Predicts the portfolio return.
        Note: This is a simple example; a real implementation would
        predict mu for X and then calculate portfolio returns.
        For GridSearchCV, we mainly care about the score from 'fit'.
        """
        # For scoring purposes, we need to return something.
        # We'll return the expected portfolio return based on the fitted optimizer.
        if hasattr(self.optimizer, 'port_return_'):
            return np.full(len(X), self.optimizer.port_return_)
        return np.zeros(len(X))
        
    def get_params(self, deep=True):
        return {"predictor": self.predictor, "optimizer": self.optimizer}

    def set_params(self, **params):
        if "predictor" in params:
            self.predictor = params["predictor"]
        if "optimizer" in params:
            self.optimizer = params["optimizer"]
        
        predictor_params = {key.split('__')[1]: value for key, value in params.items() if key.startswith('predictor__')}
        if predictor_params:
            self.predictor.set_params(**predictor_params)
            
        return self

def main():
    """Main function to run the GridSearchCV demonstration."""
    print("Loading data...")
    X, y = load_data()

    # 1. Define the predictor pipeline
    predictor_pipeline = Pipeline([
        ('ewm', EWMTransformer()),
        ('model', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
    ])

    # 2. Define the optimizer
    optimizer = ClassicOptimizer(
        method_cov='hist', # Covariance from historical returns
        obj='Sharpe',
        sht=True # Allow shorting
    )

    # 3. Define the custom meta-estimator
    predictive_optimizer = PredictiveOptimizer(
        predictor=predictor_pipeline,
        optimizer=optimizer
    )

    # 4. Define the parameter grid for GridSearchCV
    param_grid = {
        'predictor__ewm__halflife': [10, 30],
        'predictor__model__estimator__n_estimators': [20, 50],
        'predictor__model__estimator__max_depth': [3, 5]
    }

    # Use TimeSeriesSplit for cross-validation on time series data
    tscv = TimeSeriesSplit(n_splits=3)

    # 5. Set up and run GridSearchCV
    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=predictive_optimizer,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error', # Example scoring
        n_jobs=-1
    )
    
    grid_search.fit(X, y)

    # 6. Print the results
    print("\n--- GridSearchCV Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    
    print("\n--- Detailed Results ---")
    results_df = pd.DataFrame(grid_search.cv_results_)
    print(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

    print("\n--- Optimized Weights with Best Estimator ---")
    best_estimator = grid_search.best_estimator_
    print(best_estimator.weights_)


if __name__ == "__main__":
    main()
