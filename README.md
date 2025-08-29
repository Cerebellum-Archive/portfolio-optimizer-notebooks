# Portfolio Optimizer Framework

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drewtilley/portfolio-optimizer-notebooks/blob/main/notebooks/PortfolioOptimizer.ipynb)

A comprehensive scikit-learn compatible portfolio optimization framework featuring multiple optimization models, factor-based risk management, and flexible constraint handling. Built on top of Riskfolio-Lib with custom extensions for factor models and Black-Litterman optimization.

## Key Features

- **Scikit-learn Compatible**: Full integration with sklearn pipelines, cross-validation, and model selection
- **Multiple Optimization Models**: 
  - Classic Mean-Variance Optimization
  - Factor Model Optimization with custom factor structures
  - Black-Litterman Model with equilibrium returns and investor views
- **Flexible Risk Models**: Market, sector, and residual risk decomposition
- **Advanced Constraints**: Sector neutrality, long/short limits, leverage controls, custom exposure bounds
- **Production Ready**: Robust error handling, comprehensive logging, and extensive validation

---

## Architecture Overview

### Core Components

**BasePortfolioOptimizer**: Abstract base class providing the sklearn-compatible interface with common functionality for validation, fitting, prediction, and statistics calculation.

**ClassicOptimizer**: Traditional mean-variance optimization with historical or EWMA-based parameter estimation. Supports custom expected returns and covariance matrices.

**FactorModelOptimizer**: Factor-based optimization using market and sector factor exposure. Computes expected returns and covariance through factor decomposition (μ = Bf, Σ = BFB' + D).

**BlackLittermanOptimizer**: Bayesian approach combining market equilibrium returns with investor views. Supports custom view matrices and uncertainty specifications.

**PortfolioPipeline**: Utility class for creating sklearn pipelines and ensemble models.

### Data Requirements

The framework expects input data as either:
- **xarray.Dataset**: Multi-dimensional labeled arrays with date/ticker coordinates
- **pandas.DataFrame**: Traditional tabular format with MultiIndex (date, ticker)

Required columns vary by optimizer:
- **Classic**: `return` (asset returns)
- **Factor Model**: `market_beta`, `sector_beta`, `bw_sector_name`, `residual_variance`, factor returns
- **Black-Litterman**: `return` plus optional market capitalization data

---

## Quick Start

### Basic Usage

```python
import pandas as pd
import numpy as np
from PortfolioOptimizer import ClassicOptimizer, FactorModelOptimizer

# Load your returns data (date × ticker format)
returns_data = load_your_data()

# Classic Mean-Variance Optimization
classic_opt = ClassicOptimizer(
    method_mu='ewma1',
    method_cov='ewma1', 
    ewma_mu_halflife=30,
    ewma_cov_halflife=60,
    obj='Sharpe',
    sht=True  # Allow short positions
)

# Fit and get optimal weights
classic_opt.fit(returns_data)
weights = classic_opt.get_weights()
stats = classic_opt.get_portfolio_stats()

# Out-of-sample prediction
future_returns = classic_opt.predict(test_data)
```

### Factor Model Optimization

```python
# Factor model with market and sector factors
factor_opt = FactorModelOptimizer(
    method_f='ewma1',
    method_F='ewma1', 
    halflife=30,
    obj='Sharpe',
    sht=True
)

# Requires factor exposure data
factor_opt.fit(factor_data)  # Must include beta columns and factor returns

# Get factor exposures and return decomposition
exposures = factor_opt.get_factor_exposures()
decomp = factor_opt.decompose_returns()
```

### Black-Litterman with Views

```python
from PortfolioOptimizer import BlackLittermanOptimizer, create_bl_views_example

# Create investor views
P, Q, Omega = create_bl_views_example()  # Example view matrices

bl_opt = BlackLittermanOptimizer(
    method_mu='hist',
    method_cov='hist',
    P=P,           # Picking matrix (which assets to express views on)
    Q=Q,           # View returns
    Omega=Omega,   # View uncertainty
    tau=0.1,       # Prior uncertainty scaling
    equilibrium=True
)

bl_opt.fit(returns_data)
```

### Sklearn Pipeline Integration

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from PortfolioOptimizer import PortfolioPipeline

# Create pipeline
pipeline = PortfolioPipeline.create_factor_pipeline(
    halflife=30,
    obj='Sharpe',
    sht=True
)

# Grid search over parameters
param_grid = {
    'optimizer__halflife': [15, 30, 60],
    'optimizer__obj': ['Sharpe', 'MinRisk'],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
```

---

## Repository Structure

```
portfolio-optimizer-notebooks/
│
├── notebooks/                   # Interactive demonstrations
│   └── PortfolioOptimizer.ipynb # Main demo notebook (being rewritten)
│
├── utils/                       # Core framework
│   ├── PortfolioOptimizer.py    # Main optimization classes
│   └── logging_config.py        # Logging utilities
│
├── data/                        # Sample datasets
├── requirements.txt             # Dependencies
├── LICENSE
└── README.md
```

---

## Advanced Features

### Custom Risk Models

```python
# Use custom expected returns and covariance
custom_mu = pd.Series([0.10, 0.08, 0.12], index=['AAPL', 'MSFT', 'GOOGL'])
custom_sigma = pd.DataFrame(...)  # Custom covariance matrix

optimizer = ClassicOptimizer(
    user_input_mu=custom_mu,
    user_input_cov=custom_sigma
)
```

### Factor Model with Custom Inputs

```python
# Skip factor computation, use pre-computed mu/sigma
factor_opt = FactorModelOptimizer(
    use_custom_inputs=True,
    user_input_mu=factor_based_mu,
    user_input_cov=factor_based_sigma
)
```

### Constraint Management

```python
optimizer = ClassicOptimizer(
    sht=True,           # Enable short selling
    budget=1.0,         # 100% invested
    budgetsht=0.3,      # Max 30% short exposure
    upperlng=0.1,       # Max 10% in any long position
    uppersht=0.05       # Max 5% in any short position
)
```

---

## Getting Started

### Option 1: Google Colab (Recommended)

Click the badge above or use this direct link:
[Open Portfolio Optimizer in Colab](https://colab.research.google.com/github/drewtilley/portfolio-optimizer-notebooks/blob/main/notebooks/PortfolioOptimizer.ipynb)

All dependencies install automatically from `requirements.txt`.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/drewtilley/portfolio-optimizer-notebooks.git
cd portfolio-optimizer-notebooks

# Create environment
conda create -n portfolio-opt python=3.11
conda activate portfolio-opt

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/PortfolioOptimizer.ipynb
```

---

## Data Access

The demo uses sample datasets from a public Google Cloud bucket (no credentials required). The framework is designed to work with:

- **Blue Water Macro datasets**: Contact [support@riskmodels.net](mailto:support@riskmodels.net) for proprietary data access
- **Custom datasets**: Modify data loading in notebooks for local files (CSV, Parquet, Zarr)
- **Standard formats**: Any pandas/xarray compatible data source

---

## API Reference

### Core Methods

All optimizers implement these sklearn-compatible methods:

- `fit(X, y=None)`: Fit the optimizer to training data
- `predict(X)`: Generate out-of-sample portfolio return predictions  
- `transform(X)`: Return optimal portfolio weights
- `get_params()` / `set_params()`: Parameter management for sklearn compatibility

### Optimizer-Specific Methods

- `get_weights()`: Portfolio weights as pandas Series
- `get_portfolio_stats()`: Comprehensive portfolio statistics
- `get_factor_exposures()`: Factor exposures (FactorModelOptimizer)
- `decompose_returns()`: Return attribution (FactorModelOptimizer)
- `get_bl_stats()`: Black-Litterman statistics (BlackLittermanOptimizer)

---

## Dependencies

**Core Requirements:**
- pandas >= 1.5.0
- numpy >= 1.20.0  
- scikit-learn >= 1.2.0
- xarray >= 2022.6.0
- riskfolio-lib >= 4.0.0

**Data & Visualization:**
- zarr, gcsfs (cloud data access)
- matplotlib, seaborn, plotly (visualization)

**Python:** 3.9+ (tested on 3.11, Colab compatible)

---

## Contributing

We welcome contributions! Areas of interest:
- Additional optimization models (robust optimization, distributional robust optimization)
- Enhanced constraint handling (turnover, tracking error)
- Performance improvements and vectorization
- Extended factor model support

---

## License

MIT License - see LICENSE file for details.

---

## Contact

For questions about Blue Water Macro data or commercial licensing:
**[support@riskmodels.net](mailto:support@riskmodels.net)**
