# Portfolio Optimizer Notebooks

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drewtilley/portfolio-optimizer-notebooks/blob/main/notebooks/PortfolioOptimizer.ipynb)

This repository contains an interactive demonstration of a factor-based portfolio optimization engine using Blue Water Macro data, Riskfolio-Lib, and custom factor covariance inputs.

It includes a Jupyter/Colab-friendly notebook along with supporting utilities to showcase:

- Sharpe-ratio maximizing portfolios
- Efficient frontier visualization
- Sector exposure control using ETFs
- Custom alpha and covariance matrix support (e.g., from factor models)

## Key Features

- **Custom Risk Models**: Use market, sector, and residual decomposition for returns.
- **Flexible Constraints**: Apply sector neutrality, long/short conditions, and le limits.
- **Factor-based Optimization**: Build portfolios on top of factor betas and factor covariance matrices.
- **Visualization Tools**: Plot optimal weights and frontier with Plotly and Matplotlib.

---

## Repository Structure

    portfolio-optimizer-notebooks/
    │
    ├── notebooks/ # Main demo notebook(s)
    │ └── PortfolioOptimizer.ipynb
    │
    ├── utils/ # Supporting Python classes and helper functions
    │ └── PortfolioOptimizer.py
    │
    ├── data/ # Optional placeholder for local data
    ├── requirements.txt # Minimal dependencies (Colab-compatible)
    ├── LICENSE
    └── README.md
    

---

## Getting Started

### Option 1: Run in Google Colab (recommended)

Click the badge above or open this link directly:  
[Open Portfolio Optimizer in Colab](https://colab.research.google.com/github/drewtilley/portfolio-optimizer-notebooks/blob/main/notebooks/PortfolioOptimizer.ipynb)

> No setup needed — all dependencies install from `requirements.txt` during the first cell.

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/drewtilley/portfolio-optimizer-notebooks.git
cd portfolio-optimizer-notebooks

# Create a virtual environment (optional)
conda create -n optimizer-env python=3.11
conda activate optimizer-env

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

---

## Data Access

This demo reads a small public sample dataset stored in a public Google Cloud bucket via gcsfs.
The notebook handles this automatically — no credentials required.

If you'd like to utilize Blue Water Macro proprietary datasets contact us at **[support@riskmodels.net](mailto:support@riskmodels.net)**

If you'd like to use local Zarr files, csv, or other datasets modify the data_path inside the notebook accordingly.

---

## Dependencies

    xarray
    zarr
    gcsfs
    riskfolio-lib
    matplotlib, seaborn, plotly

Fully compatible with Python 3.11 and Google Colab

---

## License

This project is licensed under the MIT License.
See LICENSE for details.

---
