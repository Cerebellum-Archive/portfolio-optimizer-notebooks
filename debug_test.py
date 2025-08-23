import sys
sys.path.append('/home/drew/projects/bwmacro/portfolio-optimizer-notebooks/utils')

from PortfolioOptimizer_v2 import ClassicOptimizer as COptv2
from PortfolioOptimizer_v2 import FactorModelOptimizer as FMOptv2

import xarray as xr
import pandas as pd
import numpy as np

def main():
    ds  = xr.load_dataset('./data/rm_demo_ds_20250627.nc')
    # Training data
    ds_train = ds.sel(date=slice("2024-06-27", "2025-06-26"))
    # Predict data
    ds_predict = ds.sel(date="2025-06-27")

    fm_optv2 = FMOptv2()
    #cl_optv2 = COptv2()

    fm_optv2.fit(ds_train)
    #cl_optv2.fit(ds_train)

    result = fm_optv2.predict(ds_predict)
    print(result)


if __name__ == "__main__":
    main()
