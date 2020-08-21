"""Importing Dependencies"""
# %%
# python basics dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')
from pandas_profiling import ProfileReport
# dowhy framework
#import dowhy
#import dowhy.api
#from dowhy import CausalModel
#import dowhy.datasets
#from dowhy.do_samplers.weighting_sampler import WeightingSampler
## auxiliar libraries for econml support
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LassoCV
#from sklearn.ensemble import GradientBoostingRegressor

## Extracting lalonde data: http://users.nber.org/~rdehejia/nswdata2.html
#%% rct data
rct_data = pd.read_stata('../data/raw/nsw.dta')
#%% observational data
observational_data = pd.concat(
    [pd.read_stata('../data/raw/psid_controls.dta'), pd.read_stata('../data/raw/cps_controls.dta')],
    ignore_index=True
)
## ppreliminary analysis
# %%
ProfileReport(rct_data)
# %%
ProfileReport(observational_data)
# %%
