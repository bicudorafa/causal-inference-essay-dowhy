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
#ProfileReport(rct_data)
# %%
#ProfileReport(observational_data)
# %%
rct_data[[col for col in rct_data.columns if col != 're78']].groupby('treat').mean()
# %%
## Analysis strategy
#1 Talk a little bit about the original data, its distribution and the role of randomization
#2 Demonstrate how the treatment effect is scored by simple mean difference (and maybe by linear regression)
#3 Present the external data and how it difers from the original one
#4 explain the tecniques are able to create a new control based on the causal inference methods
#5 Show rhe results and conclusion