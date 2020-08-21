"""Importing Dependencies"""
# %%
# python basics dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')
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
#%% lalonde data: http://users.nber.org/~rdehejia/nswdata2.html
rct_data = pd.read_stata('../data/raw/nsw.dta')
rct_data.head()
# %%
observational_data_psid = pd.read_stata('../data/raw/psid_controls.dta')
observational_data_psid.head()
# %%
observational_data_cps = pd.read_stata('../data/raw/cps_controls.dta')
observational_data_cps.head()
# %%
