"""Importing Dependencies"""
# %%
# python basics dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')
#from pandas_profiling import ProfileReport
from statsmodels.stats.power import tt_ind_solve_power
from numpy.random import seed
from scipy.stats import ttest_ind
import statsmodels.api as sm
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
#2 Demonstrate how the treatment effect is scored by simple t test
#3 Present the external data and how it difers from the original one
# Simulates original La Londes exercice to demonstrate how to apply a simple OLS into new data generates biased results
#4 explain the tecniques are able to create a new control based on the causal inference methods
#5 Show rhe results and conclusion
# %%
def ttest(control, treatment, alpha=0.05):
    """calculates the ttest pvalue, the percentage lift and the test power based on a pre set alpha
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.power.tt_ind_solve_power.html"""
    # random seed to ensure results reproducibility 
    seed(666)
    # p_value
    stat, p = ttest_ind(a=treatment, b=control, nan_policy='omit')
    # mean lift
    mean_diff=np.nanmean(treatment)-np.nanmean(control)
    lift=(mean_diff/np.nanmean(control))*100.0
    # power parameters
    std_diff=np.sqrt(np.nanvar(treatment)+np.nanvar(control))
    effect_size = mean_diff / std_diff
    treatment_size=(~np.isnan(treatment)).sum()
    size_ratio=(~np.isnan(control)).sum()/treatment_size
    power=tt_ind_solve_power(
        effect_size=effect_size, alpha=alpha, power=None
        ,ratio=size_ratio, alternative='two-sided',nobs1=treatment_size
)
    return p,lift,power
# %%
ttest(rct_data[rct_data.treat == 0]['re78'], rct_data[rct_data.treat == 1]['re78'])
# %%
Y = rct_data['re78'].values
X = rct_data[[col for col in rct_data.columns if col not in ('re78','data_id')]].values
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
# %%
