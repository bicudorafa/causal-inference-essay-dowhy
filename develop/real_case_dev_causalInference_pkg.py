"""Importing Dependencies"""
# %%
# python basics dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')
#from pandas_profiling import ProfileReport
from statsmodels.stats.power import tt_ind_solve_power
from numpy.random import seed
from scipy.stats import ttest_ind
import statsmodels.api as sm
from causalinference import CausalModel #https://rugg2.github.io/Lalonde%20dataset%20-%20Causal%20Inference.html
## Extracting lalonde data: http://users.nber.org/~rdehejia/nswdata2.html
#%% rct data
rct_data = pd.read_stata('../data/raw/nsw_dw.dta')
#%% observational data
observational_data = pd.concat(
    [pd.read_stata('../data/raw/psid_controls.dta'), pd.read_stata('../data/raw/cps_controls.dta')],
    ignore_index=True
)
## preliminary analysis
# %%
treated = rct_data[rct_data.treat == 1]
synthetic_cps1 =pd.concat(
    [treated, observational_data[observational_data.data_id == 'CPS1']]
    ).assign(
        treat=lambda x: x.treat.astype(bool)
    )
#synthetic_psid = pd.concat([treated, observational_data[observational_data.data_id == 'PSID']])
# %%
# we use the CausalModel method from the causalinference package

causal = CausalModel(
    Y=synthetic_cps1['re78'].values, 
    D=synthetic_cps1['treat'].values, 
    X=synthetic_cps1[[col for col in synthetic_cps1.columns if col not in ('treat','re78','data_id')]].values)

causal.est_via_ols(adj=1)
# adj=1 corresponds to the simplicity of the model we entered
# This is called a "constant treatment effect"

print(causal.estimates)
# %%
print(causal.summary_stats)
# %%
#this function estimates the propensity score, so that propensity methods can be employed
causal.est_propensity_s()
print(causal.propensity)
# %%
#however, there is a procedure that tried to select an optimal cutoff value
causal.trim_s()
# %%
print(causal.summary_stats)
# %%
causal.stratify_s()
print(causal.strata)
# %%
# allowing several matches
causal.est_via_matching(bias_adj=True, matches=3)
print(causal.estimates)
# %%
causal.est_via_blocking()
print(causal.estimates)
# %%
