"""Importing Dependencies"""
# %%
# python basics dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')
from statsmodels.stats.power import tt_ind_solve_power
from numpy.random import seed
from scipy.stats import ttest_ind
import statsmodels.api as sm
import dowhy
import dowhy.api
from dowhy import CausalModel
# auxiliar libraries for econml support - https://github.com/microsoft/EconML/issues/284
import econml
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
## Extracting lalonde data: http://users.nber.org/~rdehejia/nswdata2.html
#%% rct data
rct_data = pd.read_stata('../data/raw/nsw_dw.dta')
#%% observational data
observational_data = pd.read_stata('../data/raw/cps_controls.dta')
#pd.concat(
#    [pd.read_stata('../data/raw/psid_controls.dta'), pd.read_stata('../data/raw/cps_controls.dta')],
#    ignore_index=True
#)
## preliminary analysis
# %%
#ProfileReport(rct_data)
# %%
#ProfileReport(observational_data)
# %%
rct_data.groupby('treat').agg({'mean', 'median', 'std'}).stack(1)
# %%
observational_data.groupby('data_id').agg({'mean', 'median', 'std'}).stack(1)#.pivot()
# %%
## Analysis strategy
#1 Talk a little bit about the original data, its distribution and the role of randomization
#  - La Londe main goal: all tecniques available by his time weren't capable of got similar results to the experimental design,
#    then claiming that experimental design was the only reasonable tool to infer treatment impact
#  - Try to simulate some of the proposed alternative frameworks used by La Londe (Fixed Effect,TWO-STAGE ESTIMATOR)
#      - Mas bem talvez, nem os caras do 2o paper o fizeram. Mais importante é mostrar como Dummy Nonexperimental estimation é Naivy
#2 Demonstrate how the treatment effect is scored by simple t test and an adjusted result by regression
#3 Present the external data and how it difers from the original one
# Simulates original La Londes exercice to demonstrate how to apply a simple OLS into new data generates biased results
#4 explain the tecniques are able to create a new control based on the causal inference methods
#  - Exercice proposed by Dehejia and Wahba: they claimed that most modern tecniques, such as propensity scores matching, 
#  were capable of generate better results
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
# %% why use OLS to XP data might improve its assessment: https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
rct_data_to_reg = rct_data.copy()
#rct_data_to_reg['age_sqr'] = rct_data_to_reg.age**2
Y = rct_data_to_reg['re78'].values
# exogenous variables used by lalonde:  age, age squared, years of schooling, high school dropout status, and race
X = rct_data_to_reg[[col for col in rct_data_to_reg.columns if col not in ('re78','data_id')]].values
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())
print(results.params)
# %%
treated = rct_data[rct_data.treat == 1]
for obs_data_group in observational_data.data_id.unique():
    to_reg = pd.concat([treated, observational_data[observational_data.data_id == obs_data_group]])
    Y = to_reg['re78'].values
    X = to_reg[[col for col in to_reg.columns if col not in ('re78','data_id')]].values
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    print(results.summary())
# %%
synthetic_cps1 =pd.concat(
    [treated, observational_data[observational_data.data_id == 'CPS1']]
    ).assign(
        treat=lambda x: x.treat.astype(bool)
    )
#synthetic_psid = pd.concat([treated, observational_data[observational_data.data_id == 'PSID']])
# %%
# With graph
model_cps1=CausalModel(
    data = synthetic_cps1
    , treatment='treat'
    , outcome='re78'
    , common_causes=[col for col in synthetic_cps1.columns if col not in ('treat','re78','data_id')]
)
# %%
model_cps1.view_model()
# %%
identified_estimand_cps1 = model_cps1.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand_cps1)
# %%
causal_estimate = model_cps1.estimate_effect(
    identified_estimand_cps1, method_name="backdoor.propensity_score_matching"#, confidence_intervals=True
)
print(causal_estimate)
# %%
dml_estimate = model_cps1.estimate_effect(
    identified_estimand_cps1, method_name="backdoor.econml.dml.DMLCateEstimator"
    , control_value = 0
    , treatment_value = 1
    #, target_units = lambda df: df["X0"]>1  # condition used for CATE
    , confidence_intervals=True
    , method_params={
        "init_params":{
             'model_y':GradientBoostingRegressor()
            , 'model_t': GradientBoostingRegressor()
            , "model_final":LassoCV()
            , 'featurizer':PolynomialFeatures(degree=1, include_bias=True)
        }
        , "fit_params":{}
    }
)
print(dml_estimate)
# %%
