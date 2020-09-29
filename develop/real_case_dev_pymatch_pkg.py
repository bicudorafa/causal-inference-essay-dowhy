"""Importing Dependencies"""
# %%
# python basics dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib as plt
import warnings
warnings.filterwarnings('ignore')
from pymatch.Matcher import Matcher
## Extracting lalonde data: http://users.nber.org/~rdehejia/nswdata2.html
#%% rct data
rct_data = pd.read_stata('../data/raw/nsw_dw.dta')
#%% observational data
observational_data = pd.concat( #pd.read_stata('../data/raw/cps_controls3.dta')
    [pd.read_stata('../data/raw/psid_controls.dta'), pd.read_stata('../data/raw/cps_controls.dta')],
    ignore_index=True
)
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
treated = rct_data[rct_data.treat == 1].copy().drop(columns=['data_id'])
observational_control = observational_data.copy().drop(columns=['data_id'])
# %%
m = Matcher(treated, observational_control, yvar="treat", exclude=['re78'])
# %%
np.random.seed(666)
m.fit_scores(
    balance=True, 
    nmodels=100
)
# %%
m.predict_scores()
# %%
m.plot_scores()
# %%
m.tune_threshold(method='random')
# %%
m.match(method="min", nmatches=1, threshold=0.0004)
m.record_frequency()
# %%
m.assign_weight_vector()

# %%
m.matched_data.sort_values("match_id").head(6)
# %%
df = m.data
ps = df['scores']
y = df['re78']
z = df['treat']

ey1 = z*y/ps / sum(z/ps)
ey0 = (1-z)*y/(1-ps) / sum((1-z)/(1-ps))
ate = ey1.sum()-ey0.sum()
print("Causal Estimate is " + str(ate))
# %%
