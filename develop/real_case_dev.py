"""Importing Dependencies"""
# %% - paper: https://users.nber.org/~rdehejia/papers/matching.pdf
# python basics dependencies
import numpy as np
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tools.tools import categorical
from statsmodels.stats.power import tt_ind_solve_power
from numpy.random import seed
from scipy.stats import ttest_ind
import statsmodels.api as sm
import dowhy
import dowhy.api
from dowhy import CausalModel
## Extracting lalonde data: http://users.nber.org/~rdehejia/nswdata2.html
#%% rct data
rct_data = pd.read_stata('../data/raw/nsw_dw.dta')
# observational data - possivelmente, psid3 vai dar resultado estranho devido a ser menor do que tratamento
observational_data = pd.concat(
    [ 
        pd.read_stata('../data/raw/cps_controls.dta'),
        pd.read_stata('../data/raw/psid_controls.dta'),
    ],
    ignore_index=True
)
################################## EDA ######################################
# %% - substituir isto aqui por histograma de todas as variáveis com cores dioferente por sample
def prop_stacked_chart(subject_df, groups_var, cagorical_var_list):
    agg_df = pd.DataFrame()
    for cagorical_var in cagorical_var_list: # it might be possible to avoid this loop with a groupby
        temp_df = (
            subject_df
            .loc[:, [groups_var, cagorical_var]]
            .groupby([groups_var, cagorical_var])
            .agg(n_subjects=(groups_var, 'count'))
            .groupby(level=0) # it has a similar use case of partition by in sql
            .apply(lambda x: x / float(x.sum())) # then calculating the share by index 
            .reset_index()
            .rename(columns={cagorical_var:'categorical_value'})
            .assign(cagorical_var = cagorical_var)
        )
        agg_df = pd.concat([agg_df, temp_df])
        del temp_df
    fig=(
        agg_df
        .pipe(
            px.bar, x=groups_var, y='n_subjects',
            color='categorical_value', facet_col='cagorical_var', facet_col_wrap=len(agg_df.cagorical_var.unique())
        )
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(yaxis=dict(title_text='Percentage per Sample', tickformat=".2%")) # to include percent formatting in y axe
    fig.update_xaxes(title_text='')
    fig.update(layout_coloraxis_showscale=False)
    
    return fig

def boxplot_per_groups(subject_df, groups_var, continuous_var_list):
    melted_df = (
        subject_df
        .loc[:, ([groups_var] + continuous_var_list)]
        .pipe(pd.melt, id_vars=[groups_var], value_vars=continuous_var_list)
    )
    fig = px.box(
            melted_df, y="value", x=groups_var, 
            color=groups_var, facet_row="variable"
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text='')
    fig.update_yaxes(matches=None, title_text='')
    return fig
# %% 
fig = boxplot_per_groups(rct_data, 'treat', ['age', 'education', 're74', 're75'])
fig.show()
# %%
fig = prop_stacked_chart(rct_data, 'treat', ['black', 'hispanic', 'married', 'nodegree'])
fig.show()
# %%
fig = boxplot_per_groups(pd.concat([rct_data,observational_data]), 'data_id', ['age', 'education', 're74', 're75'])
fig.show()
# %%
fig = prop_stacked_chart(pd.concat([rct_data,observational_data]), 'data_id', ['black', 'hispanic', 'married', 'nodegree'])
fig.show()
# %%
############################################## XP Causal Inference Analysis ############################
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
    #lift=(mean_diff/np.nanmean(control))*100.0
    # power parameters
    std_diff=np.sqrt(np.nanvar(treatment)+np.nanvar(control))
    effect_size = mean_diff / std_diff
    treatment_size=(~np.isnan(treatment)).sum()
    size_ratio=(~np.isnan(control)).sum()/treatment_size
    power=tt_ind_solve_power(
        effect_size=effect_size, alpha=alpha, power=None
        ,ratio=size_ratio, alternative='two-sided',nobs1=treatment_size
)
    return p,mean_diff,power
# %%
ttest(rct_data[rct_data.treat == 0]['re78'], rct_data[rct_data.treat == 1]['re78'])
# %%
def ols_xp_dataframe(rct_data_to_reg, target, no_feature_list, print_stats=True):
    # %% why use OLS to XP data might improve its assessment: https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
    if no_feature_list == None:
        no_feature_list = [target]
    else: no_feature_list.append(target)
    Y = rct_data_to_reg[target].values
    # exogenous variables used by lalonde:  age, age squared, years of schooling, high school dropout status, and race
    X = rct_data_to_reg[[col for col in rct_data_to_reg.columns if col not in no_feature_list]].values
    treatment_position = rct_data_to_reg[[col for col in rct_data_to_reg.columns if col not in no_feature_list]].columns.get_loc('treat')
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    if print_stats:
        print(results.summary())
    treatment_coef = results.params[treatment_position + 1]
    return treatment_coef
original_treatment_coef = ols_xp_dataframe(rct_data, 're78', ['data_id'])
print(f'The Unbiased ATE is {original_treatment_coef}')
# %%
treated_df = rct_data[rct_data.treat == 1]
simple_ols_coef = []

for obs_data_group in observational_data.data_id.unique():
    to_reg = pd.concat([treated_df, observational_data[observational_data.data_id == obs_data_group]])
    treatment_coef = ols_xp_dataframe(to_reg, 're78', ['data_id'], False)
    simple_ols_coef.append(treatment_coef)
    print(f'The Biased ATE when using {obs_data_group} as control is {treatment_coef}')
methods_comparison_dic = {'simple_ols_coef':simple_ols_coef}
# %%
def att_causal_estimator(df, outcome, treatment, cofounders_list, method_name, view_model=False):
    causal_model = CausalModel(
        data = df, treatment=treatment,
        outcome=outcome, common_causes=cofounders_list
    )
    if view_model:
        causal_model.view_model(layout="dot")
    identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    causal_estimate = causal_model.estimate_effect(
        identified_estimand, method_name=method_name,
        target_units = 'att',#, confidence_intervals=True
    )
    return(causal_estimate.value)

# %%
#causal_data = rct_data.assign(treat = rct_data.treat.astype(bool))
outcome = 're78'
treatment = 'treat'
causal_features = [col for col in rct_data.columns if col not in ('treat','re78','data_id')]
method_list = ["backdoor.propensity_score_matching", "backdoor.propensity_score_stratification"]

for method in method_list:
    coef_list = []
    for obs_data_group in observational_data.data_id.unique():
        causal_data = pd.concat(
            [treated_df, observational_data[observational_data.data_id == obs_data_group]]
        ).assign(treat = lambda x:x.treat.astype(bool))
        treatment_coef = att_causal_estimator(causal_data, outcome, treatment, causal_features, method)
        coef_list.append(treatment_coef)
        print(f'The Estimated ATT when using {obs_data_group} as control is {treatment_coef}')
    methods_comparison_dic[method] = coef_list
# %%
coefs_df = pd.DataFrame(methods_comparison_dic,index=observational_data.data_id.unique()) / original_treatment_coef
coefs_df.loc['avg'] = coefs_df.mean()
coefs_df