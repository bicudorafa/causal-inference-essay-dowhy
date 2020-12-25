"""Importing Dependencies"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import seed
from scipy.stats import ttest_ind
from statsmodels.stats.power import tt_ind_solve_power

def mean_ttest_analyzer(sample_1:np.array, sample_2:np.array, alpha=0.05, return_abs_diff=True):
    """
    Make a quick statistical assessment for the mean of 2 different samples (hypothesis test based)
    :param dataframe: original dataframe in a subject level
    :param group_col: the name of the group column
    :param category_col: the name of the category_col column
    :returns group_share_per_category_df: df containing the % share each category has by group
    """
    seed(666) # ensure results reproducibility
    _, pvalue = ttest_ind(a=sample_2, b=sample_1, nan_policy='omit')
    # power parameters
    std_diff = np.sqrt(np.nanvar(sample_2)+np.nanvar(sample_1))
    abs_mean_diff = np.nanmean(sample_2)-np.nanmean(sample_1)
    effect_size = abs_mean_diff / std_diff
    treatment_size = (~np.isnan(sample_2)).sum()
    size_ratio = (~np.isnan(sample_1)).sum()/treatment_size
    power = tt_ind_solve_power(
        effect_size=effect_size, alpha=alpha, power=None,
        ratio=size_ratio, alternative='two-sided',nobs1=treatment_size
    )
    if return_abs_diff:
        return pvalue, power, abs_mean_diff
    else:
        return pvalue, power

def dataframe_ols_coeffs(
        df_to_ols:pd.DataFrame, target_col:str, no_feature_col_list:list, print_stats=True
    ):
    """
    Executes an OLS regression on a dataframe a returns its coefficients (optionally, its summary)
    :param df_to_ols: original dataframe to be inputed on the OLS
    :param target_col: target varibale columns name
    :param no_feature_col_list: list containing any columns to be not included on the OLS
    :param print_stats: print OLS regression full output (or not)
    :returns coeffs_array: np.array containing coefficients
    """
    # consolidating columns that aren't regressors
    if no_feature_col_list is None:
        no_feature_col_list = [target_col]
    else:
        no_feature_col_list.append(target_col)
    features_final_list = [col for col in df_to_ols.columns if col not in no_feature_col_list]

    target_vector = df_to_ols[target_col].values
    variables_matrix = sm.add_constant(
        df_to_ols[features_final_list].values
    )
    model = sm.OLS(target_vector, variables_matrix)
    results = model.fit()
    if print_stats:
        print(results.summary())
    treatment_coef = results.params
    return treatment_coef
