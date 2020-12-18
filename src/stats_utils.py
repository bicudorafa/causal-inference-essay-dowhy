"""Importing Dependencies"""
import numpy as np
from numpy.random import seed
from scipy.stats import ttest_ind
from statsmodels.stats.power import tt_ind_solve_power

def mean_ttest_analyzer(sample_1, sample_2, alpha=0.05, return_abs_diff=True):
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
