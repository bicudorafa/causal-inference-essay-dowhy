"""Importing Dependencies"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.random import seed
from scipy.stats import ttest_ind
from statsmodels.stats.power import tt_ind_solve_power

def dowhy_quick_backdoor_estimator(
        dataframe, outcome, treatment, 
        cofounders_list, method_name,
        populaton_of_interest='ate', view_model=False
    ):
    """
    Make a quick statistical assessment for the mean of 2 different samples (hypothesis test based)
    :param dataframe: original dataframe in a subject level
    :param group_col: the name of the group column
    :param category_col: the name of the category_col column
    :returns group_share_per_category_df: df containing the % share each category has by group
    """
    pass