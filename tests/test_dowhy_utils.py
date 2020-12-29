"""Importing Dependencies"""
import sys
#import pytest
import dowhy.datasets

# Env tests
try:
    sys.path.insert(1, './src')  # the type of path is string
    import dowhy_utils as du
except (ModuleNotFoundError, ImportError) as error_message:
    print("{} fileure".format(type(error_message)))

def test_dowhy_quick_backdoor_estimator():
    """Test different hypotehsis tests for mean comparisons"""
    data = dowhy.datasets.linear_dataset(
        beta=10, # causal coef
        num_common_causes=3, # cofounders
        num_instruments = 1, #vi
        num_treatments=1, # treatment
        num_samples=100, # sample
        treatment_is_binary=True,
        outcome_is_binary=False,
    )

    method_applied = 'backdoor.propensity_score_matching'
    populaton_of_interest = 'ate'
    causal_estimate = du.dowhy_quick_backdoor_estimator(
        dataframe=data["df"], outcome=data["outcome_name"], treatment=data["treatment_name"],
        cofounders_list=data["common_causes_names"], method_name=method_applied,
        populaton_of_interest=populaton_of_interest, view_model=False
    )
    assert round(causal_estimate) == 10
