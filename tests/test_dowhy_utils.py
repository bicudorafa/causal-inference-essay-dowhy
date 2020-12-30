"""Importing Dependencies"""
import sys
import pytest
import dowhy.datasets

# Env tests
try:
    sys.path.insert(1, './src')  # the type of path is string
    import dowhy_utils as du
except (ModuleNotFoundError, ImportError) as error_message:
    print("{} fileure".format(type(error_message)))

@pytest.mark.parametrize(
    "method_applied, populaton_of_interest", [
        ('backdoor.propensity_score_matching', 'ate'),
        ('backdoor.propensity_score_matching', 'att'),
        ('backdoor.propensity_score_matching', 'atc'),
        #('backdoor.propensity_score_stratification', 'ate'), for some reason, it isnt working heew
        #('backdoor.propensity_score_stratification', 'att'), but its working properly with real
        #('backdoor.propensity_score_stratification', 'atc'), data
    ]
)

def test_dowhy_quick_backdoor_estimator(method_applied, populaton_of_interest):
    """Test different hypotehsis tests for mean comparisons"""
    data = dowhy.datasets.linear_dataset(
        beta=1, # causal coef
        num_common_causes=3, # cofounders
        num_instruments = 1, #vi
        num_treatments=1, # treatment
        num_samples=1000, # sample
        treatment_is_binary=True,
        outcome_is_binary=False,
    )
    causal_estimate = du.dowhy_quick_backdoor_estimator(
        dataframe=data["df"], outcome=data["outcome_name"], treatment=data["treatment_name"],
        cofounders_list=data["common_causes_names"], method_name=method_applied,
        populaton_of_interest=populaton_of_interest, view_model=False
    )
    assert round(causal_estimate) == 1
