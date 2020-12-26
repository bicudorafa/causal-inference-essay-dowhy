"""Importing Dependencies"""
import sys
import pytest
import pandas as pd
import numpy as np
from numpy.random import seed
# Env tests
try:
    sys.path.insert(1, './src')  # the type of path is string
    import stats_utils as su
except (ModuleNotFoundError, ImportError) as error_message:
    print("{} fileure".format(type(error_message)))

@pytest.mark.parametrize(
    "add_mu, add_sigma, add_n, p_exp_result, power_exp_result", [
        (0, 0, 0, False, False), # obvious total non rejected case
        (1, 2, 100, True, False), # Type 2 Error scenario
 # (0.1, 10, 50000000, False, True) # Type 1 Error scenario: will demand add research on it
    ]
)

def test_mean_ttest_analyzer(add_mu, add_sigma, add_n, p_exp_result, power_exp_result):
    """Test different hypotehsis tests for mean comparisons"""
    # random seed to generate the same ficitonal samples
    seed(666)
    population_1 = np.random.normal(1, 2, 200 + add_n)
    population_2 = np.random.normal(1 + add_mu, 2 + add_sigma, 50 + add_n)
    pvalue, power, _ = su.mean_ttest_analyzer(population_1, population_2)
     # reference p, power values -- no attachments to the widely used values on common tests
    p_assessment = pvalue < 0.05
    power_assessment = power > 0.6
    assert p_assessment == p_exp_result, f'the xp pvalue was {pvalue}'
    assert power_assessment == power_exp_result, f'the xp power was {power}'

@pytest.mark.parametrize(
    "add_no_feature_column, no_feature_list", [
        (False, None),
        (True, ['x_no_feature']),
    ]
)

def test_dataframe_ols_coeffs(add_no_feature_column, no_feature_list):
    """Test OLS applier to pandas dataframes based on columns selection"""
    # mock data generation - I wont use fixtures due to the impossibility of generating 2 artifacts
    const = np.random.normal(0, 1, 100)
    x_1 = np.random.normal(0, 1, 100)
    x_2 = np.random.normal(0, 1, 100)
    x_3 = np.random.normal(0, 1, 100)
    mock_ols_coeffs = np.array([0., 1., 2., 3.])
    target = np.dot(np.transpose([const, x_1, x_2, x_3]), mock_ols_coeffs)
    _mock_ols_df = pd.DataFrame(data={'x_1':x_1, 'x_2':x_2, 'x_3':x_3, 'y':target})
    if add_no_feature_column:
        _mock_ols_df = _mock_ols_df.assign(
            x_no_feature = [True if i % 2==0 else False for i in range(100)]
        )
    # proper testing
    target = 'y'
    coeffs = su.dataframe_ols_coeffs(_mock_ols_df, target, no_feature_list, print_stats=False)
    assert np.allclose(coeffs.values, mock_ols_coeffs, atol=0.3)
