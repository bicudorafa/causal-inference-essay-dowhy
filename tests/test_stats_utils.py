"""Importing Dependencies"""
import sys
import pytest
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
